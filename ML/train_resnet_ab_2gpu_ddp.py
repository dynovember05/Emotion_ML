import argparse
import glob
import json
import multiprocessing
import os
import random
import socket
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


CLASS_MAP_7 = {
    "중립": 0,
    "불안": 1,
    "기쁨": 2,
    "당황": 3,
    "분노": 4,
    "상처": 5,
    "슬픔": 6,
}
IDX_TO_NAME_7 = ["Neutral", "Anxious", "Joy", "Embarrassed", "Angry", "Hurt", "Sad"]
IDX_TO_NAME_2 = ["Neutral", "Anxious"]
BINARY_CLASS_IDS = {0, 1}

_FACE_MESH = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B pipeline on 2-GPU DDP: direct binary vs 7-class pretrain + binary finetune"
    )
    parser.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    parser.add_argument("--out-dir", default=r"c:\Users\ldy34\Desktop\Face\ML\experiments_ab_2gpu")
    parser.add_argument("--cache-path", default=r"c:\Users\ldy34\Desktop\Face\ML\cache_landmarks_7class.npz")
    parser.add_argument("--force-rebuild-cache", action="store_true")

    parser.add_argument("--max-per-zip", type=int, default=5000)
    parser.add_argument("--extract-workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--loader-workers", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--batch-size-per-gpu", type=int, default=256)

    parser.add_argument("--test-size", type=float, default=0.15, help="subject-group holdout ratio for final binary test")
    parser.add_argument("--val-size", type=float, default=0.15, help="subject-group holdout ratio inside train split")

    parser.add_argument("--epochs-binary-direct", type=int, default=120)
    parser.add_argument("--epochs-pretrain7", type=int, default=140)
    parser.add_argument("--epochs-binary-finetune", type=int, default=80)

    parser.add_argument("--lr-binary-direct", type=float, default=1e-3)
    parser.add_argument("--lr-pretrain7", type=float, default=8e-4)
    parser.add_argument("--lr-binary-finetune", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout-block", type=float, default=0.3)
    parser.add_argument("--dropout-head", type=float, default=0.2)

    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.002)

    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def ddp_setup(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        raise RuntimeError(f"Need at least {args.num_gpus} GPUs, but found {available_gpus}.")

    # `torchrun` rendezvous on this Windows build fails with libuv, so we force legacy TCPStore.
    os.environ.setdefault("USE_LIBUV", "0")

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if rank < 0 or world_size < 0 or local_rank < 0:
        raise RuntimeError(
            "Distributed env vars are missing. Launch with this script directly (it will spawn workers), "
            "or provide RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT."
        )
    if world_size != args.num_gpus:
        raise RuntimeError(f"WORLD_SIZE({world_size}) must match --num-gpus({args.num_gpus}).")

    torch.cuda.set_device(local_rank)
    backend = "gloo" if os.name == "nt" else ("nccl" if dist.is_nccl_available() else "gloo")

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    init_method = f"tcp://{master_addr}:{master_port}?use_libuv=0"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=60),
    )
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device, backend


def _spawn_worker(local_rank: int, args, master_addr: str, master_port: int):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["USE_LIBUV"] = "0"
    run_worker(args)


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def log_rank0(rank: int, message: str):
    if rank == 0:
        print(message, flush=True)


def infer_label_from_zip(zip_filename: str) -> Optional[int]:
    parts = zip_filename.split("_")
    if len(parts) < 2:
        return None
    label_token = parts[1]
    return CLASS_MAP_7.get(label_token)


def is_source_zip(zip_filename: str) -> bool:
    parts = zip_filename.split("_")
    return len(parts) >= 4


def subject_id_from_member(member: str) -> str:
    # Most files are ".../<subject_hash>_...jpg". We split by "_" and keep subject hash.
    base = os.path.basename(member)
    return base.split("_")[0] if "_" in base else base


def _init_facemesh_worker():
    global _FACE_MESH
    _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def _process_one_image(task):
    global _FACE_MESH
    zip_path, member, label, subject_id = task
    if _FACE_MESH is None:
        _init_facemesh_worker()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_bytes = np.frombuffer(zf.read(member), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None

        results = _FACE_MESH.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )
        coords -= coords[1]
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist

        return coords.flatten().astype(np.float32), int(label), str(subject_id)
    except Exception:
        return None


def build_tasks(base_dir: str, max_per_zip: int, seed: int) -> Tuple[List[Tuple[str, str, int, str]], Dict[int, int]]:
    train_dir = os.path.join(base_dir, "Training")
    zip_files = sorted(glob.glob(os.path.join(train_dir, "*.zip")))
    rng = random.Random(seed)

    tasks: List[Tuple[str, str, int, str]] = []
    class_counter = {i: 0 for i in range(7)}

    for zp in zip_files:
        zname = os.path.basename(zp)
        if not is_source_zip(zname):
            continue
        label = infer_label_from_zip(zname)
        if label is None:
            continue

        try:
            with zipfile.ZipFile(zp, "r") as zf:
                members = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
        except Exception:
            continue

        if len(members) > max_per_zip:
            members = rng.sample(members, max_per_zip)

        for member in members:
            subject_id = subject_id_from_member(member)
            tasks.append((zp, member, label, subject_id))
        class_counter[label] += len(members)

    return tasks, class_counter


def extract_or_load_cache(args, rank: int):
    cache_dir = os.path.dirname(args.cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if rank == 0:
        need_build = args.force_rebuild_cache or (not os.path.exists(args.cache_path))
        if need_build:
            tasks, class_counter = build_tasks(args.base_dir, args.max_per_zip, args.seed)
            print("[Stage 1/4] Landmark extraction", flush=True)
            print(f"[Info] Total tasks queued: {len(tasks):,}", flush=True)
            for idx in range(7):
                print(f"  - {IDX_TO_NAME_7[idx]}: {class_counter[idx]:,}", flush=True)

            features, labels, groups = [], [], []
            with ProcessPoolExecutor(
                max_workers=args.extract_workers,
                initializer=_init_facemesh_worker,
            ) as executor:
                for result in tqdm(
                    executor.map(_process_one_image, tasks, chunksize=32),
                    total=len(tasks),
                    desc="Landmarks",
                ):
                    if result is None:
                        continue
                    x, y, g = result
                    features.append(x)
                    labels.append(y)
                    groups.append(g)

            X = np.array(features, dtype=np.float32)
            y = np.array(labels, dtype=np.int64)
            g = np.array(groups, dtype="<U80")
            np.savez(args.cache_path, X=X, y=y, groups=g)

            meta = {
                "cache_path": args.cache_path,
                "num_samples": int(len(X)),
                "num_features": int(X.shape[1] if len(X) else 0),
                "class_counts": {IDX_TO_NAME_7[i]: int((y == i).sum()) for i in range(7)},
                "unique_subjects": int(len(set(g.tolist()))),
            }
            with open(os.path.splitext(args.cache_path)[0] + "_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[Info] Cache built: {args.cache_path}", flush=True)

    dist.barrier()
    cache = np.load(args.cache_path)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    groups = cache["groups"].astype("<U80")
    return X, y, groups


def stratified_group_split(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    group_label_counter: Dict[str, Counter] = defaultdict(Counter)

    for idx in indices:
        group_label_counter[str(groups[idx])][int(y[idx])] += 1

    label_to_groups: Dict[int, List[str]] = defaultdict(list)
    for group_id, counter in group_label_counter.items():
        major_label = counter.most_common(1)[0][0]
        label_to_groups[major_label].append(group_id)

    test_groups: Set[str] = set()
    for _, group_list in label_to_groups.items():
        local = group_list[:]
        rng.shuffle(local)
        n_total = len(local)
        n_test = int(round(n_total * test_size))
        if n_total > 1:
            n_test = max(1, n_test)
            n_test = min(n_total - 1, n_test)
        else:
            n_test = 0
        test_groups.update(local[:n_test])

    mask_test = np.array([str(groups[idx]) in test_groups for idx in indices], dtype=bool)
    test_idx = indices[mask_test]
    train_idx = indices[~mask_test]
    return train_idx, test_idx


def ensure_all_classes_present(train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray, required_classes: Set[int]):
    tr = set(np.unique(y[train_idx]).tolist())
    va = set(np.unique(y[val_idx]).tolist())
    return required_classes.issubset(tr) and required_classes.issubset(va)


def make_split_bundle(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
):
    all_indices = np.arange(len(y), dtype=np.int64)

    binary_mask = np.isin(y, [0, 1])
    bin_indices_all = all_indices[binary_mask]

    bin_trainval_idx, bin_test_idx = stratified_group_split(
        bin_indices_all, y, groups, test_size=test_size, seed=seed
    )

    required_binary = {0, 1}
    for offset in range(20):
        tr_idx, va_idx = stratified_group_split(
            bin_trainval_idx, y, groups, test_size=val_size, seed=seed + 100 + offset
        )
        if ensure_all_classes_present(tr_idx, va_idx, y, required_binary):
            bin_train_idx, bin_val_idx = tr_idx, va_idx
            break
    else:
        raise RuntimeError("Failed to create valid binary train/val split with both classes present.")

    test_groups = set(groups[bin_test_idx].tolist())
    pretrain_candidates = np.array(
        [idx for idx in all_indices if groups[idx] not in test_groups],
        dtype=np.int64,
    )

    required_7 = set(range(7))
    for offset in range(20):
        pre_tr, pre_va = stratified_group_split(
            pretrain_candidates, y, groups, test_size=val_size, seed=seed + 200 + offset
        )
        if ensure_all_classes_present(pre_tr, pre_va, y, required_7):
            pretrain_train_idx, pretrain_val_idx = pre_tr, pre_va
            break
    else:
        raise RuntimeError("Failed to create valid 7-class train/val split with all classes present.")

    return {
        "bin_train_idx": bin_train_idx,
        "bin_val_idx": bin_val_idx,
        "bin_test_idx": bin_test_idx,
        "pretrain_train_idx": pretrain_train_idx,
        "pretrain_val_idx": pretrain_val_idx,
        "test_groups": test_groups,
    }


class TensorIndexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = torch.from_numpy(X[indices].astype(np.float32))
        self.y = torch.from_numpy(y[indices].astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualEmotionNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_dim: int, block_dropout: float, head_dropout: float):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res_block1 = ResidualBlock(hidden_dim, block_dropout)
        self.res_block2 = ResidualBlock(hidden_dim, block_dropout)
        self.res_block3 = ResidualBlock(hidden_dim, block_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.classifier(x)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class StageConfig:
    name: str
    num_classes: int
    epochs: int
    lr: float
    output_ckpt: str
    input_size: int
    label_smoothing: float
    noise_std: float
    hidden_dim: int
    block_dropout: float
    head_dropout: float
    batch_size_per_gpu: int
    loader_workers: int
    weight_decay: float
    early_stop_patience: int
    early_stop_min_delta: float
    use_amp: bool


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
):
    model.eval()
    total_loss = 0.0
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            preds.append(pred.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(1, len(y_true))
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def make_dataloader(
    dataset: Dataset,
    rank: int,
    world_size: int,
    batch_size_per_gpu: int,
    loader_workers: int,
):
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    kwargs = {
        "batch_size": batch_size_per_gpu,
        "sampler": sampler,
        "num_workers": loader_workers,
        "pin_memory": True,
        "persistent_workers": loader_workers > 0,
    }
    if loader_workers > 0:
        kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **kwargs)
    return loader, sampler


def make_eval_loader(dataset: Dataset, batch_size: int, loader_workers: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": loader_workers,
        "pin_memory": True,
        "persistent_workers": loader_workers > 0,
    }
    if loader_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def load_matching_weights(model: nn.Module, ckpt_path: str):
    source = torch.load(ckpt_path, map_location="cpu")
    target = model.state_dict()
    matched = {}
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            matched[key] = value
    model.load_state_dict(matched, strict=False)
    return len(matched)


def train_stage_ddp(
    cfg: StageConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_weights: torch.Tensor,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
    seed: int,
    init_ckpt: Optional[str] = None,
):
    torch.manual_seed(seed + rank)

    model = ResidualEmotionNet(
        input_size=cfg.input_size,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        block_dropout=cfg.block_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)

    if init_ckpt is not None and rank == 0:
        matched = load_matching_weights(model, init_ckpt)
        print(f"[{cfg.name}] Loaded {matched} matching weights from {init_ckpt}", flush=True)
    if init_ckpt is not None:
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buf in model.buffers():
            dist.broadcast(buf.data, src=0)

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_loader, train_sampler = make_dataloader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size_per_gpu=cfg.batch_size_per_gpu,
        loader_workers=cfg.loader_workers,
    )
    val_loader = make_eval_loader(
        val_dataset,
        batch_size=cfg.batch_size_per_gpu,
        loader_workers=cfg.loader_workers,
    ) if rank == 0 else None

    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )
    criterion_eval = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.0,
    )
    optimizer = optim.AdamW(ddp_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    } if rank == 0 else None

    best_score = -1.0
    best_epoch = 0
    no_improve = 0

    os.makedirs(os.path.dirname(cfg.output_ckpt), exist_ok=True)

    for epoch in range(cfg.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        loss_sum_local = 0.0
        corr_local = 0.0
        total_local = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if cfg.noise_std > 0:
                xb = xb + (torch.randn_like(xb) * cfg.noise_std)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                logits = ddp_model(xb)
                loss = criterion_train(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = torch.argmax(logits, dim=1)
            batch_size = yb.size(0)
            loss_sum_local += loss.item() * batch_size
            corr_local += (pred == yb).sum().item()
            total_local += batch_size

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        train_stat = torch.tensor([loss_sum_local, corr_local, total_local], dtype=torch.float64, device=device)
        dist.all_reduce(train_stat, op=dist.ReduceOp.SUM)
        train_loss = (train_stat[0] / train_stat[2]).item()
        train_acc = (train_stat[1] / train_stat[2]).item()

        val_loss = 0.0
        val_acc = 0.0
        val_macro_f1 = 0.0
        improved = False
        stop_now = False

        if rank == 0 and val_loader is not None:
            eval_out = evaluate_on_loader(ddp_model.module, val_loader, device, criterion_eval, cfg.use_amp)
            val_loss = eval_out["loss"]
            val_acc = eval_out["acc"]
            val_macro_f1 = eval_out["macro_f1"]

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc))
            history["val_macro_f1"].append(float(val_macro_f1))
            history["lr"].append(float(lr_now))

            improved = (val_macro_f1 - best_score) > cfg.early_stop_min_delta
            if improved:
                best_score = val_macro_f1
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(ddp_model.module.state_dict(), cfg.output_ckpt)
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{cfg.name}] Epoch {epoch+1}/{cfg.epochs} | "
                    f"LR:{lr_now:.6f} | "
                    f"TLoss:{train_loss:.4f} TAcc:{train_acc:.4f} | "
                    f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f} VF1:{val_macro_f1:.4f} | "
                    f"BestF1:{best_score:.4f}@{best_epoch}",
                    flush=True,
                )

            if no_improve >= cfg.early_stop_patience:
                stop_now = True
                print(f"[{cfg.name}] Early stop triggered at epoch {epoch+1}", flush=True)

        stop_tensor = torch.tensor([1 if stop_now else 0], dtype=torch.int64, device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            break

    if rank == 0:
        history["best_epoch"] = int(best_epoch)
        history["best_val_macro_f1"] = float(best_score)
        history_path = os.path.splitext(cfg.output_ckpt)[0] + "_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[{cfg.name}] Saved history: {history_path}", flush=True)

    dist.barrier()
    return cfg.output_ckpt if rank == 0 else None


def evaluate_binary_checkpoint(
    ckpt_path: str,
    X: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    args,
    device: torch.device,
):
    dataset = TensorIndexDataset(X, y, test_idx)
    loader = make_eval_loader(dataset, batch_size=args.batch_size_per_gpu, loader_workers=args.loader_workers)

    model = ResidualEmotionNet(
        input_size=X.shape[1],
        num_classes=2,
        hidden_dim=args.hidden_dim,
        block_dropout=args.dropout_block,
        head_dropout=args.dropout_head,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.append(yb.numpy())
            preds.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, target_names=IDX_TO_NAME_2, digits=4),
    }


def run_worker(args):
    use_amp = args.amp or (torch.cuda.is_available() and not args.no_amp)
    rank, world_size, local_rank, device, backend = ddp_setup(args)
    try:
        log_rank0(rank, f"[Init] backend={backend} world_size={world_size} local_rank={local_rank}")
        if rank == 0:
            for i in range(torch.cuda.device_count()):
                print(f"[Init] GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
            print(f"[Init] AMP={use_amp}", flush=True)

        X, y, groups = extract_or_load_cache(args, rank)
        split_bundle = make_split_bundle(
            X=X,
            y=y,
            groups=groups,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
        )

        if rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            split_info = {
                "binary_train": int(len(split_bundle["bin_train_idx"])),
                "binary_val": int(len(split_bundle["bin_val_idx"])),
                "binary_test": int(len(split_bundle["bin_test_idx"])),
                "pretrain7_train": int(len(split_bundle["pretrain_train_idx"])),
                "pretrain7_val": int(len(split_bundle["pretrain_val_idx"])),
                "binary_test_groups": int(len(split_bundle["test_groups"])),
                "input_dim": int(X.shape[1]),
            }
            print("[Stage 2/4] Split summary", flush=True)
            for k, v in split_info.items():
                print(f"  - {k}: {v}", flush=True)
            with open(os.path.join(args.out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)

        dist.barrier()

        # A) Direct binary training
        log_rank0(rank, "[Stage 3/4-A] Direct binary training")
        bin_train_ds = TensorIndexDataset(X, y, split_bundle["bin_train_idx"])
        bin_val_ds = TensorIndexDataset(X, y, split_bundle["bin_val_idx"])
        bin_class_weights = compute_class_weights(y[split_bundle["bin_train_idx"]], num_classes=2)

        direct_ckpt = os.path.join(args.out_dir, "direct_binary_best.pth")
        direct_cfg = StageConfig(
            name="direct_binary",
            num_classes=2,
            epochs=args.epochs_binary_direct,
            lr=args.lr_binary_direct,
            output_ckpt=direct_ckpt,
            input_size=X.shape[1],
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            block_dropout=args.dropout_block,
            head_dropout=args.dropout_head,
            batch_size_per_gpu=args.batch_size_per_gpu,
            loader_workers=args.loader_workers,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            use_amp=use_amp,
        )
        train_stage_ddp(
            cfg=direct_cfg,
            train_dataset=bin_train_ds,
            val_dataset=bin_val_ds,
            class_weights=bin_class_weights,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            seed=args.seed + 10,
            init_ckpt=None,
        )

        # B) 7-class pretrain
        log_rank0(rank, "[Stage 3/4-B1] 7-class pretraining")
        pre_tr_ds = TensorIndexDataset(X, y, split_bundle["pretrain_train_idx"])
        pre_va_ds = TensorIndexDataset(X, y, split_bundle["pretrain_val_idx"])
        pre_weights = compute_class_weights(y[split_bundle["pretrain_train_idx"]], num_classes=7)

        pretrain_ckpt = os.path.join(args.out_dir, "pretrain7_best.pth")
        pre_cfg = StageConfig(
            name="pretrain7",
            num_classes=7,
            epochs=args.epochs_pretrain7,
            lr=args.lr_pretrain7,
            output_ckpt=pretrain_ckpt,
            input_size=X.shape[1],
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            block_dropout=args.dropout_block,
            head_dropout=args.dropout_head,
            batch_size_per_gpu=args.batch_size_per_gpu,
            loader_workers=args.loader_workers,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            use_amp=use_amp,
        )
        train_stage_ddp(
            cfg=pre_cfg,
            train_dataset=pre_tr_ds,
            val_dataset=pre_va_ds,
            class_weights=pre_weights,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            seed=args.seed + 20,
            init_ckpt=None,
        )

        # B2) Binary finetune from 7-class
        log_rank0(rank, "[Stage 3/4-B2] Binary finetuning from 7-class checkpoint")
        ft_ckpt = os.path.join(args.out_dir, "finetune_binary_best.pth")
        ft_cfg = StageConfig(
            name="finetune_binary_from7",
            num_classes=2,
            epochs=args.epochs_binary_finetune,
            lr=args.lr_binary_finetune,
            output_ckpt=ft_ckpt,
            input_size=X.shape[1],
            label_smoothing=args.label_smoothing,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            block_dropout=args.dropout_block,
            head_dropout=args.dropout_head,
            batch_size_per_gpu=args.batch_size_per_gpu,
            loader_workers=args.loader_workers,
            weight_decay=args.weight_decay,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            use_amp=use_amp,
        )
        train_stage_ddp(
            cfg=ft_cfg,
            train_dataset=bin_train_ds,
            val_dataset=bin_val_ds,
            class_weights=bin_class_weights,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            seed=args.seed + 30,
            init_ckpt=pretrain_ckpt,
        )

        # Final evaluation on fixed binary holdout test
        if rank == 0:
            log_rank0(rank, "[Stage 4/4] Final binary holdout evaluation")
            direct_metrics = evaluate_binary_checkpoint(
                ckpt_path=direct_ckpt,
                X=X,
                y=y,
                test_idx=split_bundle["bin_test_idx"],
                args=args,
                device=device,
            )
            ft_metrics = evaluate_binary_checkpoint(
                ckpt_path=ft_ckpt,
                X=X,
                y=y,
                test_idx=split_bundle["bin_test_idx"],
                args=args,
                device=device,
            )

            summary = {
                "config": vars(args),
                "split_sizes": {
                    "binary_train": int(len(split_bundle["bin_train_idx"])),
                    "binary_val": int(len(split_bundle["bin_val_idx"])),
                    "binary_test": int(len(split_bundle["bin_test_idx"])),
                    "pretrain7_train": int(len(split_bundle["pretrain_train_idx"])),
                    "pretrain7_val": int(len(split_bundle["pretrain_val_idx"])),
                },
                "results": {
                    "direct_binary": direct_metrics,
                    "finetune_binary_from7": ft_metrics,
                },
                "delta_finetune_minus_direct": {
                    "accuracy": float(ft_metrics["accuracy"] - direct_metrics["accuracy"]),
                    "balanced_accuracy": float(ft_metrics["balanced_accuracy"] - direct_metrics["balanced_accuracy"]),
                    "macro_f1": float(ft_metrics["macro_f1"] - direct_metrics["macro_f1"]),
                },
                "artifacts": {
                    "direct_ckpt": direct_ckpt,
                    "pretrain7_ckpt": pretrain_ckpt,
                    "finetune_ckpt": ft_ckpt,
                },
            }

            summary_path = os.path.join(args.out_dir, "ab_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print("\n=== Final Holdout Comparison (Neutral vs Anxious) ===", flush=True)
            print(f"Direct binary acc:      {direct_metrics['accuracy']:.4f}", flush=True)
            print(f"Finetune-from-7 acc:    {ft_metrics['accuracy']:.4f}", flush=True)
            print(f"Delta acc (FT - Direct): {summary['delta_finetune_minus_direct']['accuracy']:+.4f}", flush=True)
            print(f"\nSaved summary: {summary_path}", flush=True)
            print("\n[Direct Binary Report]\n" + direct_metrics["classification_report"], flush=True)
            print("\n[Finetune-from-7 Report]\n" + ft_metrics["classification_report"], flush=True)

        dist.barrier()
    finally:
        ddp_cleanup()


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(f"Need at least {args.num_gpus} GPUs, but found {torch.cuda.device_count()}.")

    # If launched by an external launcher that already sets rank envs, just run worker body.
    has_rank_env = all(
        k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    )
    if has_rank_env:
        os.environ.setdefault("USE_LIBUV", "0")
        run_worker(args)
        return

    master_addr = "127.0.0.1"
    master_port = find_free_port()
    print(
        f"[Launcher] Spawning {args.num_gpus} workers | master={master_addr}:{master_port} | USE_LIBUV=0",
        flush=True,
    )
    torch_mp.spawn(
        _spawn_worker,
        args=(args, master_addr, master_port),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
