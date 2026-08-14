import argparse
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

from train_resnet_ab_2gpu_ddp import (
    IDX_TO_NAME_2,
    compute_class_weights,
    ddp_cleanup,
    ddp_setup,
    extract_or_load_cache,
    find_free_port,
    load_matching_weights,
    make_split_bundle,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Architecture search for landmark-based emotion models")
    parser.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    parser.add_argument("--out-dir", default=r"c:\Users\ldy34\Desktop\Face\ML\experiments_arch_2gpu")
    parser.add_argument("--cache-path", default=r"c:\Users\ldy34\Desktop\Face\ML\cache_landmarks_7class.npz")
    parser.add_argument("--force-rebuild-cache", action="store_true")

    parser.add_argument("--arch", choices=["token_mixer", "hybrid_mixer", "transformer_lite"], default="hybrid_mixer")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size-per-gpu", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--extract-workers", type=int, default=12)
    parser.add_argument("--max-per-zip", type=int, default=5000)

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--epochs-binary-direct", type=int, default=140)
    parser.add_argument("--epochs-pretrain7", type=int, default=180)
    parser.add_argument("--epochs-binary-finetune", type=int, default=120)
    parser.add_argument("--lr-binary-direct", type=float, default=2e-4)
    parser.add_argument("--lr-pretrain7", type=float, default=6e-4)
    parser.add_argument("--lr-binary-finetune", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--mixer-layers", type=int, default=6)
    parser.add_argument("--token-mlp-dim", type=int, default=128)
    parser.add_argument("--channel-mlp-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--model-dropout", type=float, default=0.15)
    parser.add_argument("--global-branch-dim", type=int, default=128)

    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.0005)
    parser.add_argument("--early-stop-patience", type=int, default=35)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class LandmarkDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x = torch.from_numpy(x[indices].astype(np.float32))
        self.y = torch.from_numpy(y[indices].astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MlpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LandmarkMixerBlock(nn.Module):
    def __init__(self, num_landmarks: int, d_model: int, token_dim: int, channel_dim: int, dropout: float):
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = MlpBlock(num_landmarks, token_dim, num_landmarks, dropout)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = MlpBlock(d_model, channel_dim, d_model, dropout)

    def forward(self, x):
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class LandmarkMixerNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        arch: str,
        d_model: int,
        mixer_layers: int,
        token_dim: int,
        channel_dim: int,
        num_heads: int,
        transformer_layers: int,
        dropout: float,
        global_branch_dim: int,
    ):
        super().__init__()
        if input_size % 3 != 0:
            raise ValueError(f"input_size must be divisible by 3, got {input_size}")
        self.num_landmarks = input_size // 3
        self.arch = arch

        self.coord_proj = nn.Linear(3, d_model)
        self.landmark_embed = nn.Parameter(torch.zeros(1, self.num_landmarks, d_model))
        nn.init.trunc_normal_(self.landmark_embed, std=0.02)

        if arch in {"token_mixer", "hybrid_mixer"}:
            self.encoder = nn.Sequential(
                *[
                    LandmarkMixerBlock(self.num_landmarks, d_model, token_dim, channel_dim, dropout)
                    for _ in range(mixer_layers)
                ]
            )
        elif arch == "transformer_lite":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=channel_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        self.pool = AttentionPool(d_model)
        self.stats_proj = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.global_branch = None
        classifier_in = d_model + 64
        if arch == "hybrid_mixer":
            self.global_branch = nn.Sequential(
                nn.Linear(input_size, global_branch_dim),
                nn.LayerNorm(global_branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(global_branch_dim, global_branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_in += global_branch_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        coords = x.view(x.size(0), self.num_landmarks, 3)
        tokens = self.coord_proj(coords) + self.landmark_embed
        tokens = self.encoder(tokens)
        pooled = self.pool(tokens)

        mean = coords.mean(dim=1)
        std = coords.std(dim=1)
        amin = coords.amin(dim=1)
        amax = coords.amax(dim=1)
        stats = self.stats_proj(torch.cat([mean, std, amin, amax], dim=1))

        features = [pooled, stats]
        if self.global_branch is not None:
            features.append(self.global_branch(x))
        return self.classifier(torch.cat(features, dim=1))


@dataclass
class StageConfig:
    name: str
    num_classes: int
    epochs: int
    lr: float
    output_ckpt: str


def make_model(args, input_size: int, num_classes: int):
    return LandmarkMixerNet(
        input_size=input_size,
        num_classes=num_classes,
        arch=args.arch,
        d_model=args.d_model,
        mixer_layers=args.mixer_layers,
        token_dim=args.token_mlp_dim,
        channel_dim=args.channel_mlp_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.model_dropout,
        global_branch_dim=args.global_branch_dim,
    )


def make_train_loader(dataset, rank, world_size, batch_size, workers):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs), sampler


def make_eval_loader(dataset, batch_size, workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def evaluate(model, loader, device, criterion, use_amp):
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
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            ys.append(yb.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "loss": float(total_loss / max(1, len(y_true))),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_stage(args, cfg, train_dataset, val_dataset, class_weights, input_size, rank, world_size, local_rank, device, init_ckpt=None):
    use_amp = args.amp or (torch.cuda.is_available() and not args.no_amp)
    model = make_model(args, input_size, cfg.num_classes).to(device)

    if init_ckpt is not None and rank == 0:
        matched = load_matching_weights(model, init_ckpt)
        print(f"[{cfg.name}] Loaded {matched} matching tensors from {init_ckpt}", flush=True)
    if init_ckpt is not None:
        dist.barrier()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buf in model.buffers():
            dist.broadcast(buf.data, src=0)

    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    train_loader, train_sampler = make_train_loader(
        train_dataset, rank, world_size, args.batch_size_per_gpu, args.loader_workers
    )
    val_loader = make_eval_loader(val_dataset, args.batch_size_per_gpu, args.loader_workers) if rank == 0 else None

    criterion_train = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(ddp.parameters(), lr=cfg.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(cfg.epochs):
        ddp.train()
        train_sampler.set_epoch(epoch)
        local = torch.zeros(3, dtype=torch.float64, device=device)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if args.noise_std > 0:
                xb = xb + torch.randn_like(xb) * args.noise_std

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = ddp(xb)
                loss = criterion_train(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = yb.size(0)
            local[0] += loss.item() * bs
            local[1] += (torch.argmax(logits, dim=1) == yb).sum().item()
            local[2] += bs

        scheduler.step()
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        train_loss = (local[0] / local[2]).item()
        train_acc = (local[1] / local[2]).item()

        stop = False
        if rank == 0:
            val = evaluate(ddp.module, val_loader, device, criterion_eval, use_amp)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val["loss"]))
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val["acc"]))
            history["val_macro_f1"].append(float(val["macro_f1"]))
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))

            improved = (val["macro_f1"] - best_f1) > args.early_stop_min_delta
            if improved:
                best_f1 = val["macro_f1"]
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(ddp.module.state_dict(), cfg.output_ckpt)
            else:
                no_improve += 1

            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(
                    f"[{cfg.name}] Epoch {epoch+1}/{cfg.epochs} | "
                    f"TLoss:{train_loss:.4f} TAcc:{train_acc:.4f} | "
                    f"VLoss:{val['loss']:.4f} VAcc:{val['acc']:.4f} VF1:{val['macro_f1']:.4f} | "
                    f"BestF1:{best_f1:.4f}@{best_epoch}",
                    flush=True,
                )

            if no_improve >= args.early_stop_patience:
                print(f"[{cfg.name}] Early stop at epoch {epoch+1}", flush=True)
                stop = True

        stop_tensor = torch.tensor([1 if stop else 0], dtype=torch.int64, device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            break

    if rank == 0:
        history["best_epoch"] = int(best_epoch)
        history["best_val_macro_f1"] = float(best_f1)
        hist_path = os.path.splitext(cfg.output_ckpt)[0] + "_history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    dist.barrier()


def eval_binary_checkpoint(args, ckpt_path, x, y, test_idx, device):
    dataset = LandmarkDataset(x, y, test_idx)
    loader = make_eval_loader(dataset, args.batch_size_per_gpu, args.loader_workers)
    model = make_model(args, x.shape[1], 2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            pred = torch.argmax(model(xb), dim=1).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, target_names=IDX_TO_NAME_2, digits=4),
    }


def run_worker(args):
    rank, world_size, local_rank, device, backend = ddp_setup(args)
    try:
        if rank == 0:
            print(f"[Init] arch={args.arch} backend={backend} world_size={world_size}", flush=True)
            for i in range(torch.cuda.device_count()):
                print(f"[Init] GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

        x, y, groups = extract_or_load_cache(args, rank)
        splits = make_split_bundle(x, y, groups, args.seed, args.test_size, args.val_size)
        input_size = int(x.shape[1])

        if rank == 0:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            split_info = {
                "binary_train": int(len(splits["bin_train_idx"])),
                "binary_val": int(len(splits["bin_val_idx"])),
                "binary_test": int(len(splits["bin_test_idx"])),
                "pretrain7_train": int(len(splits["pretrain_train_idx"])),
                "pretrain7_val": int(len(splits["pretrain_val_idx"])),
                "input_dim": input_size,
            }
            with open(Path(args.out_dir) / "split_summary.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)
            print("[Split]", split_info, flush=True)
        dist.barrier()

        bin_train = LandmarkDataset(x, y, splits["bin_train_idx"])
        bin_val = LandmarkDataset(x, y, splits["bin_val_idx"])
        pre_train = LandmarkDataset(x, y, splits["pretrain_train_idx"])
        pre_val = LandmarkDataset(x, y, splits["pretrain_val_idx"])

        bin_weights = compute_class_weights(y[splits["bin_train_idx"]], 2)
        pre_weights = compute_class_weights(y[splits["pretrain_train_idx"]], 7)

        direct_ckpt = str(Path(args.out_dir) / "direct_binary_best.pth")
        pre_ckpt = str(Path(args.out_dir) / "pretrain7_best.pth")
        ft_ckpt = str(Path(args.out_dir) / "finetune_binary_best.pth")

        train_stage(
            args,
            StageConfig("direct_binary", 2, args.epochs_binary_direct, args.lr_binary_direct, direct_ckpt),
            bin_train,
            bin_val,
            bin_weights,
            input_size,
            rank,
            world_size,
            local_rank,
            device,
        )
        train_stage(
            args,
            StageConfig("pretrain7", 7, args.epochs_pretrain7, args.lr_pretrain7, pre_ckpt),
            pre_train,
            pre_val,
            pre_weights,
            input_size,
            rank,
            world_size,
            local_rank,
            device,
        )
        train_stage(
            args,
            StageConfig("finetune_binary_from7", 2, args.epochs_binary_finetune, args.lr_binary_finetune, ft_ckpt),
            bin_train,
            bin_val,
            bin_weights,
            input_size,
            rank,
            world_size,
            local_rank,
            device,
            init_ckpt=pre_ckpt,
        )

        if rank == 0:
            direct = eval_binary_checkpoint(args, direct_ckpt, x, y, splits["bin_test_idx"], device)
            ft = eval_binary_checkpoint(args, ft_ckpt, x, y, splits["bin_test_idx"], device)
            summary = {
                "config": vars(args),
                "results": {"direct_binary": direct, "finetune_binary_from7": ft},
                "delta_finetune_minus_direct": {
                    "accuracy": ft["accuracy"] - direct["accuracy"],
                    "balanced_accuracy": ft["balanced_accuracy"] - direct["balanced_accuracy"],
                    "macro_f1": ft["macro_f1"] - direct["macro_f1"],
                },
                "artifacts": {"direct_ckpt": direct_ckpt, "pretrain7_ckpt": pre_ckpt, "finetune_ckpt": ft_ckpt},
            }
            with open(Path(args.out_dir) / "ab_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[Done] Direct acc={direct['accuracy']:.4f} | Finetune acc={ft['accuracy']:.4f}", flush=True)
        dist.barrier()
    finally:
        ddp_cleanup()


def _spawn_worker(local_rank: int, args, master_addr: str, master_port: int):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["USE_LIBUV"] = "0"
    run_worker(args)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
        os.environ.setdefault("USE_LIBUV", "0")
        run_worker(args)
        return

    master_addr = "127.0.0.1"
    master_port = find_free_port()
    print(f"[Launcher] Spawning {args.num_gpus} workers | master={master_addr}:{master_port}", flush=True)
    torch_mp.spawn(_spawn_worker, args=(args, master_addr, master_port), nprocs=args.num_gpus, join=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
