import argparse
import json
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train_resnet_ab_2gpu_ddp import (
    IDX_TO_NAME_2,
    ResidualEmotionNet,
    StageConfig,
    TensorIndexDataset,
    compute_class_weights,
    ddp_cleanup,
    ddp_setup,
    extract_or_load_cache,
    find_free_port,
    load_matching_weights,
    make_eval_loader,
    make_split_bundle,
    seed_everything,
    train_stage_ddp,
)


COMPACT_LANDMARKS = [
    1,
    4,
    10,
    13,
    14,
    17,
    33,
    61,
    70,
    78,
    82,
    87,
    95,
    105,
    133,
    145,
    152,
    159,
    172,
    181,
    234,
    263,
    291,
    300,
    308,
    312,
    317,
    324,
    334,
    362,
    374,
    386,
    397,
    405,
    454,
]

WIDE_LANDMARKS = sorted(
    set(
        COMPACT_LANDMARKS
        + [
            40,
            46,
            52,
            55,
            65,
            107,
            276,
            282,
            285,
            295,
            336,
            468,
            473,
        ]
    )
)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Feature-engineered landmark experiment: derived geometric features, "
            "validation-threshold calibration, and probability outputs for ensembling."
        )
    )
    parser.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    parser.add_argument("--out-dir", default=r"c:\Users\ldy34\Desktop\Face\ML\experiments_feature_ensemble")
    parser.add_argument("--cache-path", default=r"c:\Users\ldy34\Desktop\Face\ML\cache_landmarks_7class.npz")
    parser.add_argument("--force-rebuild-cache", action="store_true")

    parser.add_argument("--max-per-zip", type=int, default=5000)
    parser.add_argument("--extract-workers", type=int, default=12)
    parser.add_argument("--loader-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=123, help="Training/model seed.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Subject split seed. Keep this fixed when building ensembles across training seeds.",
    )
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--batch-size-per-gpu", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--feature-mode", choices=["raw", "geom", "raw_geom"], default="raw_geom")
    parser.add_argument("--feature-preset", choices=["compact", "wide"], default="compact")
    parser.add_argument(
        "--scaler-fit",
        choices=["binary_train", "train_union"],
        default="binary_train",
        help="Fit standardization using binary train only, or binary train plus 7-class pretrain train.",
    )
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--epochs-binary-direct", type=int, default=140)
    parser.add_argument("--epochs-pretrain7", type=int, default=160)
    parser.add_argument("--epochs-binary-finetune", type=int, default=100)
    parser.add_argument("--train-finetune", action="store_true", help="Also run 7-class pretrain + binary finetune.")

    parser.add_argument("--lr-binary-direct", type=float, default=2e-4)
    parser.add_argument("--lr-pretrain7", type=float, default=8e-4)
    parser.add_argument("--lr-binary-finetune", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout-block", type=float, default=0.3)
    parser.add_argument("--dropout-head", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Feature noise after standardization. Use smaller values if --no-standardize is set.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=35)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--threshold-recall-target", type=float, default=0.85)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def selected_landmarks(preset: str):
    return WIDE_LANDMARKS if preset == "wide" else COMPACT_LANDMARKS


def pairwise_distances(coords: np.ndarray, ids):
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    values = [
        np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1, keepdims=True).astype(np.float32)
        for a, b in pairs
    ]
    return np.concatenate(values, axis=1), pairs


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6):
    return (num / np.maximum(den, eps)).astype(np.float32)


def distance(coords: np.ndarray, a: int, b: int):
    return np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=1, keepdims=True).astype(np.float32)


def build_geometric_features(x_raw: np.ndarray, preset: str) -> Tuple[np.ndarray, Dict]:
    coords = x_raw.reshape(len(x_raw), -1, 3).astype(np.float32)
    ids = selected_landmarks(preset)

    mean = coords.mean(axis=1)
    std = coords.std(axis=1)
    amin = coords.min(axis=1)
    amax = coords.max(axis=1)
    arange = amax - amin
    l2 = np.linalg.norm(coords, axis=2)
    l2_stats = np.stack([l2.mean(axis=1), l2.std(axis=1), l2.min(axis=1), l2.max(axis=1)], axis=1)

    pairwise, pairs = pairwise_distances(coords, ids)

    mouth_open = distance(coords, 13, 14)
    mouth_width = distance(coords, 61, 291)
    left_eye_open = distance(coords, 159, 145)
    left_eye_width = distance(coords, 33, 133)
    right_eye_open = distance(coords, 386, 374)
    right_eye_width = distance(coords, 362, 263)
    face_width = distance(coords, 234, 454)
    face_height = distance(coords, 10, 152)
    brow_eye_left = distance(coords, 105, 159)
    brow_eye_right = distance(coords, 334, 386)
    nose_mouth = distance(coords, 1, 13)

    ratios = np.concatenate(
        [
            safe_ratio(mouth_open, mouth_width),
            safe_ratio(left_eye_open, left_eye_width),
            safe_ratio(right_eye_open, right_eye_width),
            safe_ratio(face_height, face_width),
            safe_ratio(brow_eye_left, face_height),
            safe_ratio(brow_eye_right, face_height),
            safe_ratio(nose_mouth, face_height),
        ],
        axis=1,
    )

    symmetry = np.concatenate(
        [
            left_eye_open - right_eye_open,
            left_eye_width - right_eye_width,
            brow_eye_left - brow_eye_right,
            distance(coords, 61, 1) - distance(coords, 291, 1),
            distance(coords, 33, 1) - distance(coords, 263, 1),
            distance(coords, 234, 1) - distance(coords, 454, 1),
        ],
        axis=1,
    ).astype(np.float32)

    geom = np.concatenate(
        [
            mean,
            std,
            amin,
            amax,
            arange,
            l2_stats,
            pairwise,
            ratios,
            symmetry,
        ],
        axis=1,
    ).astype(np.float32)

    meta = {
        "preset": preset,
        "selected_landmarks": ids,
        "pairwise_count": len(pairs),
        "stats_dim": int(mean.shape[1] + std.shape[1] + amin.shape[1] + amax.shape[1] + arange.shape[1] + l2_stats.shape[1]),
        "ratio_dim": int(ratios.shape[1]),
        "symmetry_dim": int(symmetry.shape[1]),
        "geom_dim": int(geom.shape[1]),
    }
    return geom, meta


def make_feature_matrix(x_raw: np.ndarray, mode: str, preset: str) -> Tuple[np.ndarray, Dict]:
    geom, geom_meta = build_geometric_features(x_raw, preset)
    if mode == "raw":
        x = x_raw.astype(np.float32)
    elif mode == "geom":
        x = geom
    elif mode == "raw_geom":
        x = np.concatenate([x_raw.astype(np.float32), geom], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    meta = {
        "feature_mode": mode,
        "raw_dim": int(x_raw.shape[1]),
        "output_dim": int(x.shape[1]),
        "geom": geom_meta,
    }
    return x, meta


def fit_scaler(x: np.ndarray, fit_idx: np.ndarray) -> FeatureScaler:
    mean = x[fit_idx].mean(axis=0).astype(np.float32)
    std = x[fit_idx].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureScaler(mean=mean, std=std)


def apply_scaler(x: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    return ((x - scaler.mean) / scaler.std).astype(np.float32)


def softmax_logits(model, loader, device, use_amp: bool):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)
            probs.append(prob.float().cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def metrics_at_threshold(y_true: np.ndarray, prob_anxious: np.ndarray, threshold: float) -> Dict:
    y_pred = (prob_anxious >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "neutral_recall": float(recall_score(y_true, y_pred, pos_label=0)),
        "anxious_recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "anxious_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def find_thresholds(y_true: np.ndarray, probs: np.ndarray, recall_target: float) -> Dict:
    prob_anxious = probs[:, 1]
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), prob_anxious]))
    rows = [metrics_at_threshold(y_true, prob_anxious, float(t)) for t in grid]
    best_macro = max(rows, key=lambda r: (r["macro_f1"], r["balanced_accuracy"]))
    best_balanced = max(rows, key=lambda r: (r["balanced_accuracy"], r["macro_f1"]))
    recall_candidates = [r for r in rows if r["anxious_recall"] >= recall_target]
    if recall_candidates:
        best_recall_target = max(recall_candidates, key=lambda r: (r["macro_f1"], r["accuracy"]))
    else:
        best_recall_target = max(rows, key=lambda r: r["anxious_recall"])
    return {
        "default_0p5": metrics_at_threshold(y_true, prob_anxious, 0.5),
        "best_macro_f1": best_macro,
        "best_balanced_accuracy": best_balanced,
        f"best_macro_with_anxious_recall_ge_{recall_target:.2f}": best_recall_target,
    }


def eval_binary_checkpoint_with_thresholds(
    args,
    ckpt_path: str,
    x: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    stage_name: str,
):
    use_amp = args.amp or (torch.cuda.is_available() and not args.no_amp)
    model = ResidualEmotionNet(
        input_size=x.shape[1],
        num_classes=2,
        hidden_dim=args.hidden_dim,
        block_dropout=args.dropout_block,
        head_dropout=args.dropout_head,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    val_ds = TensorIndexDataset(x, y, val_idx)
    test_ds = TensorIndexDataset(x, y, test_idx)
    val_loader = make_eval_loader(val_ds, batch_size=args.batch_size_per_gpu, loader_workers=args.loader_workers)
    test_loader = make_eval_loader(test_ds, batch_size=args.batch_size_per_gpu, loader_workers=args.loader_workers)

    y_val, p_val = softmax_logits(model, val_loader, device, use_amp)
    y_test, p_test = softmax_logits(model, test_loader, device, use_amp)
    val_thresholds = find_thresholds(y_val, p_val, args.threshold_recall_target)

    test_by_threshold = {}
    for name, val_metric in val_thresholds.items():
        test_by_threshold[name] = metrics_at_threshold(y_test, p_test[:, 1], val_metric["threshold"])

    np.savez(
        Path(args.out_dir) / f"{stage_name}_val_probs.npz",
        indices=val_idx,
        y_true=y_val,
        probs=p_val,
    )
    np.savez(
        Path(args.out_dir) / f"{stage_name}_test_probs.npz",
        indices=test_idx,
        y_true=y_test,
        probs=p_test,
    )

    default_pred = np.argmax(p_test, axis=1)
    return {
        "checkpoint": ckpt_path,
        "val_thresholds": val_thresholds,
        "test_by_val_threshold": test_by_threshold,
        "classification_report_default_0p5": classification_report(
            y_test, default_pred, target_names=IDX_TO_NAME_2, digits=4
        ),
    }


def write_bundle(
    args,
    stage_name: str,
    ckpt_path: str,
    scaler: Optional[FeatureScaler],
    feature_meta: Dict,
    threshold_summary: Dict,
):
    state = torch.load(ckpt_path, map_location="cpu")
    bundle = {
        "state_dict": state,
        "model_class": "ResidualEmotionNet",
        "model_kwargs": {
            "input_size": int(feature_meta["output_dim"]),
            "num_classes": 2,
            "hidden_dim": int(args.hidden_dim),
            "block_dropout": float(args.dropout_block),
            "head_dropout": float(args.dropout_head),
        },
        "feature_meta": feature_meta,
        "standardize": not args.no_standardize,
        "scaler": None
        if scaler is None
        else {
            "mean": scaler.mean,
            "std": scaler.std,
        },
        "threshold_summary": threshold_summary,
        "class_names": IDX_TO_NAME_2,
    }
    torch.save(bundle, Path(args.out_dir) / f"{stage_name}_bundle.pth")


def log0(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def run_worker(args):
    use_amp = args.amp or (torch.cuda.is_available() and not args.no_amp)
    rank, world_size, local_rank, device, backend = ddp_setup(args)
    try:
        log0(
            rank,
            f"[Init] feature_mode={args.feature_mode} preset={args.feature_preset} "
            f"backend={backend} world_size={world_size} seed={args.seed} split_seed={args.split_seed}",
        )

        x_raw, y, groups = extract_or_load_cache(args, rank)
        split_bundle = make_split_bundle(
            X=x_raw,
            y=y,
            groups=groups,
            seed=args.split_seed,
            test_size=args.test_size,
            val_size=args.val_size,
        )

        x_feat, feature_meta = make_feature_matrix(x_raw, args.feature_mode, args.feature_preset)
        if args.scaler_fit == "train_union":
            fit_idx = np.unique(
                np.concatenate([split_bundle["bin_train_idx"], split_bundle["pretrain_train_idx"]])
            ).astype(np.int64)
        else:
            fit_idx = split_bundle["bin_train_idx"]

        scaler = None
        if not args.no_standardize:
            scaler = fit_scaler(x_feat, fit_idx)
            x_feat = apply_scaler(x_feat, scaler)

        if rank == 0:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                Path(args.out_dir) / "feature_scaler.npz",
                mean=np.array([], dtype=np.float32) if scaler is None else scaler.mean,
                std=np.array([], dtype=np.float32) if scaler is None else scaler.std,
            )
            run_info = {
                "config": vars(args),
                "backend": backend,
                "split_sizes": {
                    "binary_train": int(len(split_bundle["bin_train_idx"])),
                    "binary_val": int(len(split_bundle["bin_val_idx"])),
                    "binary_test": int(len(split_bundle["bin_test_idx"])),
                    "pretrain7_train": int(len(split_bundle["pretrain_train_idx"])),
                    "pretrain7_val": int(len(split_bundle["pretrain_val_idx"])),
                },
                "feature_meta": feature_meta,
                "standardize": not args.no_standardize,
                "scaler_fit": args.scaler_fit,
            }
            with open(Path(args.out_dir) / "run_info.json", "w", encoding="utf-8") as f:
                json.dump(run_info, f, ensure_ascii=False, indent=2)
            print("[Feature]", json.dumps(feature_meta, ensure_ascii=False), flush=True)
        dist.barrier()

        bin_train_ds = TensorIndexDataset(x_feat, y, split_bundle["bin_train_idx"])
        bin_val_ds = TensorIndexDataset(x_feat, y, split_bundle["bin_val_idx"])
        bin_weights = compute_class_weights(y[split_bundle["bin_train_idx"]], num_classes=2)

        direct_ckpt = str(Path(args.out_dir) / "direct_binary_best.pth")
        direct_cfg = StageConfig(
            name="direct_binary",
            num_classes=2,
            epochs=args.epochs_binary_direct,
            lr=args.lr_binary_direct,
            output_ckpt=direct_ckpt,
            input_size=x_feat.shape[1],
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
        log0(rank, "[Stage] Direct binary with engineered features")
        train_stage_ddp(
            cfg=direct_cfg,
            train_dataset=bin_train_ds,
            val_dataset=bin_val_ds,
            class_weights=bin_weights,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            seed=args.seed + 10,
        )

        pretrain_ckpt = str(Path(args.out_dir) / "pretrain7_best.pth")
        finetune_ckpt = str(Path(args.out_dir) / "finetune_binary_best.pth")
        if args.train_finetune:
            pre_tr_ds = TensorIndexDataset(x_feat, y, split_bundle["pretrain_train_idx"])
            pre_va_ds = TensorIndexDataset(x_feat, y, split_bundle["pretrain_val_idx"])
            pre_weights = compute_class_weights(y[split_bundle["pretrain_train_idx"]], num_classes=7)
            pre_cfg = StageConfig(
                name="pretrain7",
                num_classes=7,
                epochs=args.epochs_pretrain7,
                lr=args.lr_pretrain7,
                output_ckpt=pretrain_ckpt,
                input_size=x_feat.shape[1],
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
            log0(rank, "[Stage] 7-class pretrain with engineered features")
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
            )

            ft_cfg = StageConfig(
                name="finetune_binary_from7",
                num_classes=2,
                epochs=args.epochs_binary_finetune,
                lr=args.lr_binary_finetune,
                output_ckpt=finetune_ckpt,
                input_size=x_feat.shape[1],
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
            log0(rank, "[Stage] Binary finetune from 7-class engineered features")
            train_stage_ddp(
                cfg=ft_cfg,
                train_dataset=bin_train_ds,
                val_dataset=bin_val_ds,
                class_weights=bin_weights,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                device=device,
                seed=args.seed + 30,
                init_ckpt=pretrain_ckpt,
            )

        if rank == 0:
            results = {}
            direct_eval = eval_binary_checkpoint_with_thresholds(
                args=args,
                ckpt_path=direct_ckpt,
                x=x_feat,
                y=y,
                val_idx=split_bundle["bin_val_idx"],
                test_idx=split_bundle["bin_test_idx"],
                device=device,
                stage_name="direct_binary",
            )
            results["direct_binary"] = direct_eval
            write_bundle(
                args=args,
                stage_name="direct_binary",
                ckpt_path=direct_ckpt,
                scaler=scaler,
                feature_meta=feature_meta,
                threshold_summary=direct_eval["val_thresholds"],
            )

            if args.train_finetune:
                ft_eval = eval_binary_checkpoint_with_thresholds(
                    args=args,
                    ckpt_path=finetune_ckpt,
                    x=x_feat,
                    y=y,
                    val_idx=split_bundle["bin_val_idx"],
                    test_idx=split_bundle["bin_test_idx"],
                    device=device,
                    stage_name="finetune_binary",
                )
                results["finetune_binary_from7"] = ft_eval
                write_bundle(
                    args=args,
                    stage_name="finetune_binary",
                    ckpt_path=finetune_ckpt,
                    scaler=scaler,
                    feature_meta=feature_meta,
                    threshold_summary=ft_eval["val_thresholds"],
                )

            summary = {
                "config": vars(args),
                "feature_meta": feature_meta,
                "standardize": not args.no_standardize,
                "split_sizes": {
                    "binary_train": int(len(split_bundle["bin_train_idx"])),
                    "binary_val": int(len(split_bundle["bin_val_idx"])),
                    "binary_test": int(len(split_bundle["bin_test_idx"])),
                    "pretrain7_train": int(len(split_bundle["pretrain_train_idx"])),
                    "pretrain7_val": int(len(split_bundle["pretrain_val_idx"])),
                },
                "results": results,
                "artifacts": {
                    "direct_ckpt": direct_ckpt,
                    "direct_bundle": str(Path(args.out_dir) / "direct_binary_bundle.pth"),
                    "pretrain7_ckpt": pretrain_ckpt if args.train_finetune else None,
                    "finetune_ckpt": finetune_ckpt if args.train_finetune else None,
                    "finetune_bundle": str(Path(args.out_dir) / "finetune_binary_bundle.pth")
                    if args.train_finetune
                    else None,
                },
            }
            with open(Path(args.out_dir) / "feature_ab_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

            best_default = results["direct_binary"]["test_by_val_threshold"]["default_0p5"]
            best_tuned = results["direct_binary"]["test_by_val_threshold"]["best_macro_f1"]
            print(
                f"[Done] direct default acc={best_default['accuracy']:.4f} f1={best_default['macro_f1']:.4f} | "
                f"threshold-tuned acc={best_tuned['accuracy']:.4f} f1={best_tuned['macro_f1']:.4f}",
                flush=True,
            )
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(f"Need at least {args.num_gpus} GPUs, but found {torch.cuda.device_count()}.")

    has_rank_env = all(
        k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    )
    if has_rank_env:
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
