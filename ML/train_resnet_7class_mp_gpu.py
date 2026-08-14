import argparse
import glob
import json
import multiprocessing
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Train 7-class ResNet with multiprocessing + multi-GPU")
    p.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--max-per-zip", type=int, default=5000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="emotion_resnet_7class_best.pth")

    p.add_argument("--best-metric", choices=["val_acc", "val_loss"], default="val_loss")
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    p.add_argument("--amp", action="store_true", help="use mixed precision")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    return p.parse_args()


CLASS_MAP = {
    "중립": 0,
    "불안": 1,
    "기쁨": 2,
    "당황": 3,
    "분노": 4,
    "상처": 5,
    "슬픔": 6,
}
IDX_TO_NAME = ["Neutral", "Anxious", "Joy", "Embarrassed", "Angry", "Hurt", "Sad"]


def infer_label(zip_filename):
    if "[라벨]" in zip_filename:
        return None
    for k, v in CLASS_MAP.items():
        if k in zip_filename:
            return v
    if "Neutral" in zip_filename:
        return 0
    if "Anxiety" in zip_filename or "Anxious" in zip_filename:
        return 1
    return None


def process_image(task):
    zip_path, member, label = task
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_bytes = np.frombuffer(zf.read(member), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as fm:
            results = fm.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None

            coords = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark], dtype=np.float32)
            coords -= coords[1]
            max_dist = np.max(np.linalg.norm(coords, axis=1))
            if max_dist > 0:
                coords /= max_dist
            return coords.flatten().astype(np.float32), label
    except Exception:
        return None


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
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
    def __init__(self, input_size, num_classes=7, hidden_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.classifier(x)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.base_dir, "Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    amp_enabled = args.amp or (torch.cuda.is_available() and not args.no_amp)

    print(f"[Info] Device: {device} | GPUs: {gpu_count}")
    if gpu_count > 0:
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"[Info] AMP: {amp_enabled}")
    print(f"[Info] CPU workers(requested): {args.workers}")

    zip_files = sorted(glob.glob(os.path.join(train_dir, "*.zip")))
    tasks, class_counter = [], {i: 0 for i in range(7)}

    for zp in zip_files:
        label = infer_label(os.path.basename(zp))
        if label is None:
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                images = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
                take = images[: min(len(images), args.max_per_zip)]
                tasks.extend((zp, m, label) for m in take)
                class_counter[label] += len(take)
        except Exception as e:
            print(f"[Warn] {zp}: {e}")

    print("[Info] Collected samples per class:")
    for i, c in class_counter.items():
        print(f"  - {IDX_TO_NAME[i]}: {c}")

    X, y = [], []
    print(f"[Stage 1/2] CPU multiprocessing landmark extraction start")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_image, tasks, chunksize=32), total=len(tasks), desc="Landmarks"):
            if res is not None:
                X.append(res[0])
                y.append(res[1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[Info] Valid samples: {len(X):,}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Stage 1/2] done. Valid samples: {len(X):,}")
    print(f"[Stage 2/2] GPU training start")
    model = ResidualEmotionNet(input_size=X_train.shape[1], num_classes=7).to(device)
    if gpu_count > 1:
        print(f"[Info] DataParallel enabled on {gpu_count} GPUs")
        model = nn.DataParallel(model)
    else:
        print("[Info] Single-GPU mode")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "lr": []}
    best_state, best_epoch = None, 0

    if args.best_metric == "val_acc":
        best_score = -1.0
    else:
        best_score = float("inf")

    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        run_loss, tr_correct, tr_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            tr_correct += (out.argmax(dim=1) == yb).sum().item()
            tr_total += yb.size(0)

        model.eval()
        val_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(xb)
                    vloss = criterion(out, yb)
                val_loss_sum += vloss.item()
                va_correct += (out.argmax(dim=1) == yb).sum().item()
                va_total += yb.size(0)

        train_loss = run_loss / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))
        train_acc = tr_correct / max(1, tr_total)
        val_acc = va_correct / max(1, va_total)
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(val_acc))
        history["lr"].append(float(cur_lr))

        if args.best_metric == "val_acc":
            improved = (val_acc - best_score) > args.early_stop_min_delta
            current_score = val_acc
        else:
            improved = (best_score - val_loss) > args.early_stop_min_delta
            current_score = val_loss

        if improved:
            best_score = current_score
            best_epoch = epoch + 1
            no_improve = 0
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in save_model.state_dict().items()}
            torch.save(best_state, args.out)
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] LR:{cur_lr:.6f} | "
                f"TLoss:{train_loss:.4f} TAcc:{train_acc:.4f} | "
                f"VLoss:{val_loss:.4f} VAcc:{val_acc:.4f} | "
                f"Best({args.best_metric}):{best_score:.4f} @ {best_epoch}"
            )

        if no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] no improve for {no_improve} epochs. stop at epoch {epoch+1}")
            break

    history_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved model: {args.out}")
    print(f"[Done] Saved history: {history_path}")

    if best_state is not None:
        report_model = ResidualEmotionNet(input_size=X_train.shape[1], num_classes=7).to(device)
        report_model.load_state_dict(best_state)
        report_model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                preds.extend(torch.argmax(report_model(xb), dim=1).cpu().numpy())
        print("\n[Classification Report @ Best]")
        print(classification_report(y_val, preds, target_names=IDX_TO_NAME, digits=4))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
