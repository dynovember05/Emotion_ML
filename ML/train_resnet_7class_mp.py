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
    p = argparse.ArgumentParser(description="Train 7-class ResNet-style emotion model with multiprocessing")
    p.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--max-per-zip", type=int, default=5000, help="max images to read per zip")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="emotion_resnet_7class_best.pth")
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
            nose_tip = coords[1]
            coords = coords - nose_tip
            max_dist = np.max(np.linalg.norm(coords, axis=1))
            if max_dist > 0:
                coords = coords / max_dist
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

    train_dir = os.path.join(args.base_dir, "Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Device: {device}")
    print(f"[Info] Train dir: {train_dir}")

    zip_files = sorted(glob.glob(os.path.join(train_dir, "*.zip")))
    print(f"[Info] Found zip files: {len(zip_files)}")

    tasks = []
    class_counter = {i: 0 for i in range(7)}

    for zp in zip_files:
        zname = os.path.basename(zp)
        label = infer_label(zname)
        if label is None:
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                image_files = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
                take = image_files[: min(len(image_files), args.max_per_zip)]
                for member in take:
                    tasks.append((zp, member, label))
                class_counter[label] += len(take)
        except Exception as e:
            print(f"[Warn] failed to read {zname}: {e}")

    print("[Info] Collected samples per class:")
    for idx, count in class_counter.items():
        print(f"  - {IDX_TO_NAME[idx]}: {count}")
    print(f"[Info] Total queued samples: {len(tasks)}")

    X, y = [], []
    print(f"[Info] Extracting landmarks with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_image, tasks, chunksize=32), total=len(tasks), desc="Landmarks"):
            if res is None:
                continue
            X.append(res[0])
            y.append(res[1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[Info] Valid samples: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ResidualEmotionNet(input_size=X_train.shape[1], num_classes=7).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[Info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "lr": []}

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        train_correct, train_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            pred_train = torch.argmax(out, dim=1)
            train_correct += (pred_train == yb).sum().item()
            train_total += yb.size(0)

        model.eval()
        correct, total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_val = criterion(out, yb)
                pred = torch.argmax(out, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                val_loss_sum += loss_val.item()

        test_acc = correct / total if total else 0.0
        train_loss = run_loss / max(1, len(train_loader))
        train_acc = train_correct / train_total if train_total else 0.0
        val_loss = val_loss_sum / max(1, len(test_loader))
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        history["lr"].append(float(cur_lr))

        if test_acc > best_acc:
            best_acc = test_acc
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in save_model.state_dict().items()}
            torch.save(best_state, args.out)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] LR: {cur_lr:.6f} | "
                f"TLoss: {train_loss:.4f} TAcc: {train_acc:.4f} | "
                f"VLoss: {val_loss:.4f} VAcc: {test_acc:.4f} | Best: {best_acc:.4f}"
            )

    print(f"\n[Done] Best accuracy: {best_acc:.4f}")
    print(f"[Done] Saved: {args.out}")
    history_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[Done] History saved: {history_path}")

    if best_state is not None:
        model_to_report = ResidualEmotionNet(input_size=X_train.shape[1], num_classes=7).to(device)
        model_to_report.load_state_dict(best_state)
        model_to_report.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                preds.extend(torch.argmax(model_to_report(xb), dim=1).cpu().numpy())
        print("\n[Classification Report]")
        print(classification_report(y_test, preds, target_names=IDX_TO_NAME, digits=4))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
