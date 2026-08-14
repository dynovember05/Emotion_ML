import os
import glob
import zipfile
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from data_loader import process_single_image

BASE_DIR = r"c:\Users\ldy34\Desktop\Face\video"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
BINARY_MODEL_PATH = r"c:\Users\ldy34\Desktop\Face\ML\emotion_resnet_best.pth"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 15
MAX_PER_CLASS = 120

CLASS_KEYWORDS = {
    "중립": 0,
    "불안": 1,
    "기쁨": 2,
    "당황": 3,
    "분노": 4,
    "상처": 5,
    "슬픔": 6,
}
IDX_TO_NAME = {v: k for k, v in CLASS_KEYWORDS.items()}


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
    def __init__(self, input_size, out_dim=2, hidden_dim=512):
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
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.classifier(x)


def pick_label_from_filename(filename: str):
    for kw, idx in CLASS_KEYWORDS.items():
        if kw in filename:
            return idx
    if "Neutral" in filename:
        return 0
    if "Anxiety" in filename or "Anxious" in filename:
        return 1
    return None


def collect_image_bytes(max_per_class=500):
    zip_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.zip")))
    per_class_counts = {idx: 0 for idx in CLASS_KEYWORDS.values()}
    byte_label_pairs = []

    random.shuffle(zip_files)

    for zp in zip_files:
        zname = os.path.basename(zp)
        if "[라벨]" in zname:
            continue
        label = pick_label_from_filename(zname)
        if label is None:
            continue
        if per_class_counts[label] >= max_per_class:
            continue

        remaining = max_per_class - per_class_counts[label]
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                members = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
                random.shuffle(members)
                take = members[:remaining]
                for m in take:
                    try:
                        byte_label_pairs.append((zf.read(m), label))
                        per_class_counts[label] += 1
                    except Exception:
                        continue
        except Exception:
            continue

        if all(c >= max_per_class for c in per_class_counts.values()):
            break

    return byte_label_pairs, per_class_counts


def extract_landmarks(pairs):
    X, y = [], []
    for sample in tqdm(pairs, total=len(pairs), desc="Landmark"):
        r = process_single_image(sample)
        if r is None:
            continue
        X.append(r[0])
        y.append(r[1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def evaluate_binary_model_on_subset(X_sub, y_sub):
    model = ResidualEmotionNet(input_size=X_sub.shape[1], out_dim=2).to(DEVICE)
    sd = torch.load(BINARY_MODEL_PATH, map_location=DEVICE)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_sub, dtype=torch.float32, device=DEVICE))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    return pred


def train_multiclass(X_train, y_train, X_val, y_val, out_dim=7):
    model = ResidualEmotionNet(input_size=X_train.shape[1], out_dim=out_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    best_acc = 0.0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        corr = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = torch.argmax(model(xb), dim=1)
                corr += (pred == yb).sum().item()
                total += yb.size(0)
        acc = corr / max(1, total)
        print(f"Epoch {ep+1}/{EPOCHS} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    print(f"Device: {DEVICE}")
    pairs, counts = collect_image_bytes(MAX_PER_CLASS)
    print("Collected per class:")
    for idx, cnt in sorted(counts.items()):
        print(f"  {IDX_TO_NAME[idx]}: {cnt}")
    print(f"Raw samples: {len(pairs)}")

    X, y = extract_landmarks(pairs)
    print(f"Valid landmark samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )

    multi_model = train_multiclass(X_train, y_train, X_test, y_test, out_dim=7)
    multi_model.eval()

    # Evaluate multiclass on anxious/neutral subset only
    mask_an_ne = np.isin(y_test, [0, 1])
    X_sub = X_test[mask_an_ne]
    y_sub = y_test[mask_an_ne]

    with torch.no_grad():
        logits_multi = multi_model(torch.tensor(X_sub, dtype=torch.float32, device=DEVICE))
        pred_multi = torch.argmax(logits_multi, dim=1).cpu().numpy()

    pred_bin = evaluate_binary_model_on_subset(X_sub, y_sub)

    acc_bin = accuracy_score(y_sub, pred_bin)
    acc_multi_subset = accuracy_score(y_sub, pred_multi)

    print("\n===== Neutral/Anxious Subset Comparison =====")
    print(f"Subset size: {len(y_sub)}")
    print(f"Binary model (emotion_resnet_best.pth) acc: {acc_bin:.4f}")
    print(f"7-class model acc on same subset: {acc_multi_subset:.4f}")

    print("\n[Binary model report]")
    print(classification_report(y_sub, pred_bin, target_names=["Neutral", "Anxious"]))

    print("\n[7-class model report on Neutral/Anxious subset]")
    print(classification_report(y_sub, pred_multi, target_names=["Neutral", "Anxious"]))


if __name__ == "__main__":
    main()
