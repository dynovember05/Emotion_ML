import argparse
import glob
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CLASS_MAP = {"중립": 0, "불안": 1, "기쁨": 2, "당황": 3, "분노": 4, "상처": 5, "슬픔": 6}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=r"c:\Users\ldy34\Desktop\Face\video")
    p.add_argument("--bin-model", default=r"c:\Users\ldy34\Desktop\Face\ML\emotion_resnet_best.pth")
    p.add_argument("--multi-model", default=r"c:\Users\ldy34\Desktop\Face\ML\emotion_resnet_7class_best.pth")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-per-zip", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def infer_label(name):
    if "[라벨]" in name:
        return None
    for k, v in CLASS_MAP.items():
        if k in name:
            return v
    if "Neutral" in name:
        return 0
    if "Anxiety" in name or "Anxious" in name:
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
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
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
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualEmotionNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        hidden_dim = 512
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU())
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.classifier(x)


def load_state(path, input_size, num_classes, device):
    model = ResidualEmotionNet(input_size, num_classes).to(device)
    sd = torch.load(path, map_location=device)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zip_files = sorted(glob.glob(os.path.join(args.base_dir, "Training", "*.zip")))
    tasks = []
    for zp in zip_files:
        name = os.path.basename(zp)
        label = infer_label(name)
        if label not in (0, 1):
            continue
        with zipfile.ZipFile(zp, "r") as zf:
            imgs = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))][: args.max_per_zip]
            tasks.extend((zp, m, label) for m in imgs)

    X, y = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(process_image, tasks, chunksize=32), total=len(tasks), desc="Extract"):
            if r is None:
                continue
            X.append(r[0]); y.append(r[1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed, stratify=y)

    bin_model = load_state(args.bin_model, X_test.shape[1], 2, device)
    multi_model = load_state(args.multi_model, X_test.shape[1], 7, device)

    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_bin = torch.argmax(bin_model(xb), dim=1).cpu().numpy()
        pred_multi = torch.argmax(multi_model(xb), dim=1).cpu().numpy()

    # 7-class 모델의 0/1 외 예측은 오답 처리
    pred_multi_binary = np.where(np.isin(pred_multi, [0, 1]), pred_multi, -1)

    acc_bin = accuracy_score(y_test, pred_bin)
    acc_multi = (pred_multi_binary == y_test).mean()

    print("=== Neutral/Anxious Comparison ===")
    print(f"Test size: {len(y_test)}")
    print(f"2-class (emotion_resnet_best.pth) acc: {acc_bin:.4f}")
    print(f"7-class model acc on same set: {acc_multi:.4f}")

    print("\n[2-class report]")
    print(classification_report(y_test, pred_bin, target_names=["Neutral", "Anxious"], digits=4))


if __name__ == "__main__":
    main()
