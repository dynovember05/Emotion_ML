
import os
import torch
import torch.nn as nn
import numpy as np
import glob
import zipfile
import cv2
import mediapipe as mp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 외부 모듈 (process_single_image) 사용
from data_loader import process_single_image

# ============================
# 설정
# ============================
BASE_DIR = r"c:\Users\ldy34\Desktop\Face\video"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
MODEL_PATH = "emotion_resnet_best.pth"
BATCH_SIZE = 128
NUM_WORKERS = os.cpu_count() or 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 모델 정의 (학습 때와 동일해야 함)
# ============================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class ResidualEmotionNet(nn.Module):
    def __init__(self, input_size, hidden_dim=512):
        super(ResidualEmotionNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.classifier(x)

# ============================
# 데이터 로드 (평가용으로 소량만 로드)
# ============================
def load_eval_data(max_samples=2000):
    zip_files = glob.glob(os.path.join(TRAIN_DIR, "*.zip"))
    print(f"[Info] Found {len(zip_files)} zip files.")
    
    data_list = []
    
    # 각 ZIP 파일에서 파일 목록 수집
    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        if "중립" in filename or "Neutral" in filename: label = 0
        elif "불안" in filename or "Anxiety" in filename: label = 1
        else: continue
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.png'))]
            # 테스트용으로 파일 앞부분 2000개만 사용 (빠른 평가 위해)
            # (주의: 학습 때 안 쓴 뒷부분 데이터를 쓰면 더 좋지만 복잡해지므로, 
            #  전체에서 일부를 로드해서 Test Set으로 분리하는 방식 사용)
            use_count = min(len(file_list), max_samples)
            for img_name in file_list[:use_count]:
                with z.open(img_name) as f:
                    data_list.append((f.read(), label))
    
    print(f"[Info] Loading {len(data_list)} images for evaluation...")
    
    X, y = [], []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_image, data_list), total=len(data_list), desc="Extracting Landmarks"))
        for res in results:
            if res is not None:
                X.append(res[0])
                y.append(res[1])
                
    return np.array(X), np.array(y)

# ============================
# 메인 실행
# ============================
if __name__ == '__main__':
    # 1. 데이터 로드 및 분할
    X, y = load_eval_data(max_samples=2000) # 평가용 데이터 로드
    
    # 학습 때와 동일하게 Split하되, Test Set만 사용
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[Info] Test Set Size: {len(X_test)}")

    # 2. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file '{MODEL_PATH}' not found!")
        exit()
        
    model = ResidualEmotionNet(input_size=X_test.shape[1]).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("[Info] Model loaded successfully.")

    # 3. 평가 실행
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, y_b in tqdm(test_loader, desc="Evaluating"):
            X_b = X_b.to(DEVICE)
            outputs = model(X_b)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())
            
    # 4. 결과 출력
    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "="*40)
    print(f"      Final Test Accuracy: {acc:.4f}")
    print("="*40)
    
    print("\n[Detailed Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=['Neutral', 'Anxious']))
    
    # 5. Confusion Matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Anxious'], yticklabels=['Neutral', 'Anxious'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\n[Info] Confusion Matrix saved as 'confusion_matrix.png'")
