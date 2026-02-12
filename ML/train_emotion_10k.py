import zipfile
import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ============================
# 설정 및 초기화
# ============================
# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")
if torch.cuda.device_count() > 1:
    print(f"[Info] Detected {torch.cuda.device_count()} GPUs! Using DataParallel.")

# MediaPipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 데이터 경로
BASE_DIR = r"c:\Users\ldy34\Desktop\Face\video"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TARGET_FILES = {
    "Neutral": {"filename": "[원천]EMOIMG_중립_TRAIN_01.zip", "label": 0},
    "Anxious": {"filename": "[원천]EMOIMG_불안_TRAIN_01.zip", "label": 1}
}
MAX_SAMPLES = 10000

# ============================
# 데이터 추출 함수
# ============================
def extract_landmarks_from_zip(zip_path, label, max_samples=10000):
    data = []
    labels = []
    
    if not os.path.exists(zip_path):
        print(f"[Error] File not found: {zip_path}")
        return [], []

    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files = len(file_list)
        use_count = min(total_files, max_samples)
        
        print(f"\n[{os.path.basename(zip_path)}]")
        print(f"  - Total files: {total_files:,}")
        print(f"  - Extracting:  {use_count:,}")
        
        target_files = file_list[:use_count]
        
        for img_name in tqdm(target_files, desc="Processing"):
            try:
                with z.open(img_name) as f:
                    img_data = f.read()
                    img_array = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is None: continue
                    
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        landmarks = []
                        for lm in results.multi_face_landmarks[0].landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        data.append(landmarks)
                        labels.append(label)
            except: continue
                
    return data, labels

# ============================
# 실행
# ============================
X, y = [], []
for emotion, info in TARGET_FILES.items():
    path = os.path.join(TRAIN_DIR, info['filename'])
    d, l = extract_landmarks_from_zip(path, info['label'], max_samples=MAX_SAMPLES)
    X.extend(d); y.extend(l)

X = np.array(X)
y = np.array(y)
print(f"\n[Info] Total Dataset Size: {len(X):,} samples")

if len(X) == 0:
    print("No data loaded."); exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)

# 모델 정의
class EmotionMLP(nn.Module):
    def __init__(self, input_size):
        super(EmotionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.network(x)

model = EmotionMLP(X_train.shape[1]).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 50
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

print(f"\n[Training] Starting {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    r_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        r_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += yb.size(0); correct += (pred == yb).sum().item()
    
    train_loss = r_loss / len(train_loader)
    train_acc = correct / total
    
    model.eval()
    t_loss, correct_t, total_t = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            t_loss += criterion(out, yb).item()
            _, pred = torch.max(out, 1)
            total_t += yb.size(0); correct_t += (pred == yb).sum().item()
    
    test_loss = t_loss / len(test_loader)
    test_acc = correct_t / total_t
    
    history['train_loss'].append(train_loss); history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc); history['test_acc'].append(test_acc)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] T-Loss: {train_loss:.4f} T-Acc: {train_acc:.4f} | V-Loss: {test_loss:.4f} V-Acc: {test_acc:.4f}")

# 그래프 저장
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['test_acc'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.savefig('result_graph.png')
print("\n[Info] Graph saved to result_graph.png")

torch.save(model.state_dict(), 'emotion_model_10k.pth')
print("[Info] Model saved.")
