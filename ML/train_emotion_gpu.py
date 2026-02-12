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

# ==========================================
# 1. 설정 및 초기화
# ==========================================

# GPU 설정 (CUDA 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")
if torch.cuda.device_count() > 1:
    print(f"[Info] Using {torch.cuda.device_count()} GPUs (DataParallel)!")
else:
    print(f"[Info] GPU Count: {torch.cuda.device_count()}")

# MediaPipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,       # 정지 이미지 모드 (정확도 높음)
    max_num_faces=1,              # 얼굴 하나만 감지
    refine_landmarks=True,        # 눈동자 등 정교한 랜드마크 포함
    min_detection_confidence=0.5
)

# 데이터 경로 설정
BASE_DIR = r"c:\Users\ldy34\Desktop\Face\video"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")

# 학습할 감정 및 파일명 매핑 (Label: 0=중립, 1=불안)
TARGET_EMOTIONS = {
    "Neutral": {"file": "[원천]EMOIMG_중립_TRAIN_01.zip", "label": 0},
    "Anxious": {"file": "[원천]EMOIMG_불안_TRAIN_01.zip", "label": 1}
}

# 학습 데이터 샘플 수 (테스트용으로 적게 설정, 필요시 늘리세요)
SAMPLE_COUNT = 1000 

# ==========================================
# 2. 데이터 추출 함수 (ZIP 파일 처리)
# ==========================================
def extract_landmarks_from_zip(zip_path, label, max_samples=500):
    """
    ZIP 파일 내부의 이미지에서 랜드마크를 추출합니다.
    """
    data = []
    labels = []
    
    if not os.path.exists(zip_path):
        print(f"[Error] File not found: {zip_path}")
        return [], []

    print(f"[Processing] Reading {os.path.basename(zip_path)}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # 이미지 파일 목록만 필터링 (대소문자 무시)
            file_list = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 파일이 너무 많으면 셔플하거나 앞에서부터 자름
            if len(file_list) > max_samples:
                # np.random.shuffle(file_list) # 랜덤하게 뽑으려면 주석 해제 (단, 실행 때마다 다를 수 있음)
                target_files = file_list[:max_samples]
            else:
                target_files = file_list
            
            for img_name in tqdm(target_files, desc=f"Extracting Landmarks"):
                try:
                    # 이미지 바이트 읽기
                    with z.open(img_name) as f:
                        img_data = f.read()
                        img_array = np.frombuffer(img_data, np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if image is None: continue
                        
                        # 랜드마크 추출 (BGR -> RGB 변환 필수)
                        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                        if results.multi_face_landmarks:
                            # 478개 랜드마크의 x, y, z 좌표 모두 사용 (Flatten)
                            landmarks = []
                            for lm in results.multi_face_landmarks[0].landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])
                            
                            data.append(landmarks)
                            labels.append(label)
                except Exception as e:
                    # 개별 이미지 처리 실패 시 무시하고 진행
                    continue
    except Exception as e:
        print(f"[Error] Failed to open zip file: {e}")
                
    return data, labels

# ==========================================
# 3. 데이터 로드 및 전처리
# ==========================================
X = []
y = []

# 각 감정별 데이터 추출
for emotion, info in TARGET_EMOTIONS.items():
    zip_path = os.path.join(TRAIN_DIR, info["file"])
    d, l = extract_landmarks_from_zip(zip_path, label=info["label"], max_samples=SAMPLE_COUNT)
    
    if len(d) == 0:
        print(f"[Warning] No data extracted for {emotion}. Check file path.")
    else:
        print(f"  -> Extracted {len(d)} samples for {emotion}")
        X.extend(d)
        y.extend(l)

X = np.array(X)
y = np.array(y)

print(f"\n[Info] Total Dataset Size: {len(X)}")

if len(X) == 0:
    print("[Error] No data found. Exiting...")
    exit()

# 데이터 분할 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# PyTorch Tensor 변환
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# DataLoader 생성
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ==========================================
# 4. 모델 정의 (PyTorch MLP)
# ==========================================
class EmotionMLP(nn.Module):
    def __init__(self, input_size):
        super(EmotionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output: Neutral(0), Anxious(1)
        )
    
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
model = EmotionMLP(input_size).to(device)

# Multi-GPU 사용 설정
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. 모델 학습
# ==========================================
print("\n[Training] Starting Training Process...")
epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    # 10 에포크마다 로그 출력
    if (epoch+1) % 10 == 0:
        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {acc:.2f}%")

print("[Training] Complete.")

# ==========================================
# 6. 모델 평가 및 저장
# ==========================================
model.eval()
with torch.no_grad():
    X_test_device = X_test_t.to(device)
    y_test_device = y_test_t.to(device)
    
    outputs = model(X_test_device)
    _, predicted = torch.max(outputs, 1)
    
    # CPU로 가져와서 평가
    y_pred = predicted.cpu().numpy()
    y_true = y_test_device.cpu().numpy()

print("\n" + "="*30)
print("       Final Evaluation       ")
print("="*30)
print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Neutral', 'Anxious']))

# 모델 저장
model_save_path = "emotion_model_gpu.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\n[Info] Model saved to {model_save_path}")
