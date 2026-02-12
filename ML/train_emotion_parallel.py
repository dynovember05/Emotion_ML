import os
import glob
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import mediapipe as mp
import cv2
from tqdm import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 경로 설정 (프로젝트 구조에 맞게 수정)
BASE_PATH = r"C:\Users\ldy34\Desktop\Face"
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "video", "Training")

# 하이퍼파라미터
BATCH_SIZE = 128    # 메모리가 넉넉하므로 크게 설정
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_WORKERS = os.cpu_count()  # CPU 코어 수만큼 병렬 처리

# 미디어파이프 설정
mp_face_mesh = mp.solutions.face_mesh

# ==========================================
# 2. 데이터 전처리 함수 (Process)
# ==========================================
def process_image(args):
    """
    워커 프로세스에서 실행될 함수입니다.
    ZIP 파일 내의 이미지를 읽어 랜드마크를 추출합니다.
    """
    zip_path, filename, label = args
    
    try:
        # ZIP 파일 열기 (압축 해제 없이 메모리 로드)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_bytes = np.frombuffer(zf.read(filename), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return None

        # MediaPipe FaceMesh 적용
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                return None

            # 랜드마크 추출 (478개 포인트)
            landmarks = results.multi_face_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]) # (478, 3)

            # --- [데이터 정규화: Centering & Scaling] ---
            # 1. 코 끝(인덱스 1)을 원점으로 이동 (Centering)
            nose_tip = coords[1]
            coords -= nose_tip

            # 2. 절대값의 최대 크기로 나누어 스케일링 (Scaling)
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords /= max_val
            
            # 1차원 벡터로 변환 (478 * 3 = 1434)
            return (coords.flatten().astype(np.float32), label)

    except Exception as e:
        # 손상된 이미지 등은 무시
        return None

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class FaceEmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 4. 모델 정의 (MLP)
# ==========================================
class EmotionMLP(nn.Module):
    def __init__(self):
        super(EmotionMLP, self).__init__()
        self.input_size = 478 * 3 # 1434
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.output = nn.Linear(128, 2) # [Neutral, Anxiety]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

# ==========================================
# 5. 메인 실행 코드
# ==========================================
if __name__ == '__main__':
    # 윈도우 멀티프로세싱 이슈 방지
    import multiprocessing
    multiprocessing.freeze_support()

    print(f"[Start] 학습 데이터 로드 및 전처리 시작 (CPUs: {NUM_WORKERS})")
    print(f"데이터 경로: {TRAIN_DATA_PATH}")

    # 1. 파일 리스트 수집
    data_list = [] # (zip_path, filename, label)
    
    # glob을 사용하여 ZIP 파일 찾기 (파일명에 '중립' 또는 '불안'이 포함된 파일)
    zip_files = glob.glob(os.path.join(TRAIN_DATA_PATH, "*.zip"))
    
    print(f"발견된 ZIP 파일: {len(zip_files)}개")

    for zip_path in zip_files:
        filename_only = os.path.basename(zip_path)
        
        # 라벨링: 중립=0, 불안=1
        if "중립" in filename_only or "Neutral" in filename_only:
            label = 0
            label_name = "중립(Neutral)"
        elif "불안" in filename_only or "Anxiety" in filename_only:
            label = 1
            label_name = "불안(Anxiety)"
        else:
            continue # 해당 감정이 아니면 스킵

        print(f"   -> Reading ZIP: {filename_only} ({label_name})")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 이미지 파일만 필터링 (jpg, png)
                image_files = [f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                # 리스트에 추가
                for img_file in image_files:
                    data_list.append((zip_path, img_file, label))
        except Exception as e:
            print(f"Error reading zip {zip_path}: {e}")

    print(f"총 이미지 파일 수: {len(data_list)}장")
    
    # 2. 병렬 처리로 랜드마크 추출
    X_data = []
    y_data = []

    print("MediaPipe 랜드마크 추출 중 (병렬 처리)...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # tqdm으로 진행률 표시
        results = list(tqdm(executor.map(process_image, data_list), total=len(data_list)))

    # None(얼굴 미검출) 제거 및 데이터 병합
    for res in results:
        if res is not None:
            X_data.append(res[0])
            y_data.append(res[1])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"전처리 완료: 유효 데이터 {len(X_data)}개")
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

    # 데이터셋 & 데이터로더 생성
    train_dataset = FaceEmotionDataset(X_train, y_train)
    test_dataset = FaceEmotionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 모델 초기화 및 GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}")
    
    model = EmotionMLP()
    
    # GPU가 여러 개일 경우 DataParallel 사용
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()}개의 GPU를 사용합니다! (DataParallel)")
        model = nn.DataParallel(model)
    
    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 학습 루프
    best_acc = 0.0
    
    print("\n모델 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 검증 (Validation)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

        # 최고 성능 모델 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # DataParallel 사용 시 module에 접근하여 저장
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), "best_emotion_model.pth")
            print(f"   --> Best Model Saved! ({best_acc:.2f}%)")

    print(f"\n🏆 학습 종료. 최종 최고 정확도: {best_acc:.2f}%")