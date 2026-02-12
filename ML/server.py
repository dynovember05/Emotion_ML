
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
import base64
import os

# ============================
# 1. 모델 정의 (ResNet) - 학습 때와 동일해야 함
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
    def __init__(self, input_size=1434, hidden_dim=512):
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
        return self.classifier(self.res_block3(self.res_block2(self.res_block1(self.input_layer(x)))))

# ============================
# 2. 서버 초기화 및 모델 로드
# ============================
app = FastAPI(title="Emotion Detection API", description="Real-time Face Emotion Analysis")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ML/emotion_resnet_best.pth"
model = None

# MediaPipe 초기화 (서버에서 이미지를 받을 때 사용)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # 모델 구조 생성 (Input Size: 478개 랜드마크 * 3좌표 = 1434)
    model = ResidualEmotionNet(input_size=1434).to(DEVICE)
    
    # DataParallel로 저장된 가중치를 로드할 때의 처리 (module. prefix 제거)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Model loaded on {DEVICE}")

# ============================
# 3. 데이터 정의 (Pydantic)
# ============================
class LandmarkInput(BaseModel):
    landmarks: list[float]  # [x1, y1, z1, x2, y2, z2, ...] (길이 1434)

class ImageInput(BaseModel):
    image_base64: str       # Base64 encoded image string

# ============================
# 4. API 엔드포인트
# ============================

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Emotion Detection Server is Running"}

@app.post("/predict/landmarks")
async def predict_landmarks(data: LandmarkInput):
    """
    클라이언트가 이미 정규화된 랜드마크를 보낼 때 사용
    """
    try:
        # 입력 데이터 텐서 변환
        input_tensor = torch.tensor(data.landmarks, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 추론
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        prediction = predicted_class.item()
        label = "Anxious" if prediction == 1 else "Neutral"
        prob = probabilities[0][1].item() # 불안일 확률
        
        return {
            "prediction": label,
            "confidence": float(confidence.item()),
            "anxiety_score": round(prob * 100, 2) # 퍼센트
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image(data: ImageInput):
    """
    클라이언트가 이미지를 보내면 서버가 랜드마크 추출 후 분석
    """
    try:
        # 1. Base64 -> Image 변환
        image_bytes = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. MediaPipe로 랜드마크 추출
        results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return {"prediction": "No Face Detected", "anxiety_score": 0.0}
            
        # 3. 좌표 추출 및 정규화 (학습 때와 동일한 로직)
        landmarks_raw = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks_raw.append([lm.x, lm.y, lm.z])
        
        landmarks_raw = np.array(landmarks_raw)
        
        # 중심 이동 (코 끝 기준)
        nose_tip = landmarks_raw[1]
        landmarks_centered = landmarks_raw - nose_tip
        
        # 스케일링
        max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
        if max_dist > 0:
            landmarks_normalized = landmarks_centered / max_dist
        else:
            landmarks_normalized = landmarks_centered
            
        # 평탄화 (Flatten)
        features = landmarks_normalized.flatten().tolist()
        
        # 4. 모델 추론 호출
        # (위의 predict_landmarks 로직 재사용 가능하지만, 텐서 변환 바로 수행)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
        prob_anxious = probabilities[0][1].item()
        label = "Anxious" if prob_anxious > 0.5 else "Neutral"
        
        return {
            "prediction": label,
            "anxiety_score": round(prob_anxious * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
