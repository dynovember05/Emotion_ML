
# 🎭 ResNet 기반 실시간 얼굴 감정 인식 시스템 (중립 vs 불안)

이 프로젝트는 **한국인 안면 감정 데이터**를 활용하여 사용자가 느끼는 '불안'을 실시간으로 탐지하는 AI 시스템입니다. **MediaPipe FaceMesh**와 **ResNet 스타일의 심층 신경망**을 결합하여, 이미지 전체가 아닌 안면의 구조적 변화(랜드마크)만을 분석해 매우 빠르고 정확합니다.

---

## 📊 학습 데이터 상세 (Data Details)
학습 과정에 대한 더 자세한 기술 리포트는 [TRAINING_DETAIL.md](./TRAINING_DETAIL.md)에서 확인하실 수 있습니다.

### 1. 데이터 출처 및 분류
- **데이터셋**: AI Hub "한국인 감정인식을 위한 복합 영상" (수만 장의 한국인 데이터 활용)
- **라벨링 로직**: 파일명의 키워드를 자동 분석하여 분류
  - **Neutral (중립)**: 평상시 표정
  - **Anxious (불안)**: 눈썹, 입술의 미세한 떨림 및 수축이 포함된 표정
- **샘플링**: 과적합 방지를 위해 ZIP 파일당 최대 5,000장씩, 총 10만 장 이상의 데이터를 골고루 추출하여 학습에 사용했습니다.

### 2. 전처리 및 정규화
- **FaceMesh (478 Landmarks)**: 얼굴 전체 이미지가 아닌 1,434개의 3D 좌표 피처만 추출하여 모델을 경량화했습니다.
- **정규화**: 얼굴의 크기나 위치에 상관없이 감정만을 읽기 위해 모든 랜드마크를 **'코 끝 중심'**으로 정렬하고 스케일을 표준화했습니다.

---

## ✨ 핵심 기술 (Key Features)
- **ResNet-style MLP**: 스킵 커넥션(Skip Connection) 구조를 도입하여 85%의 성능 벽을 돌파했습니다.
- **고급 증강 (Noise Injection)**: 랜드마크 데이터에 미세한 노이즈를 섞어, 어떤 얼굴 각도에서도 안정적으로 작동합니다.
- **FastAPI 서버**: Python 기반의 고성능 API 서버를 통해 실시간 추론을 지원합니다.
- **Web Client**: 브라우저에서 바로 카메라를 켜고 분석 결과를 확인할 수 있는 전용 웹 페이지를 포함합니다.

---

## 🚀 빠른 시작 (Getting Started)

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행 (Backend)
```bash
python ML/server.py
```
*서버는 `http://localhost:8001`에서 가동됩니다.*

### 3. 실시간 웹캠 테스트 (Frontend)
- `ML/webcam_client.html` 파일을 브라우저(Chrome 권장)로 직접 엽니다.
- 카메라 권한을 허용하면 실시간으로 **불안도(Anxiety Score)**와 감정 상태가 시각화됩니다.

---

## � 성능 리포트 (Performance)
실제 운영 환경에서의 성능을 보장하기 위해, 학습에 단 한 번도 사용되지 않은 인물들의 데이터(**Validation Set**)로만 검토했습니다.

- **최종 정확도 (Accuracy)**: **85.2%**
- **불안 재현율 (Anxious Recall)**: **86%** (실제 불안함을 놓치지 않고 찾아내는 능력)
- **모델 크기**: 약 10MB (저사양 환경에서도 원활히 작동)

---

## 📂 프로젝트 구조
```
Face/
├── ML/
│   ├── Emotion_Training_ResNet.ipynb  # 최강 성능(85%)을 뽑아낸 최종 학습 노트북
│   ├── emotion_resnet_best.pth        # 학습 완료된 모델 가중치
│   ├── server.py                      # FastAPI 추론 서버
│   ├── webcam_client.html             # 실시간 웹 브라우저 클라이언트
│   ├── data_loader.py                 # 데이터 로딩 및 전처리 모듈
│   └── evaluate_resnet.py             # 모델 성능 평가 스크립트
├── requirements.txt                   # 설치 필요 패키지 목록
└── README.md                          # 현재 문서
```

---

## 📝 라이선스
본 프로젝트는 MIT 라이선스 하에 배포됩니다.
