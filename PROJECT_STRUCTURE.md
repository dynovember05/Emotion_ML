# Face Emotion Recognition Project (MediaPipe + PyTorch)

이 문서는 이 프로젝트의 파일 구조와 각 스크립트의 역할을 설명합니다. 이 프로젝트는 한국인 감정 인식 데이터(NIA)를 활용하여 MediaPipe FaceMesh로 랜드마크를 추출하고, PyTorch MLP 모델로 감정을 분류하는 것을 목표로 합니다.

## 1. 디렉토리 구조 (Directory Structure)

```
c:\Users\ldy34\Desktop\Face\
├── .venv/                      # Python 가상 환경 (MediaPipe, PyTorch 설치됨)
├── ML/                         # 머신러닝 학습 및 테스트 코드
│   ├── Emotion_Training_MediaPipe.ipynb  # [메인] 데이터 탐색 및 시각화용 주피터 노트북
│   ├── train_emotion_gpu.py              # [스크립트] 다중 GPU 학습용 코드 (Sequential Load)
│   ├── train_emotion_parallel.py         # [스크립트] CPU 병렬 데이터 로드 + GPU 학습 (추천)
│   └── train_emotion_10k.py              # [테스트] 1만장 데이터 학습 테스트
│
├── video/                      # 학습 데이터 (대용량 ZIP 파일)
│   ├── Training/               # 학습 데이터셋
│   │   ├── [원천]EMOIMG_중립_TRAIN_01.zip  (약 30GB)
│   │   ├── [원천]EMOIMG_불안_TRAIN_01.zip  (약 30GB)
│   │   └── ... (기타 감정 데이터)
│   └── Validation/             # 검증 데이터셋
│
├── video_data/                 # 추가 데이터 (참고용)
│   └── 01.데이터/
│
└── PROJECT_STRUCTURE.md        # (현재 파일) 프로젝트 구조 설명서
```

---

## 2. 주요 스크립트 설명

### 1) `ML/Emotion_Training_MediaPipe.ipynb`
*   **역할:** 데이터 탐색, 전처리 로직 테스트, 시각화(Loss/Accuracy 그래프)를 위한 주피터 노트북입니다.
*   **특징:**
    *   `extract_landmarks_from_zip`: ZIP 압축을 풀지 않고 이미지를 메모리로 올려 랜드마크를 추출합니다.
    *   **전처리:** 얼굴 랜드마크의 **중심 맞추기(Centering)** 및 **크기 정규화(Scaling)** 로직이 포함되어 있습니다.
    *   **데이터:** '중립(0)'과 '불안(1)' 두 가지 감정을 분류합니다.

### 2) `ML/train_emotion_parallel.py` (★ 추천)
*   **역할:** 대용량 데이터를 빠르게 로드하여 학습시키는 실행 스크립트입니다.
*   **핵심 기술:**
    *   **Multi-Processing:** `ProcessPoolExecutor`를 사용하여 CPU의 모든 코어를 활용, 랜드마크 추출 속도를 극대화했습니다.
    *   **DataParallel:** 2개의 GPU(RTX 3090 Ti)를 모두 사용하여 VRAM을 최적화했습니다.
*   **사용법:**
    ```bash
    # 터미널에서 실행
    .\.venv\Scripts\python.exe ML\train_emotion_parallel.py
    ```

### 3) `ML/train_emotion_gpu.py`
*   **역할:** 병렬 처리가 아닌 순차적(Sequential) 방식으로 데이터를 로드하는 안정성 위주의 스크립트입니다.
*   **단점:** 데이터 로딩 속도가 느릴 수 있습니다.

---

## 3. 학습 모델 구조 (PyTorch)
*   **입력:** MediaPipe FaceMesh 랜드마크 (478개 점 x 3차원 = 1434 Features)
*   **구조:** MLP (Multi-Layer Perceptron)
    *   Input(1434) -> Linear(512) -> ReLU -> BatchNorm -> Dropout(0.3)
    *   -> Linear(256) -> ReLU -> BatchNorm -> Dropout(0.2)
    *   -> Linear(128) -> ReLU
    *   -> Output(2) : [Neutral, Anxious]

---

## 4. 환경 설정 (Environment)
*   **Python:** 3.12.11
*   **CUDA:** 12.1 (PyTorch 2.5.1)
*   **GPU:** NVIDIA GeForce RTX 3090 Ti (x2)
*   **주요 라이브러리:**
    *   `mediapipe`: 얼굴 랜드마크 추출 (0.10.14 권장)
    *   `torch`: 딥러닝 프레임워크
    *   `opencv-python`: 이미지 처리
    *   `scikit-learn`: 데이터 분할 및 평가
