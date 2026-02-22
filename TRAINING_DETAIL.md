
# 🧠 Detailed Training Process: Emotion Detection ResNet

이 문서는 본 프로젝트의 감정 인식 모델이 어떻게 학습되었는지에 대한 상세한 기술적 리포트입니다. 단순히 층을 쌓는 것을 넘어, 정확도 85% 이상의 안정적인 모델을 만들기 위해 적용된 핵심 기법들을 설명합니다.

---

## 1. 데이터 전처리 (Data Preprocessing)

감정 인식의 핵심은 얼굴의 **"형태(Geometry)"**를 얼마나 잘 추출하느냐에 있습니다.

### 1.1 MediaPipe FaceMesh 활용
- **입력 데이터**: 640x480 이상의 안면 이미지
- **추출 피처**: MediaPipe FaceMesh를 통해 478개의 3D 안면 랜드마크 추출 (x, y, z 좌표 총 1,434개 피처)
- **병렬 처리**: `ProcessPoolExecutor`를 사용하여 CPU 멀티코어로 수만 장의 이미지를 고속으로 전처리 (약 10배 속도 향상)

### 1.2 좌표 정규화 (Normalization) - 성능 향상의 핵심
얼굴이 화면 어디에 있든, 크기가 어떻든 동일하게 인식하기 위해 다음 과정을 거칩니다.
1.  **중심 이동 (Translation)**: 모든 좌표에서 '코 끝(Landmark Index 1)' 좌표를 빼서, 코 끝을 항상 (0,0,0)으로 맞춥니다.
2.  **스케일링 (Scaling)**: 중심으로부터 가장 먼 랜드마크까지의 거리를 계산하여 모든 좌표를 해당 거리로 나눕니다. 이를 통해 얼굴의 크기 변화에 무관한 데이터를 얻습니다.

---

## 2. 모델 아키텍처 (Model Architecture)

단순한 MLP(Multi-Layer Perceptron)는 층이 깊어질수록 학습 효율이 떨어지는 문제가 있어 **ResNet(Residual Network)** 스타일의 아키텍처를 도입했습니다.

### 2.1 Residual MLP 구조
- **Skip Connection**: `x = x + block(x)` 구조를 사용하여 정보 손실 없이 네트워크를 깊게 쌓았습니다. 이는 85% 성능 정체를 돌파한 핵심 요인이었습니다.
- **주요 레이어**:
    - **Linear Layer**: 고차원 피처 추출
    - **Batch Normalization**: 학습 안정성 및 속도 향상
    - **GELU Activation**: ReLU보다 부드러운 곡선을 가져 최적화에 유리한 최신 활성화 함수 사용
    - **Dropout (0.3~0.4)**: 과적합(Overfitting) 방지

---

## 3. 고급 학습 기법 (Advanced Training Techniques)

### 3.1 Noise Injection (데이터 증강)
- 훈련 시 랜드마크 좌표에 미세한 가우시안 노이즈(std=0.002)를 추가했습니다.
- 이는 모델이 특정 좌표를 외우는 것이 아니라, 얼굴의 전반적인 **구조적 특징**을 배우게 하여 일반화 성능을 높입니다.

### 3.2 Learning Rate Scheduling
- **Cosine Annealing LR**: 학습률을 코사인 곡선에 따라 부드럽게 감소시켜, 최적의 수렴 지점에 안정적으로 도달하게 합니다.
- 초반에는 빠르게 배우고, 후반에는 미세하게 조정하여 최상의 성능을 끌어냈습니다.

### 3.3 Label Smoothing (0.05)
- 정답 라벨을 완벽한 0 또는 1이 아닌 0.05, 0.95로 부드럽게 만들어 모델이 데이터에 과도하게 확신(Overconfidence)하여 과적합되는 것을 방지했습니다.

---

## 4. 검증 전략 (Evaluation Strategy)

본 프로젝트는 **'진짜 성능'**을 측정하기 위해 매우 엄격한 검증 방식을 택했습니다.

- **Data Leakage 방지**: 같은 영상 내의 프레임들은 매우 유사하므로, 단순히 섞어서 나누는 것(Simple Split)은 정확도가 가짜로 높게 나옵니다(99% 등).
- **Strict Validation**: 학습에 단 한 번도 사용되지 않은 **완전히 새로운 인물과 영상(Validation Set)** 폴더를 사용하여 최종 성능을 측정했습니다. 그 결과 **실전 정확도 85%**를 확보했습니다.

---

## 5. 결과 (Results)

| Metric | Neutral (중립) | Anxious (불안) | Overall |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.85 | 0.84 | - |
| **Recall** | 0.84 | 0.86 | - |
| **Accuracy** | - | - | **85%** |

특히 불안(Anxious) 감정의 **Recall(재현율)이 86%**로 높게 나타나, 실제 불안 상태를 놓치지 않고 감지하는 능력이 탁월함을 입증했습니다.
