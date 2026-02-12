
# 🎭 Real-time Facial Emotion Recognition with ResNet

This project demonstrates a robust facial emotion recognition system (Neutral vs. Anxious) using **MediaPipe Face Mesh** and a custom **ResNet-style MLP** model.

It achieves **85% accuracy** on unseen validation data and includes a real-time web client powered by **FastAPI**.

## ✨ Key Features
- **High Performance**: Uses a Residual MLP architecture with GeLU activation and label smoothing.
- **Robustness**: Trained with noise injection augmentation to handle various face angles.
- **Efficiency**: Processes only facial landmarks (not full images), making it extremely lightweight and fast.
- **Real-time Demo**: Includes a web-based client that runs in the browser and communicates with a local Python server.

## 🛠️ Tech Stack
- **Model**: PyTorch (ResNet MLP), MediaPipe
- **Server**: FastAPI, Uvicorn
- **Client**: HTML5, JavaScript (MediaPipe JS)
- **Data Processing**: NumPy, OpenCV

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Face-Emotion-Recognition.git
cd Face-Emotion-Recognition
pip install -r requirements.txt
```

### 2. Run the Server
Start the FastAPI backend server (loads `emotion_resnet_best.pth`):
```bash
python ML/server.py
```
*Server will start at `http://0.0.0.0:8001`*

### 3. Run the Client
Open `ML/webcam_client.html` in your web browser.
- Allow webcam access.
- The client will extract facial landmarks in real-time and send them to the server.
- The server responds with the predicted emotion (Neutral/Anxious) and confidence score.

## 📊 Model Performance
- **Validation Accuracy**: 85%
- **Recall (Anxious)**: 0.86 (High sensitivity to anxiety)
- **Precision (Neutral)**: 0.85

## 📂 Project Structure
```
Face/
├── ML/
│   ├── Emotion_Training_ResNet.ipynb  # Main training notebook (The Best Model)
│   ├── emotion_resnet_best.pth        # Trained model weights
│   ├── server.py                      # FastAPI inference server
│   ├── webcam_client.html             # Web-based real-time demo
│   ├── data_loader.py                 # Data preprocessing module
│   └── evaluate_resnet.py             # Evaluation script
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 📝 License
This project is open-source and available under the MIT License.
