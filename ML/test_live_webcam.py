
import cv2
import mediapipe as mp
import numpy as np
import requests
import json
import time

# ============================
# 설정
# ============================
SERVER_URL = "http://127.0.0.1:8000/predict/landmarks"
WEBCAM_INDEX = 0  # 웹캠 번호 (보통 0번)

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def normalize_landmarks(landmarks):
    """
    서버/학습 코드와 동일한 정규화 로직 적용
    """
    # 1. (x, y, z) 좌표 추출
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 2. 중심 이동 (코 끝: 1번 랜드마크 기준)
    nose_tip = coords[1]
    coords_centered = coords - nose_tip
    
    # 3. 스케일링 (최대 거리로 나눔)
    max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
    if max_dist > 0:
        coords_normalized = coords_centered / max_dist
    else:
        coords_normalized = coords_centered
        
    return coords_normalized.flatten().tolist()

def main():
    # 윈도우 호환성을 위해 CAP_DSHOW 옵션 추가
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Warning: 웹캠({WEBCAM_INDEX})을 열 수 없습니다. 1번 카메라를 시도합니다.")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("Error: 사용 가능한 웹캠을 찾을 수 없습니다. (다른 프로그램이 사용 중인지 확인하세요)")
        return

    print(f"웹캠이 시작되었습니다. 서버({SERVER_URL})로 데이터를 전송합니다.")
    print("종료하려면 'q' 키를 누르세요.")

    prev_time = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        # 성능 향상을 위해 쓰기 가능 여부 비활성화 (MediaPipe 요구사항)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(image_rgb)

        # 다시 그리기 위해 활성화
        image.flags.writeable = True
        
        # 기본 감정 상태
        emotion_text = "Analysis..."
        color = (255, 255, 255) # 흰색
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # 2. 데이터 전송 및 예측 요청
                try:
                    # 정규화된 랜드마크 추출
                    landmarks_list = normalize_landmarks(face_landmarks.landmark)
                    
                    # 서버 요청
                    response = requests.post(SERVER_URL, json={"landmarks": landmarks_list}, timeout=0.5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        pred = result['prediction']
                        score = result['anxiety_score']
                        
                        # 시각화 (불안하면 빨강, 중립이면 초록)
                        if pred == "Anxious":
                            emotion_text = f"Anxious ({score:.1f}%)"
                            color = (0, 0, 255) # Red
                        else:
                            emotion_text = f"Neutral"
                            color = (0, 255, 0) # Green
                            
                except Exception as e:
                    print(f"Server Error: {e}")
                    emotion_text = "Server Disconnected"
                    color = (0, 0, 0)

        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 화면에 텍스트 출력
        cv2.putText(image, f"Status: {emotion_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('Emotion Detection Client', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
