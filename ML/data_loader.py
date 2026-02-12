
import cv2
import mediapipe as mp
import numpy as np

# 병렬 처리를 위해 별도 파일로 분리된 함수
def process_single_image(args):
    """
    이미지 데이터(bytes) 하나를 받아서 랜드마크를 추출하고 정규화하는 함수
    (ProcessPoolExecutor에서 실행됨)
    """
    img_data, label = args
    try:
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None: return None
        
        # FaceMesh 객체를 프로세스마다 새로 생성해야 충돌이 없음
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5
        ) as face_mesh_local:
            
            # BGR -> RGB 변환
            results = face_mesh_local.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                landmarks_raw = []
                for lm in results.multi_face_landmarks[0].landmark:
                    landmarks_raw.append([lm.x, lm.y, lm.z])
                
                landmarks_raw = np.array(landmarks_raw)
                
                # 1. 중심 이동
                nose_tip = landmarks_raw[1]
                landmarks_centered = landmarks_raw - nose_tip
                
                # 2. 스케일링
                max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
                if max_dist > 0:
                    landmarks_normalized = landmarks_centered / max_dist
                else:
                    landmarks_normalized = landmarks_centered
                
                return landmarks_normalized.flatten(), label

    except Exception:
        return None
    return None
