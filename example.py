# 필요한 라이브러리 및 모듈 가져오기

# 로봇 조작 및 통신을 위한 SDK 모듈
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

# 객체 감지와 이미지 처리를 위한 모듈
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from dataclasses import dataclass

# 이미지 처리를 위한 모듈
import cv2
import numpy as np
import sys
import time
import imutils



# 객체 감지 및 랜드 표시를 위한 객체 초기화
mp_hands = mp.solutions.hands                   # 손 감지를 위한 솔루션 객체 초기화
mp_drawing = mp.solutions.drawing_utils         # 인식된 모델에 맞게 랜드 그리기 위한 객체 초기화
mp_pose = mp.solutions.pose                     # 자세 감지를 위한 솔루션 객체 초기화


# 손 감지를 위한 매개변수 초기화
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 자세 감지를 위한 매개변수 초기화
pose = mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)


# 동작 인식을 위한 변수 초기화
prev_x = None                   # 손목의 이전 x좌표값
motion_counter = 0              # 모션 횟수 감지
threshold = 5                   # 모션 감지 임계값


# 손 제스쳐에 따른 명령어 정의
move_order = {
        ("fold", "----", "fold", "fold", "fold") : "sport_client.Move(0.3, 0, 0)",
        ("fold", "fold", "fold", "fold", "fold") : "sport_client.StopMove()",
        ("----", "----", "fold", "fold", "fold") : "sport_client.Move(0.3, 0, 0.7)",
        ("fold", "----", "----", "fold", "fold") : "sport_client.Move(0.3, 0, -0.7)",
        }


# 두 점 사이의 거리 계산 함수
def two_point_distance(point_one, point_two):

    return math.sqrt(math.pow((point_one.x - point_two.x),2) + math.pow((point_one.y - point_two.y), 2))

# 손가락 접힘 여부 판단 함
def fold_finger(wrist, tip, mcp):
    if two_point_distance(wrist, tip) < two_point_distance(wrist, mcp):
        return "fold"
    else:
        return "----"


# 손 제스처 인식 함수
def recognize_gesture(hand_landmarks):
    
    # 손 랜드마크 정의
    landmarks = hand_landmarks.landmark
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    # 각 손가락의 접힌 유무 판단 후 변수에 저장
    thumb = fold_finger(pinky_mcp, thumb_tip, thumb_cmc)
    index = fold_finger(wrist, index_tip, index_pip)
    middle = fold_finger(wrist, middle_tip, middle_pip)
    ring = fold_finger(wrist, ring_tip, ring_pip)
    pinky = fold_finger(wrist, pinky_tip, pinky_pip)

    # 손가락 접힘 유무 튜플로 반환
    return (thumb, index, middle, ring, pinky)


# 객체 탐지결과 시각화 함수
def visualize(
    image,
    detection_result
) -> np.ndarray:


  for detection in detection_result.detections:
    # 상자 경계 그리기
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origi수n_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # 레이블 및 점수 표시
    category = detection.categories[0]
    category_name = category.categoryrom unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image



if __name__ == "__main__":
    
    # 네트워크 인터페이스 변수 확인
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    # 채널 초기화
    ChannelFactoryInitialize(0, sys.argv[1])
    
    # 시각화 설정
    MARGIN = 10                 # 여백
    ROW_SIZE = 10               # 행 크기
    FONT_SIZE = 3               # 글꼴 크기
    FONT_THICKNESS = 1          # 글꼴 두께
    TEXT_COLOR = (255, 0, 0)    # 텍스트 색


    # 로봇 컨트롤 객체 초기화
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # 전면 카메라 객체 초기화
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()
    
    # 영상 이미지 데이터 가져오기
    code, data = video_client.GetImageSample()

    # 영상이 정상적으로 인식 되는 동안 지속적으로 화면 송출
    while code == 0:
        
        code, data = video_client.GetImageSample()

        # 데이터 형식으로 불러온 이미지 디코딩
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        # 이미지 좌우 반전
        image = cv2.flip(image, 1)
        
        # 이미지 채널 rgb로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 손 모양과 자세 탐지
        hand_results = hands.process(image_rgb)
        pose_result = pose.process(image_rgb)

        # 손 모양 탐지 결과 처리 과정
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # 손 랜드 마크 시각화
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 손가락 접힘 유무(손 제스처) 인식 
                fold = recognize_gesture(hand_landmarks)
                try:
                    # 손 제스처에 따른 로봇 행동 명령 실행
                    exec(move_order[fold])
                except:
                    pass
                # 손 제스처 텍스트로 화면상에 표시
                cv2.putText(image, f'Fold: {fold}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 자세 탐지 결과 처리 과정
        if pose_result.pose_landmarks:

            # 자세 랜드 마크 시각화
            mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 어깨 좌표 계산
            left_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # 손목 좌표 계산
            left_wrist = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # 양 어깨의 높이 평균값 계산
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * image.shape[0]
            
            # 손목 위치에 따른 동작 감지
            for wrist in [left_wrist, right_wrist]:
                if wrist.y * image.shape[0] < shoulder_y:
                    x = wrist.x
                    
                    # 손목 움직임 감지
                    if prev_x is not None:
                        if abs(x - prev_x) > 0.03:
                            motion_counter += 1
                        else:
                            motion_counter = max(0, motion_counter - 1)
                        
                        # 움직임 계산이 임계값을 넘었을시 로봇 행동
                        if motion_counter > threshold:
                            sport_client.Hello()

                    prev_x = x
        
        # 송출될 이미지  크기 조정
        image = imutils.resize(image, width = 1000)

        
        # 화면 송출
        cv2.imshow("front_camera", image)
        
        # ESC 입력시 종료
        if cv2.waitKey(20) == 27:
            break
    
    # 영상을 정상적으로 불러오지 못하여 종료되었을 경우
    if code != 0:
        print("Get image sample error. code:", code)

    # ESC를 눌러 종료했을 경우 마지막 이미지 프레임 저장
    else:
        cv2.imwrite("front_image.jpg", image)
    
    cv2.destroyWindow("front_camera")
