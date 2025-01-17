# ---------------------------------------------------
# combined_dance_voice.py
# (손 제스처 + 웨이브 감지 + 음성 인식)
# ---------------------------------------------------
import sys
import time
import math
import imutils
import numpy as np
import cv2

# --- Unitree SDK ---
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# --- Mediapipe (손 제스처) ---
import mediapipe as mp

# --- Speech Recognition (음성 인식) ---
import speech_recognition as sr


# -----------------------------
# 음성 인식용 함수
# -----------------------------
def listen_command(recognizer, mic, timeout=3, phrase_time_limit=2):
    """
    마이크로부터 음성을 받아 텍스트로 반환.
    - timeout: 이 시간 동안 말이 없으면 listen()에서 예외 발생
    - phrase_time_limit: 이 시간을 넘으면 자동으로 녹음 중단
    """
    with mic as source:
        print(">>> [음성] 듣는 중... (최대 {0}초)".format(timeout))
        # 주변 소음으로부터 마이크 수준 보정
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        # 구글 STT (한국어)
        text = recognizer.recognize_google(audio, language="ko-KR")
        print("    [인식 결과]:", text)
        return text
    except sr.WaitTimeoutError:
        print("    [음성] 대기 시간 초과.")
    except sr.UnknownValueError:
        print("    [음성] 인식 불가.")
    except sr.RequestError as e:
        print("    [음성] 구글 API 오류:", e)
    return None


# -----------------------------
# 손가락 접힘 판별 함수들
# -----------------------------
def two_point_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def fold_finger(wrist, tip, mcp):
    # wrist -> tip 거리 < wrist -> mcp 거리면 접힘(fold)
    if two_point_distance(wrist, tip) < two_point_distance(wrist, mcp):
        return "fold"
    else:
        return "----"

def recognize_gesture(hand_landmarks):
    """
    (thumb, index, middle, ring, pinky) 상태로 제스처 구분 -> 동작 매핑
    """
    mp_hands = mp.solutions.hands
    landmarks = hand_landmarks.landmark

    WRIST       = mp_hands.HandLandmark.WRIST
    THUMB_TIP   = mp_hands.HandLandmark.THUMB_TIP
    THUMB_CMC   = mp_hands.HandLandmark.THUMB_CMC
    INDEX_TIP   = mp_hands.HandLandmark.INDEX_FINGER_TIP
    INDEX_PIP   = mp_hands.HandLandmark.INDEX_FINGER_PIP
    MIDDLE_TIP  = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    MIDDLE_PIP  = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
    RING_TIP    = mp_hands.HandLandmark.RING_FINGER_TIP
    RING_PIP    = mp_hands.HandLandmark.RING_FINGER_PIP
    PINKY_TIP   = mp_hands.HandLandmark.PINKY_TIP
    PINKY_PIP   = mp_hands.HandLandmark.PINKY_PIP

    thumb  = fold_finger(landmarks[WRIST], landmarks[THUMB_TIP],  landmarks[THUMB_CMC])
    index  = fold_finger(landmarks[WRIST], landmarks[INDEX_TIP],  landmarks[INDEX_PIP])
    middle = fold_finger(landmarks[WRIST], landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP])
    ring   = fold_finger(landmarks[WRIST], landmarks[RING_TIP],   landmarks[RING_PIP])
    pinky  = fold_finger(landmarks[WRIST], landmarks[PINKY_TIP],  landmarks[PINKY_PIP])

    fingers_state = (thumb, index, middle, ring, pinky)

    gesture_map = {
        ("fold","fold","fold","fold","fold"):   "stop",      # 주먹
        ("fold","----","----","----","----"):   "forward",   # 엄지만 접힘
        ("fold","fold","----","----","----"):   "left",      # 왼쪽
        ("fold","----","fold","----","----"):   "right",     # 오른쪽
        # 필요하면 추가
    }

    return gesture_map.get(fingers_state, None)


# -----------------------------
# 메인 실행
# -----------------------------
if __name__ == "__main__":

    # (1) 네트워크 인터페이스 체크
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    # (2) 채널 초기화
    ChannelFactoryInitialize(0, sys.argv[1])

    # (3) 로봇 컨트롤 객체 초기화
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # (4) 전면 카메라 객체 초기화
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()

    # (5) Mediapipe Hands 객체 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # (6) 음성 인식 객체 초기화
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # (7) 웨이브(손 흔들기) 감지 변수
    prev_wrist_x = None
    wave_count = 0
    WAVE_COUNT_THRESHOLD = 4
    WAVE_MOVE_THRESH = 0.05
    wave_cooldown = False
    wave_cooldown_time = 3.0
    last_wave_time = 0.0

    # (8) 로봇 자세 초기화(선택)
    sport_client.StandUp()
    time.sleep(1.5)

    # (9) 카메라 루프
    code, data = video_client.GetImageSample()
    while code == 0:
        # ---------------------
        # A) 음성 명령 듣기
        # ---------------------
        # (여기서는 매 프레임마다 호출 -> 실제론 주기 조정 or 쓰레드 분리 고려)
        command_text = listen_command(recognizer, mic, timeout=3, phrase_time_limit=2)
        if command_text:
            # "도비야"라는 호출어가 있는지 확인
            if "도비야" in command_text:
                # 명령어 파싱
                if "전진" in command_text or "앞으로" in command_text:
                    print(">>> [음성 명령] 전진")
                    sport_client.Move(0.2, 0, 0)
                elif "멈춰" in command_text or "정지" in command_text:
                    print(">>> [음성 명령] 멈춰")
                    sport_client.StopMove()
                elif "우회전" in command_text or ("오른쪽" in command_text and "회전" in command_text):
                    print(">>> [음성 명령] 우회전")
                    sport_client.Move(0, 0, -0.2)
                elif "좌회전" in command_text or ("왼쪽" in command_text and "회전" in command_text):
                    print(">>> [음성 명령] 좌회전")
                    sport_client.Move(0, 0, 0.2)
                elif "춤춰" in command_text or "춤" in command_text:
                    print(">>> [음성 명령] 춤추기")
                    sport_client.StopMove()
                    sport_client.Dance1()
                    time.sleep(2.0)
                else:
                    print(">>> (도비야) 알 수 없는 명령:", command_text)
            else:
                print(">>> 호출어(도비야) 없음:", command_text)

        # ---------------------
        # B) 카메라 영상 처리
        # ---------------------
        code, data = video_client.GetImageSample()
        if code != 0:
            break
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            break

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # Mediapipe Hands 처리
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(image_rgb)

        gesture_result = None
        wrist_x_normalized = None

        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                # 손가락 제스처
                gesture_result = recognize_gesture(hand_landmarks)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                # 웨이브 감지
                WRIST = mp_hands.HandLandmark.WRIST
                wrist = hand_landmarks.landmark[WRIST]
                wrist_x_normalized = wrist.x
                break  # 한 손만 처리

            # 화면에 표시
            if gesture_result:
                cv2.putText(frame, f"Gesture: {gesture_result}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            else:
                cv2.putText(frame, "Gesture: None", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        else:
            cv2.putText(frame, "Gesture: None", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        # (C) 웨이브 -> 춤추기 로직
        current_time = time.time()
        if wrist_x_normalized is not None and not wave_cooldown:
            if prev_wrist_x is not None:
                dx = abs(wrist_x_normalized - prev_wrist_x)
                if dx > WAVE_MOVE_THRESH:
                    wave_count += 1
            prev_wrist_x = wrist_x_normalized

            if wave_count > WAVE_COUNT_THRESHOLD:
                cv2.putText(frame, "Wave Detected -> Dancing!", (10,120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)
                # 춤
                sport_client.StopMove()
                sport_client.Dance1()
                time.sleep(2.0)
                # 초기화
                wave_count = 0
                wave_cooldown = True
                last_wave_time = current_time
        else:
            # 쿨다운 중 -> 일정 시간 후 해제
            if wave_cooldown and (current_time - last_wave_time > wave_cooldown_time):
                wave_cooldown = False
                wave_count = 0
                prev_wrist_x = None

        # (D) 제스처 -> 로봇 동작 (음성 명령보다 우선순위 낮게 예시)
        if gesture_result == "forward":
            sport_client.Move(0.2, 0, 0)
        elif gesture_result == "stop":
            sport_client.StopMove()
        elif gesture_result == "left":
            sport_client.Move(0, 0, 0.2)
        elif gesture_result == "right":
            sport_client.Move(0, 0, -0.2)
        else:
            # 아무 제스처 아니면 그대로 두거나 멈춤
            sport_client.StopMove()

        # (E) 결과 화면 표시
        display_frame = imutils.resize(frame, width=1000)
        cv2.imshow("front_camera", display_frame)

        # ESC 종료
        if cv2.waitKey(20) == 27:
            break

    # 카메라 에러 종료
    if code != 0:
        print("Get image sample error. code:", code)
    else:
        cv2.imwrite("front_image.jpg", frame)

    # 종료 시 로봇 정지
    sport_client.StopMove()
    cv2.destroyAllWindows()
