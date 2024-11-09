import cv2
import time
from ultralytics import YOLO

# YOLOv8 모델 불러오기
person_model = YOLO('yolov8n.pt')
badge_model = YOLO(r'/home/jhchoman/YOLO_1101/runs/detect/train2/weights/best.pt') #만들어진 pt파일의 경로

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 추적기 및 타이머 초기화
tracker = None
tracking = False
timer_start = None
timer_duration = 3  # 타이머 지속 시간 (초)
last_box = None
badge_found = False  # 뱃지 발견 여부 초기화
person_status = None  # "employee" 또는 "visitor" 상태 저장

# 화면의 중앙 영역 정의 (30%로 설정)
center_threshold = 0.3  # 중앙 영역 크기 비율
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x_min = int(frame_width * (0.5 - center_threshold / 2))
center_x_max = int(frame_width * (0.5 + center_threshold / 2))
center_y_min = int(frame_height * (0.5 - center_threshold / 2))
center_y_max = int(frame_height * (0.5 + center_threshold / 2))

def is_in_center(bbox):
    """객체가 화면 중앙에 있는지 확인"""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    return center_x_min <= cx <= center_x_max and center_y_min <= cy <= center_y_max

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라로부터 프레임을 가져오지 못했습니다.")
        break

    # 추적이 시작되지 않은 경우에만 최초 탐지된 객체를 추적기로 설정
    if not tracking:
        # YOLOv8을 통해 사람 인식 수행
        results = person_model.predict(frame)
        largest_box = None

        # 인식된 객체 중에서 가장 큰 바운딩 박스 찾기
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # '사람' 클래스
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_area = (x2 - x1) * (y2 - y1)
                    if largest_box is None or box_area > largest_box[0]:
                        largest_box = (box_area, (x1, y1, x2, y2))

        # 사람이 화면의 중앙에 있을 때만 추적기 초기화
        if largest_box:
            _, (x1, y1, x2, y2) = largest_box
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            tracking = True
            timer_start = time.time()
            last_box = (x1, y1, x2, y2)
            badge_found = False  # 뱃지 초기화
            person_status = None  # 상태 초기화
            print("사람 인식 완료, 추적기로 전환합니다.")

    else:
        # 추적기 업데이트
        success, bbox = tracker.update(frame)
        if success and is_in_center(bbox):
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Tracking Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 타이머가 켜진 상태에서 5초 동안 뱃지 판별
            if timer_start is not None:
                elapsed_time = time.time() - timer_start
                remaining_time = int(timer_duration - elapsed_time)
                cv2.putText(frame, f"Timer: {remaining_time}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 5초 동안 뱃지 판별 수행 (70% 이상의 신뢰도 기준)
                if elapsed_time <= timer_duration:
                    person_roi = frame[y:y + h, x:x + w]  # 사람 영역
                    
                    # person_roi가 유효한 이미지인지 확인
                    if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        badge_results = badge_model.predict(person_roi)

                        # 뱃지 신뢰도 확인 및 위치 제한
                        for badge_result in badge_results:
                            for badge_box in badge_result.boxes:
                                if badge_box.cls[0] == 0 and badge_box.conf[0] >= 0.5:
                                    bx1, by1, bx2, by2 = map(int, badge_box.xyxy[0])

                                    # 사람 바운딩 박스 내의 상반신 부분에 뱃지가 있는지 확인
                                    if y + h // 2 <= y + by1 <= y + h:  # 상반신에 뱃지 위치 제한
                                        badge_found = True

                # 타이머가 5초 경과했을 때 결과를 표시
                if elapsed_time >= timer_duration:
                    if badge_found:
                        person_status = "Employee"
                        print("직원 확인")
                    else:
                        person_status = "Visitor"
                        print("방문자 확인")
                    timer_start = None  # 타이머 종료

            # 직원/방문자 상태를 프레임에 표시
            if person_status:
                status_text = "Employee" if person_status == "Employee" else "Visitor"
                color = (0, 255, 0) if person_status == "Employee" else (0, 0, 255)
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        else:
            print("객체가 중앙을 벗어났습니다. 추적을 중지합니다.")
            tracking = False
            timer_start = None
            last_box = None
            badge_found = False  # 추적 중지 시 뱃지 초기화
            person_status = None  # 상태 초기화

    # 결과 프레임 디스플레이
    cv2.imshow("YOLOv8 Person Detection and Tracking with Badge Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
