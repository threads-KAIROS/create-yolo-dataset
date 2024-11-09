from ultralytics import YOLO
import torch

def train_model():
    # YOLOv8 모델 로드
    model = YOLO('yolov8m.pt')  # 모델을 GPU로 전송

    # 훈련 설정
    model.train(data=r'/home/jhchoman/YOLO_1101/detection/panda.yaml',epochs=40,device=0)

if __name__ == '__main__':
    train_model()
