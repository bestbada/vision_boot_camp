from roboflow import Roboflow
from ultralytics import YOLO
from config.config import ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION

def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset

def train_model(dataset, epochs=30, batch=16):
    # 의미: batch 파라미터를 추가해 배치 크기를 바꿔가며 실험 가능하게 함
    # 사용 이유: 실제 Colab에서 batch 8/16/32로 재학습했던 과정을 코드로도 재현 가능하게 하기 위함
    model = YOLO("yolov8n.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name=f"batch{batch}",   # runs/detect/batch8, batch16, batch32로 구분 저장됨
    )

if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    dataset = download_dataset()
    train_model(dataset, batch=batch_size)
