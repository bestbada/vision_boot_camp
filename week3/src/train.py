from roboflow import Roboflow
from ultralytics import YOLO
from config.config import ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION

def download_dataset():
    # 의미: Roboflow에서 데이터셋을 내려받음
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset

def train_model(dataset, epochs=30):
    # 의미: YOLOv8n 모델을 지정한 epoch만큼 학습시킴
    model = YOLO("yolov8n.pt")
    model.train(data=f"{dataset.location}/data.yaml", epochs=epochs, imgsz=640)

if __name__ == "__main__":
    dataset = download_dataset()
    train_model(dataset)
