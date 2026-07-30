from roboflow import Roboflow
from ultralytics import YOLO
from config.config import ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION


def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset


def train_model(dataset, epochs=30, batch=16, fl_gamma=0.0):
    # 의미: fl_gamma 파라미터 추가
    # 사용 이유: 0보다 큰 값을 주면 분류(classification) loss가 Focal Loss로 전환되어,
    #           tomato_overripe처럼 데이터가 적은 클래스에 더 집중해서 학습하게 됨
    model = YOLO("yolov8n.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        fl_gamma=fl_gamma,
        name=f"batch{batch}_fl{fl_gamma}",   # 예: batch16_fl0.0, batch16_fl1.5
    )


if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    fl_gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    dataset = download_dataset()
    train_model(dataset, batch=batch_size, fl_gamma=fl_gamma)
