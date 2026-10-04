import os
import shutil
import sys

from roboflow import Roboflow
from ultralytics import YOLO
from config.config import ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION

# 의미: 4주차 성능 향상 실험에서 사용한 증강·loss 설정을 한 곳에 모아 둠
# 사용 이유: README에 적힌 실험 조건과 실제 학습 코드가 항상 같도록 하기 위함
AUGMENT_ARGS = {
    "copy_paste": 0.3,  # 객체 복사-붙여넣기 증강
    "mixup": 0.1,       # 두 이미지를 섞는 증강
    "cls": 1.0,         # 분류 loss 가중치 (기본값 0.5)
}


def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset


def get_run_name(batch, augment):
    # 의미: 실험 이름을 batch16 / batch16_augmented 형식으로 만듦
    # 사용 이유: config.get_weight_path()가 찾는 폴더 이름과 맞추기 위함
    return f"batch{batch}_augmented" if augment else f"batch{batch}"


def train_model(dataset, epochs=30, batch=16, augment=False):
    run_name = get_run_name(batch, augment)
    extra_args = AUGMENT_ARGS if augment else {}

    model = YOLO("yolov8n.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name=run_name,
        **extra_args,
    )

    # 의미: 학습이 끝나면 runs/detect/<이름>/weights/best.pt를 weights/<이름>/best.pt로 복사
    # 사용 이유: evaluate.py, infer.py, app.py가 weights/ 폴더에서 가중치를 찾기 때문에
    #           손으로 옮기는 단계를 없애기 위함
    os.makedirs(f"weights/{run_name}", exist_ok=True)
    shutil.copy(str(model.trainer.best), f"weights/{run_name}/best.pt")
    return run_name


if __name__ == "__main__":
    # 사용 예: python -m src.train 16            → 기본 학습 (batch16)
    #          python -m src.train 16 augment    → 증강 적용 학습 (batch16_augmented)
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    use_augment = len(sys.argv) > 2 and sys.argv[2] == "augment"
    dataset = download_dataset()
    train_model(dataset, batch=batch_size, augment=use_augment)
