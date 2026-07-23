import sys
import os
from ultralytics import YOLO
from config.config import get_weight_path, DATA_YAML_PATH, OUTPUT_DATA_DIR
from utils.plot_metrics import plot_training_curve, plot_class_performance


def evaluate_model(weight_path, data_yaml_path):
    model = YOLO(weight_path)
    metrics = model.val(data=data_yaml_path, split="test")

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision(전체 평균): {metrics.box.mp:.4f}")
    print(f"Recall(전체 평균): {metrics.box.mr:.4f}")

    for i, class_name in model.names.items():
        print(f"  {class_name}: AP50 = {metrics.box.ap50[i]:.4f}")

    return metrics, model.names


if __name__ == "__main__":
    # 의미: 터미널에서 "python -m src.evaluate 32"처럼 배치 번호를 인자로 받음
    # 사용 이유: config.py를 안 건드리고, 실행할 때마다 원하는 배치를 골라 테스트하기 위함
    batch_size = sys.argv[1] if len(sys.argv) > 1 else "16"
    weight_path = get_weight_path(batch_size)

    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    print(f"=== batch{batch_size} 모델 평가 ===")
    metrics, class_names = evaluate_model(weight_path, DATA_YAML_PATH)

    plot_training_curve(
        f"runs/detect/batch{batch_size}/results.csv",
        save_path=os.path.join(OUTPUT_DATA_DIR, f"training_curve_batch{batch_size}.png")
    )
    plot_class_performance(
        metrics, class_names,
        save_path=os.path.join(OUTPUT_DATA_DIR, f"class_performance_batch{batch_size}.png")
    )
