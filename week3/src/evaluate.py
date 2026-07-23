import os
from ultralytics import YOLO
from config.config import MODEL_WEIGHT_PATH, DATA_YAML_PATH, TRAIN_RESULTS_CSV, OUTPUT_DATA_DIR
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
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    metrics, class_names = evaluate_model(MODEL_WEIGHT_PATH, DATA_YAML_PATH)

    plot_training_curve(
        TRAIN_RESULTS_CSV,
        save_path=os.path.join(OUTPUT_DATA_DIR, "training_curve.png")
    )
    plot_class_performance(
        metrics, class_names,
        save_path=os.path.join(OUTPUT_DATA_DIR, "class_performance.png")
    )
