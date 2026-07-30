import sys
import os
from ultralytics import YOLO
from config.config import get_weight_path, DATA_YAML_PATH, OUTPUT_DATA_DIR, normalize_run_name
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
    batch_size = sys.argv[1] if len(sys.argv) > 1 else "16"
    weight_path = get_weight_path(batch_size)

    # 의미: config.py의 normalize_run_name()을 그대로 재사용
    # 사용 이유: get_weight_path()도 내부적으로 이 함수를 쓰고 있으므로,
    #           여기서 직접 if문을 새로 짜지 않고 같은 함수를 불러써서
    #           "숫자면 batch 붙이기" 규칙이 한 곳(config.py)에서만 관리되게 함
    run_name = normalize_run_name(batch_size)

    print(f"=== {run_name} 모델 평가 ===")
    metrics, class_names = evaluate_model(weight_path, DATA_YAML_PATH)

    plot_training_curve(
        f"runs/detect/{run_name}/results.csv",
        save_path=os.path.join(OUTPUT_DATA_DIR, f"training_curve_{run_name}.png")
    )
    plot_class_performance(
        metrics, class_names,
        save_path=os.path.join(OUTPUT_DATA_DIR, f"class_performance_{run_name}.png")
    )
