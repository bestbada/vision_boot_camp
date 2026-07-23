import sys
import glob
import os
import cv2
from ultralytics import YOLO
from config.config import get_weight_path, INPUT_DATA_DIR, OUTPUT_DATA_DIR
from utils.visualize import draw_detections


def run_inference(weight_path):
    # 의미: weight_path를 파라미터로 받게 바꿈
    # 사용 이유: find_latest_weight()처럼 "추측"하지 않고,
    #           evaluate.py와 동일하게 "어떤 배치를 쓸지" 호출하는 쪽에서 명시하게 하기 위함
    model = YOLO(weight_path)

    test_image_path = glob.glob(f"{INPUT_DATA_DIR}/*.jpg")[0]
    image = cv2.imread(test_image_path)

    results = model(image, conf=0.2, iou=0.4)
    image = draw_detections(image, results)

    output_path = os.path.join(OUTPUT_DATA_DIR, "output_image.jpg")
    cv2.imwrite(output_path, image)
    print(f"결과 저장 완료: {output_path}")


if __name__ == "__main__":
    # 의미: "python -m src.infer 32"처럼 배치 번호를 인자로 받음
    batch_size = sys.argv[1] if len(sys.argv) > 1 else "16"
    weight_path = get_weight_path(batch_size)

    print(f"=== batch{batch_size} 모델로 추론 실행 ===")
    run_inference(weight_path)
