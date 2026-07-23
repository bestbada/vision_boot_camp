import glob
import os
import cv2
from ultralytics import YOLO
from config.config import MODEL_WEIGHT_PATH, INPUT_DATA_DIR, OUTPUT_DATA_DIR
from utils.visualize import draw_detections

def find_latest_weight():
    # 의미: 여러 train* 폴더 중 가장 최근 학습된 weight를 자동으로 찾음
    train_dirs = glob.glob('runs/detect/train*/weights/best.pt')
    return max(train_dirs, key=os.path.getmtime) if train_dirs else MODEL_WEIGHT_PATH

def run_inference():
    weight_path = find_latest_weight()
    model = YOLO(weight_path)

    test_image_path = glob.glob(f"{INPUT_DATA_DIR}/*.jpg")[0]
    image = cv2.imread(test_image_path)

    results = model(image, conf=0.2, iou=0.4)
    image = draw_detections(image, results)

    output_path = os.path.join(OUTPUT_DATA_DIR, "output_image.jpg")
    cv2.imwrite(output_path, image)
    print(f"결과 저장 완료: {output_path}")

if __name__ == "__main__":
    run_inference()
