import sys
import glob
import os
import cv2
from ultralytics import YOLO
from config.config import get_weight_path, INPUT_DATA_DIR, OUTPUT_DATA_DIR, CLASS_CONF_THRESHOLDS, DEFAULT_CONF
from utils.visualize import draw_detections
from src.depth_processing import process_full_image_to_3d

def run_inference(weight_path, run_name="result"):
    # 의미: run_name 파라미터 추가
    # 사용 이유: 배치/실험별로 결과 파일이 서로 덮어쓰지 않고 구분되어 남도록 하기 위함
    test_image_path = glob.glob(f"{INPUT_DATA_DIR}/*.jpg")[0]
    image = cv2.imread(test_image_path)

    model = YOLO(weight_path)

    lowest_conf = min(list(CLASS_CONF_THRESHOLDS.values()) + [DEFAULT_CONF])
    results = model(image, conf=lowest_conf, iou=0.4, agnostic_nms=True)

    boxed_image = draw_detections(image.copy(), results, CLASS_CONF_THRESHOLDS)

    # 의미: 파일명에 run_name을 붙여서 구분
    output_path = os.path.join(OUTPUT_DATA_DIR, f"output_image_{run_name}.jpg")
    cv2.imwrite(output_path, boxed_image)

    depth_path, depth_boxed_path, cloud_path, cloud_viz_path = process_full_image_to_3d(
        image, results, os.path.join(OUTPUT_DATA_DIR, "depth"),
        CLASS_CONF_THRESHOLDS, run_name=run_name
    )
    print(f"point cloud 3D 시각화: {cloud_viz_path}")
    print(f"결과 저장 완료: {output_path}")
    print(f"depth map: {depth_path}")
    print(f"depth map (박스 포함): {depth_boxed_path}")
    print(f"point cloud: {cloud_path}")


if __name__ == "__main__":
    batch_size = sys.argv[1] if len(sys.argv) > 1 else "16"
    weight_path = get_weight_path(batch_size)

    print(f"=== {batch_size} 모델로 추론 실행 ===")   # batchbatch 중복도 같이 수정
    run_inference(weight_path, run_name=str(batch_size))
