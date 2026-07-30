import os
import cv2
import numpy as np
from utils.depth_utils import generate_depth_map, generate_point_cloud, visualize_point_cloud_3d
from utils.visualize import draw_detections


def process_full_image_to_3d(image, results, output_dir, class_conf_thresholds=None, run_name="result"):
    os.makedirs(output_dir, exist_ok=True)

    depth_map, grayscale = generate_depth_map(image)
    points_3d = generate_point_cloud(grayscale)

    depth_map_with_boxes = draw_detections(depth_map.copy(), results, class_conf_thresholds)

    depth_path = os.path.join(output_dir, f"depth_map_{run_name}.jpg")
    depth_boxed_path = os.path.join(output_dir, f"depth_map_with_boxes_{run_name}.jpg")
    cloud_path = os.path.join(output_dir, f"point_cloud_{run_name}.npy")
    cloud_viz_path = os.path.join(output_dir, f"point_cloud_3d_{run_name}.png")  # 추가

    cv2.imwrite(depth_path, depth_map)
    cv2.imwrite(depth_boxed_path, depth_map_with_boxes)
    np.save(cloud_path, points_3d)
    visualize_point_cloud_3d(points_3d, save_path=cloud_viz_path)  # 추가

    return depth_path, depth_boxed_path, cloud_path, cloud_viz_path
