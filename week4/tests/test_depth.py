import os
import numpy as np
import pytest
from utils.depth_utils import generate_depth_map, generate_point_cloud
from src.depth_processing import process_full_image_to_3d


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [xyxy]
        self.cls = [cls]
        self.conf = [conf]


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def test_generate_depth_map_normal():
    crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    depth_map, grayscale = generate_depth_map(crop)
    assert depth_map.shape == crop.shape
    assert grayscale.shape == crop.shape[:2]


def test_generate_depth_map_none_input():
    with pytest.raises(ValueError):
        generate_depth_map(None)


def test_generate_point_cloud_shape():
    grayscale = np.zeros((30, 40), dtype=np.uint8)
    points_3d = generate_point_cloud(grayscale)
    assert points_3d.shape == (30, 40, 3)


def test_process_full_image_to_3d(tmp_path):
    image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    box = FakeBox(xyxy=[10, 10, 50, 50], cls=0, conf=0.9)
    result = FakeResult(boxes=[box], names={0: "tomato_ripe"})

    # 의미: 3D 시각화 이미지 경로(cloud_viz_path)까지 4개를 받도록 수정
    depth_path, depth_boxed_path, cloud_path, cloud_viz_path = process_full_image_to_3d(
        image, [result], str(tmp_path)
    )

    assert os.path.exists(depth_path)
    assert os.path.exists(depth_boxed_path)
    assert os.path.exists(cloud_viz_path)   # 3D 시각화 이미지도 생성됐는지 확인

    cloud = np.load(cloud_path)
    assert cloud.shape == (100, 150, 3)
