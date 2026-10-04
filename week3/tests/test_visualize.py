import numpy as np
from utils.visualize import draw_detections

class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [xyxy]
        self.cls = [cls]
        self.conf = [conf]

class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

def test_draw_detections_normal():
    # 의미: 정상적인 탐지 결과가 주어졌을 때 이미지 크기가 그대로 유지되는지 확인
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FakeBox(xyxy=[10, 10, 50, 50], cls=0, conf=0.9)
    result = FakeResult(boxes=[box], names={0: "ripe"})

    output = draw_detections(image, [result])

    assert output.shape == image.shape, "출력 이미지 크기가 입력과 달라짐"
