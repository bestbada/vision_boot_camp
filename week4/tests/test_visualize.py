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
    
def test_draw_detections_filters_by_class_threshold():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    box_pass = FakeBox(xyxy=[10, 10, 30, 30], cls=0, conf=0.3)
    box_fail = FakeBox(xyxy=[40, 40, 60, 60], cls=1, conf=0.3)

    result = FakeResult(boxes=[box_pass, box_fail], names={0: "tomato_half_ripe", 1: "tomato_ripe"})

    thresholds = {"tomato_half_ripe": 0.25, "tomato_ripe": 0.4}
    output = draw_detections(image.copy(), [result], class_conf_thresholds=thresholds)

    # 의미: cv2.rectangle은 테두리만 그리므로, 사각형 "안쪽"이 아니라
    #      실제로 선이 그려지는 가장자리(y=10, x=10~30 구간)를 확인해야 함
    # 사용 이유: box_pass는 그려져야 하니 위쪽 테두리 선(row=10)에 초록색이 있어야 하고,
    #           box_fail은 아예 안 그려져야 하니 그 영역(40~60)은 전부 검은색이어야 함
    assert output[10, 15].sum() > 0        # half_ripe 박스 테두리는 그려짐
    assert output[40, 50].sum() == 0       # ripe 박스는 그려지지 않아 여전히 검은색


def test_draw_detections_no_threshold_draws_all():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FakeBox(xyxy=[10, 10, 30, 30], cls=0, conf=0.1)
    result = FakeResult(boxes=[box], names={0: "tomato_ripe"})

    output = draw_detections(image.copy(), [result])

    # 의미: 마찬가지로 테두리 선 위의 점(row=10)을 확인
    assert output[10, 15].sum() > 0
