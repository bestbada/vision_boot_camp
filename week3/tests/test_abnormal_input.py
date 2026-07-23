import numpy as np
import pytest
import cv2
from src.evaluate import evaluate_model

def test_evaluate_model_invalid_weight_path():
    # 의미: 존재하지 않는 weight 경로를 주면 명확한 예외가 나야 함
    # 사용 이유: 실무에서는 경로가 잘못됐을 때 원인 모를 에러 대신
    #           바로 알아챌 수 있는 예외 처리가 중요함
    with pytest.raises(Exception):
        evaluate_model("weights/존재하지_않는_파일.pt", "dataset/data.yaml")

def test_infer_with_empty_image():
    # 의미: cv2.imread가 실패해서 None이 반환되는 상황(잘못된 이미지 경로)을 흉내냄
    image = None
    with pytest.raises(Exception):
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # None을 넣으면 에러가 나야 정상
