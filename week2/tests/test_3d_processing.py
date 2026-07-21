import pytest
import numpy as np

# src 폴더의 depth_to_3d.py 파일에서 generate_depth_map 함수를 가져옴.
from src.depth_to_3d import generate_depth_map

# --- [테스트 1] 형태(Shape) 검증 ---
def test_generate_depth_map_shape():
    # 100x100 크기의 가짜 컬러(3채널) 이미지를 메모리에 만듦.
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 만들어둔 가짜 데이터를 함수에 넣음.
    result = generate_depth_map(dummy_image)
    
    # 결과물이 100x100 크기의 2차원(흑백) 배열인지 검증함.
    assert result.shape == (100, 100)

# --- [테스트 2] 빈 데이터(None) 입력 예외 검증 ---
def test_generate_depth_map_none_input():
    # 이미지 대신 빈 데이터(None)를 강제로 집어넣고 ValueError가 발생하는지 검사함.
    with pytest.raises(ValueError):
        generate_depth_map(None)

# --- [테스트 3] 반환 데이터 타입(Type) 검증 ---
def test_generate_depth_map_return_type():
    # 50x50 크기의 가짜 이미지를 만듦.
    dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
    
    # 함수에 데이터를 넣어 결과를 받음.
    result = generate_depth_map(dummy_image)
    
    # 결과물이 NumPy의 다차원 배열(ndarray) 형태인지 검증함.
    assert isinstance(result, np.ndarray)

# --- [테스트 4] 잘못된 데이터 입력 예외 검증 ---
def test_generate_depth_map_invalid_type():
    # 이미지 배열 대신 "image"라는 텍스트(문자열)를 함수에 강제로 넣고 Exception이 발생하는지 검사함.
    with pytest.raises(Exception):
        generate_depth_map("이것은 이미지가 아닙니다")

# --- [테스트 5] 극단적인 픽셀 값 변환 정확도 검증 ---
def test_generate_depth_map_extreme_colors():
    # 모든 픽셀이 255인 완전한 순백색 이미지를 만듦.
    white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    
    # 모든 픽셀이 0인 완전한 칠흑색 이미지를 만듦.
    black_image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # 백색/흑색 이미지를 각각 함수에 넣어 변환함.
    result_white = generate_depth_map(white_image)
    result_black = generate_depth_map(black_image)
    
    # 결과 배열의 모든 숫자가 흰색은 255, 검은색은 0인지 검증함.
    assert np.all(result_white == 255)
    assert np.all(result_black == 0)
