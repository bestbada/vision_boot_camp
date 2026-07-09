import cv2
import numpy as np
from datasets import load_dataset
import os

# 1. 환경 설정
output_dir = 'preprocessed_samples'
os.makedirs(output_dir, exist_ok=True)

# 2. 데이터셋 로드 (알짜배기 5장이 모일 때까지 스트리밍)
dataset = load_dataset("ethz/food101", split="train", streaming=True)
count = 0

for example in dataset:
    # 이미지를 OpenCV용 배열로 변환
    img = np.array(example['image'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 크기 조정 (224x224) 먼저 수행 (크기 기준 일원화)
    img = cv2.resize(img, (224, 224))
    
    # Grayscale 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # [심화 문제 1] 너무 어두운 이미지 제거 (평균 밝기 40 미만이면 버림)
    avg_brightness = np.mean(gray)
    if avg_brightness < 40:
        continue  # 어두운 이미지는 무시하고 다음 데이터로 점프
    
    # [심화 문제 2] 객체 크기가 너무 작은 이미지 제거 (윤곽선 면적 합산 150 미만이면 버림)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
    if total_contour_area < 150:
        continue  # 객체가 너무 작으면 무시하고 다음 데이터로 점프
    
    # [기본 문제] 색상 정규화 (0~1 실수형 변환)
    img_float = gray.astype(np.float32) / 255.0
    
    # [기본 문제] 노이즈 제거 (Gaussian Blur)
    blur = cv2.GaussianBlur((img_float * 255).astype(np.uint8), (5, 5), 0)
    
    # [기본 문제/데이터 증강] 중심점 기준 15도 회전
    h, w = blur.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(blur, matrix, (w, h))
    
    # [기본 문제/데이터 증강] 좌우 반전
    flipped = cv2.flip(rotated, 1)
    
    # [검증 통과] 이상치 검수를 완벽히 통과한 시점에서만 파일 저장 및 카운트 증가
    cv2.imwrite(os.path.join(output_dir, f'preprocessed_{count}.jpg'), flipped)
    
    count += 1
    if count >= 5: 
        break

print("전처리 완료: preprocessed_samples 폴더를 확인하세요.")
