# Week 3 - 토마토 숙성도 탐지 프로젝트

Roboflow의 Tomatoes Detection 데이터셋으로 YOLOv8n 모델을 학습시켜, 이미지 속 토마토를 5단계 숙성도로 분류하고 탐지하는 프로젝트입니다.

week3/
├── config/
│   └── config.py            # 경로, API 키 등 설정값 중앙 관리
├── src/
│   ├── train.py              # 배치 크기별 YOLOv8n 학습 (python -m src.train <batch>)
│   ├── evaluate.py           # 배치별 성능 평가 및 그래프 생성 (python -m src.evaluate <batch>)
│   └── infer.py              # 배치별 모델로 추론 실행 (python -m src.infer <batch>)
├── utils/
│   ├── visualize.py
│   └── plot_metrics.py
├── tests/
│   ├── test_pipeline.py
│   └── test_visualize.py
├── weights/
│   ├── batch8/best.pt
│   ├── batch16/best.pt
│   └── batch32/best.pt       # 배치별 학습된 가중치 (gitignore)
├── runs/detect/batch8, batch16, batch32/  # 배치별 학습 로그 (gitignore)
├── dataset/, input_data/, output/         # (gitignore)
└── .env

## 데이터셋

- **출처**: [Roboflow - Tomatoes Detection](https://universe.roboflow.com/tomatoes-g0kjm/tomatoes-detection-ml1e1)
- **클래스 (5개)**: `tomato_half_ripe`, `tomato_overripe`, `tomato_ripe`, `tomato_rotten`, `tomato_unripe`
- **테스트셋**: 114장 이미지, 397개 객체 인스턴스

## 실행 방법

```bash
pip install -r requirements.txt

# .env 파일 생성 (최초 1회)
cp .env.example .env
# .env 안에 ROBOFLOW_API_KEY 입력

python -m src.train      # 데이터 다운로드 + 모델 학습
python -m src.evaluate 32   # test 데이터로 성능 평가 + 그래프 저장
python -m src.evaluate 16
python -m src.evaluate 8

python -m src.infer 32      # 임의 이미지로 탐지 실행
python -m src.infer 16
python -m src.infer 8

python3 -m pytest tests/ -v  # 테스트 실행
```

## 배치 크기별 성능 비교

| Batch Size | mAP50 | mAP50-95 | Precision | Recall |
|------------|-------|----------|-----------|--------|
|      8     | 0.9078|  0.7642  |  0.8203   | 0.8511 |
|      16    | 0.9008|  0.7591  |  0.8479   | 0.8437 |
|      32    | 0.9006|  0.7560  |  0.8034   | 0.8448 |

- 배치가 작을수록(8) mAP50-95가 가장 높고 Recall도 우수함
- 배치가 클수록(32) Precision이 다소 떨어짐

