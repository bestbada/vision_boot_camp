# Week 3 - 토마토 숙성도 탐지 프로젝트

Roboflow의 Tomatoes Detection 데이터셋으로 YOLOv8n 모델을 학습시켜, 이미지 속 토마토를 5단계 숙성도로 분류하고 탐지하는 프로젝트입니다.

## 프로젝트 구조

week3/
├── config/
│ └── config.py # 경로, API 키 등 설정값 중앙 관리
├── src/
│ ├── train.py # Roboflow 데이터셋 다운로드 및 YOLOv8n 학습
│ ├── evaluate.py # test 데이터 기준 성능 평가 및 그래프 생성
│ └── infer.py # 학습된 모델로 이미지에서 탐지 실행
├── utils/
│ ├── visualize.py # 탐지 박스/라벨을 이미지에 그리는 함수
│ └── plot_metrics.py # 학습 곡선 및 클래스별 성능 그래프 함수
├── tests/
│ ├── test_pipeline.py # evaluate/infer 관련 정상·비정상 케이스 테스트
│ └── test_visualize.py # visualize 관련 정상 케이스 테스트
├── weights/best.pt # 학습된 모델 가중치 (gitignore)
├── dataset/ # Roboflow 데이터셋 (gitignore)
├── input_data/, output/ # 입출력 파일 (gitignore, .gitkeep만 유지)
└── .env # ROBOFLOW_API_KEY 등 민감정보 (gitignore)

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
python -m src.evaluate   # test 데이터로 성능 평가 + 그래프 저장
python -m src.infer      # 임의 이미지로 탐지 실행

python3 -m pytest tests/  # 테스트 실행
```

## 결과

전체 평균 (test set 기준):

| 지표 | 값 |
|---|---|
| mAP50 | 0.9008 |
| mAP50-95 | 0.7591 |
| Precision | 0.8479 |
| Recall | 0.8437 |

클래스별 AP50:

| 클래스 | AP50 |
|---|---|
| tomato_half_ripe | 0.8694 |
| tomato_overripe | 0.8882 |
| tomato_ripe | 0.9064 |
| tomato_rotten | 0.9049 |
| tomato_unripe | 0.9350 |

