# Week 4 - 토마토 숙성도 탐지 모델 고도화

3주차 YOLOv8n 기반 토마토 숙성도 탐지 모델을 기반으로, 성능 향상 전략 적용과 depth 기반 후처리 기능을 추가한 프로젝트임.

## 프로젝트 구조
week4/
├── app.py
├── config/
│   └── config.py              # 경로, API 키, 클래스별 confidence 임계값 등 설정 중앙 관리
├── src/
│   ├── train.py                 # YOLOv8n 학습 (배치 크기, 증강 옵션 인자화)
│   ├── evaluate.py              # test 데이터 기준 성능 평가 및 그래프 생성
│   ├── infer.py                  # 추론 실행 + 박스 시각화 + depth 후처리 연결
│   ├── depth_processing.py       # 탐지 결과를 depth map/포인트 클라우드로 변환
├── utils/
│   ├── visualize.py              # 탐지 박스 시각화 (클래스별 confidence 필터링 지원)
│   ├── plot_metrics.py           # 학습 곡선 및 클래스별 성능 그래프
│   └── depth_utils.py            # depth map 생성, 포인트 클라우드 생성, 3D 산점도 시각화
├── tests/
│   ├── test_pipeline.py          # evaluate/infer/train 관련 정상·비정상 케이스 테스트
│   ├── test_visualize.py         # visualize 및 클래스별 임계값 필터링 테스트
│   └── test_depth.py             # depth map/포인트 클라우드 생성 테스트
├── weights/
│   ├── batch16_fl0.0/best.pt      # 기존 기본 학습 결과 (gitignore)
│   └── batch16_augmented/best.pt  # 증강 적용 학습 결과 (gitignore)
├── dataset/                       # Roboflow 데이터셋 (gitignore)
├── input_data/, output/           # 입출력 파일 (gitignore, .gitkeep만 유지)
└── .env                           # ROBOFLOW_API_KEY 등 민감정보 (gitignore)

## 데이터셋

- **출처**: [Roboflow - Tomatoes Detection](https://universe.roboflow.com/tomatoes-g0kjm/tomatoes-detection-ml1e1)
- **클래스 (5개)**: `tomato_half_ripe`, `tomato_overripe`, `tomato_ripe`, `tomato_rotten`, `tomato_unripe`
- **테스트셋**: 114장 이미지, 397개 객체 인스턴스

## 이번 주 작업 내용

### 1. 성능 향상 전략 적용

당초 클래스 불균형(`tomato_overripe` 6장/18개) 개선을 위해 **Focal Loss(`fl_gamma`)** 적용을 시도함. 이 인자는 YOLOv5 시절의 레거시 옵션으로 현재 YOLOv8에서는 공식 지원되지 않아 `SyntaxError`가 발생함. 이에 따라 YOLOv8이 공식 지원하는 증강 옵션으로 대체함.

```python
model.train(
    ...,
    copy_paste=0.3,   # 객체를 다른 배경에 복사해 붙여넣는 증강 - 데이터 적은 클래스 보강
    mixup=0.1,        # 두 이미지를 섞어 합성 - 일반화 성능 향상
    cls=1.0,          # 분류 loss 가중치 상향 (기본값 0.5 → 1.0)
)
```

### 2. 클래스별 confidence 임계값 조정

클래스마다 confidence 분포가 달라, 일괄된 임계값(0.5)으로는 특정 클래스(`tomato_half_ripe` 등)가 과소 탐지되는 문제가 있었음. `config.py`에 클래스별 임계값을 정의하고, 가장 낮은 임계값 기준으로 넉넉히 탐지한 뒤 `draw_detections()`에서 클래스별로 재선별하는 방식으로 개선함.

```python
CLASS_CONF_THRESHOLDS = {
    "tomato_half_ripe": 0.25,
    "tomato_ripe": 0.4,
    "tomato_unripe": 0.35,
}
DEFAULT_CONF = 0.5
```

또한 동일한 물체가 서로 다른 클래스 라벨로 중복 탐지되는 문제(예: 한 토마토가 `ripe`와 `half_ripe`로 겹쳐 탐지됨)를 줄이기 위해 `agnostic_nms=True`를 적용함.

### 3. Depth map / 포인트 클라우드 후처리

탐지된 이미지를 대상으로, 그레이스케일 밝기값을 깊이(depth)의 근사치로 사용하는 pseudo-depth 방식을 적용함.

- `cv2.applyColorMap`으로 컬러 depth map 생성함
- 가우시안 블러로 픽셀 단위 노이즈를 완화함
- `(X, Y, Z)` 3D 포인트 클라우드를 생성하고 `.npy`로 저장함
- matplotlib 3D 산점도로 정적 이미지를 저장함 (`visualize_point_cloud_3d`)
- Plotly 기반 인터랙티브 3D 시각화를 Streamlit 앱에 내장 (`utils/depth_utils.py`의 `build_plotly_point_cloud`)

**한계**: 이 방식은 실제 깊이 센서 값이 아니라 밝기 기반 근사치이며, 토마토 표면의 하이라이트(반사광)와 그림자를 각각 "높음"/"낮음"으로 잘못 해석해 실제 토마토의 둥근 3D 형태를 정확히 복원하지는 못함. 

## 실행 방법

```bash
pip install -r requirements.txt

cp .env.example .env
# .env 안에 ROBOFLOW_API_KEY 입력

python -m src.train 16          # 데이터 다운로드 + 학습
python -m src.evaluate 16       # 성능 평가 + 그래프 저장
python -m src.infer batch16_augmented   # 추론 + 박스 시각화 + depth 후처리

streamlit run app.py   # 제품 실행 (이미지 업로드 → 탐지 결과 + depth map + 3D 포인트 클라우드 확인)

python3 -m pytest tests/ -v     # 전체 테스트 실행
```

## 결과

### Batch 16 (기존, fl0.0) vs Batch 16 (증강 적용) 비교

| 지표 | 기존 (batch16) | 증강 적용 (batch16_augmented) |
|---|---|---|
| mAP50 | 0.9008 | 0.9189 |
| mAP50-95 | 0.7591 | 0.7594 |
| Precision | 0.8479 | 0.8435 |
| Recall | 0.8437 | 0.8569 |

### 클래스별 AP50 비교

| 클래스 | 기존 (batch16) | 증강 적용 (batch16_augmented) |
|---|---|---|
| tomato_half_ripe | 0.8694 | 0.9246 |
| tomato_overripe | 0.8882 | 0.9077 |
| tomato_ripe | 0.9064 | 0.9201 |
| tomato_rotten | 0.9049 | 0.8973 |
| tomato_unripe | 0.9350 | 0.9451 |

