# Week 3 - 토마토 숙성도 탐지 프로젝트

week3/
├── src/
│   ├── train.py       # 모델 학습
│   ├── evaluate.py    # 성능 평가 및 시각화
│   └── infer.py       # 추론 실행
├── utils/
│   ├── visualize.py   # 탐지 결과 시각화 함수
│   └── plot_metrics.py # 성능 그래프 함수
├── config/
│   └── config.py       # 경로/설정값 중앙 관리
├── tests/
│   ├── test_visualize.py       # 정상 케이스 테스트
│   └── test_abnormal_input.py  # 비정상 입력 테스트
├── weights/best.pt      # 학습된 모델 (gitignore)
├── dataset/              # Roboflow 데이터셋 (gitignore)
├── input_data/, output/  # 입출력 (gitignore, .gitkeep만 유지)
└── .env                  # API 키 등 민감정보 (gitignore)

## 실행 방법
\`\`\`bash
pip install -r requirements.txt
python -m src.train      # 학습
python -m src.evaluate   # 평가 + 그래프
python -m src.infer      # 추론
pytest tests/            # 테스트
\`\`\`

## 결과
mAP50: 0.9008 / Precision: 0.8479 / Recall: 0.8437

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

`output/training_curve.png` — epoch별 precision/recall 변화
`output/class_performance.png` — 클래스별 precision/recall 비교

## 한계 및 다음 단계

- `tomato_overripe` 클래스는 테스트셋에 6장(18개 인스턴스)뿐이라 다른 클래스 대비 데이터가 적음. 추가 데이터 확보 시 성능 개선 여지 있음.
- 4주차에는 이 모델을 FastAPI 기반 API로 배포하여, 이미지를 업로드하면 숙성도 판별 결과를 반환하는 서비스로 확장할 예정.
