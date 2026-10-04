# Vision Boot Camp

컴퓨터 비전 부트캠프에서 4주간 수행한 과제 모음입니다.  
OpenCV 이미지 전처리에서 시작해, YOLOv8n으로 **토마토 숙성도 5단계 탐지 모델**을 학습·개선하고 Streamlit 데모 앱까지 만들었습니다.

## ⭐ 대표 결과: 토마토 숙성도 탐지 (week3 → week4)

- **데이터**: Roboflow Tomatoes Detection (5개 클래스, 테스트셋 114장 / 397개 객체)
- **모델**: YOLOv8n
- **week3**: 배치 크기 8 / 16 / 32 비교 실험
- **week4**: 증강(`copy_paste`, `mixup`)과 분류 loss 가중치 조정, 클래스별 confidence 임계값, `agnostic_nms`로 중복 탐지 감소, Streamlit 데모 앱

| 실험 | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| week3 baseline (batch16) | 0.9008 | 0.7591 | 0.8479 | 0.8437 |
| week4 증강 적용 (batch16) | 0.9189 | 0.7594 | 0.8435 | 0.8569 |

> 각 수치는 테스트셋 1회 학습 결과입니다. 세부 내용과 한계는 각 주차 README에 정리했습니다.

## 📂 주차별 과제

| 주차 | 주제 | 핵심 내용 |
|---|---|---|
| [week1](./week1) | 이미지 전처리 및 이상치 필터링 | `food101` 스트리밍, 밝기·윤곽선 면적 기준 불량 이미지 제거 |
| [week2](./week2) | 2D 이미지의 3D 시각화 | 밝기 기반 의사(pseudo) 깊이, 3D 산점도, `pytest` 테스트 5개 |
| [week3](./week3) | 토마토 숙성도 탐지 | YOLOv8n 학습, 배치 크기별 성능 비교 |
| [week4](./week4) | 탐지 모델 고도화 | 증강·임계값 조정, pseudo-depth 후처리, Streamlit 앱 |

## ⚙️ 실행

각 주차 폴더마다 실행 환경이 다르므로, 해당 폴더로 이동해 그 폴더의 `requirements.txt`와 README를 따르세요.

```bash
cd week4
pip install -r requirements.txt
cp .env.example .env        # ROBOFLOW_API_KEY 입력
streamlit run app.py
```
