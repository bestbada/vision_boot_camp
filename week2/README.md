# [2주 차] 2D 이미지의 3D Point Cloud 변환 및 결과물 자동화

## 📌 프로젝트 개요
본 프로젝트는 2D 컬러 이미지 데이터를 기반으로 명암(Grayscale)을 추출하여 Z축 깊이(Depth) 맵을 생성하고, 이를 3D Point Cloud 형태로 시각화 및 자동 저장하는 파이프라인입니다.
실무 환경을 고려하여 소스 코드와 테스트 코드를 분리하였으며, `pytest`를 통한 단위 테스트(Unit Test)를 도입했습니다.

## 🚀 핵심 구현 내용
1. **실무형 모듈화 및 단위 테스트**
   - 구동 코드(`src`)와 테스트 코드(`tests`)의 폴더 구조를 분리함.
   - `import` 방식을 사용하여 독립적인 모듈 테스트 환경을 구축함.
   - 예외 처리(None 입력 방어), 타입(Type) 및 형태(Shape) 검증, 극단적 색상 변환 검증 테스트 수행.
2. **3D Point Cloud 시각화 및 Z축 색상 매핑**
   - 흑백 이미지의 밝기 값을 Z축(깊이) 데이터로 치환함.
   - matplotlib의 `scatter`와 `cmap='jet'`를 활용하여, Z축 깊이에 따라 색상이 변하는(가장 깊은 곳: 파란색, 가장 얕은 곳: 빨간색) 직관적인 시각화 구현.
3. **결과물 자동 분류 및 저장**
   - `preprocessed_samples` 폴더의 모든 이미지를 일괄 처리함.
   - 처리된 결과를 3가지 형태(원본, 2D 컬러맵, 3D Point Cloud)로 각각의 지정된 폴더에 자동 분류하여 저장함.

## 📁 디렉토리 구조
```text
week2/
├── src/
│   ├── __init__.py
│   └── save_results.py      # 메인 구동 및 결과 자동 저장 스크립트
├── tests/
│   ├── __init__.py
│   └── test_3d_processing.py # 단위 테스트 스크립트
├── results_1_preprocessed/  # 전처리 원본 이미지 저장 폴더
├── results_2_colormap/      # JET 컬러맵 적용 이미지 저장 폴더
├── results_3_pointcloud/    # 3D Point Cloud 캡처 저장 폴더
└── README.md
