import cv2
import numpy as np
import os

# 결과물을 저장할 폴더 생성
os.makedirs("output", exist_ok=True)

# 0부터 4까지 총 5번 반복하는 for문 (i 값은 0, 1, 2, 3, 4로 변함)
for i in range(5):
    
    # 1. 원본 이미지 불러오기 (f-string을 사용하여 파일명 동적 변경)
    # 의미/이유: 변수 i가 바뀔 때마다 preprocessed_0.jpg, preprocessed_1.jpg 순서로 경로가 자동으로 바뀝니다.
    image_path = f'/home/lee/boot_camp/vision_boot_camp/week1/preprocessed_samples/preprocessed_{i}.jpg'
    image = cv2.imread(image_path)

    # 이미지 로드 실패 시 예외 처리
    if image is None:
        print(f"오류: {image_path} 경로에서 이미지를 찾을 수 없습니다. 다음 사진으로 넘어갑니다.")
        continue  # 의미/이유: 하나가 오류 나더라도 프로그램 전체를 종료(exit)하지 않고, 그 다음 사진 작업을 계속 진행하게 합니다.

    # 2. 그레이스케일 변환
    # 의미/이유: 깊이(Depth)를 추정하려면 색상 정보(RGB)보다 빛의 밝기(명암) 정보가 중요하기 때문에 흑백으로 변환합니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Depth Map (가상 깊이 맵) 생성
    # 의미/이유: 흑백 이미지에 COLORMAP_JET 필터를 씌워 깊이감을 붉은색~푸른색으로 시각화합니다.
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # 4. [심화] 3D 포인트 클라우드 좌표 생성
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32) 
    points_3d = np.dstack((X, Y, Z))

    # 5. 결과 이미지 저장 (결과 파일명도 숫자에 맞춰 다르게 저장되도록 설정)
    # 의미/이유: 출력 파일명도 depth_map_result_0.jpg 처럼 각각 저장되게 하여 덮어쓰기를 방지합니다.
    output_file = f'output/depth_map_result_{i}.jpg'
    cv2.imwrite(output_file, depth_map)

    # 각 사진마다 정상 출력되었는지 확인하는 메시지
    print(f"[{i+1}/5] 사진 변환 완료! (배열 형태: {points_3d.shape}) -> {output_file} 저장 성공")

print("\n 5장의 사진이 모두 성공적으로 변환 및 저장되었습니다!")
