import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def generate_depth_map(image):
    # 입력된 이미지 데이터가 존재하는지 검사함.
    if image is None:
        raise ValueError("이미지 데이터가 없습니다.")
    
    # BGR 컬러 이미지를 흑백(Grayscale)으로 변환함.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def save_rviz_style_pointcloud(image, depth_map, save_path):
    # 픽셀을 5칸씩 건너뛰며 읽도록 간격을 설정함.
    step = 5
    
    # 이미지의 세로(h)와 가로(w) 길이를 추출함.
    h, w = depth_map.shape
    
    # 2차원 평면의 X, Y 좌표망(격자)을 생성함.
    x, y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    
    # 흑백 이미지의 밝기 값을 Z 좌표(높이/깊이)로 가져옴.
    z = depth_map[::step, ::step]
    
    # 3D 그래프를 그릴 도화지(Figure)를 만들고 배경을 검은색으로 칠함.
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # 원래 색상 대신 Z축 배열(z.flatten())을 넣고, cmap='jet'를 사용하여 깊이에 따라 색이 변하게 만듦.
    ax.scatter(x, y, z, c=z.flatten(), cmap='jet', s=2)
    
    # Y축을 뒤집음.
    ax.invert_yaxis()
    
    # 3D 뷰어 창을 화면에 띄우는 대신, 인자로 받은 경로(save_path)에 캡처 이미지로 저장함.
    plt.savefig(save_path, facecolor=fig.get_facecolor())
    
    # 메모리 누수를 방지하기 위해 도화지를 닫음.
    plt.close(fig)

def main():
    # 결과물을 저장할 3개의 폴더 이름을 지정함.
    dir_preprocessed = "results_1_preprocessed"
    dir_colormap = "results_2_colormap"
    dir_pointcloud = "results_3_pointcloud"
    
    # 폴더들을 생성함.
    os.makedirs(dir_preprocessed, exist_ok=True)
    os.makedirs(dir_colormap, exist_ok=True)
    os.makedirs(dir_pointcloud, exist_ok=True)
    
    # 전처리된 사진들이 있는 폴더 경로를 설정함.
    folder_path = "../week1/preprocessed_samples"
    
    # 해당 폴더 안의 모든 jpg 파일 경로를 가져와 정렬함.
    img_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    img_paths.sort()
    
    if not img_paths:
        print(f"{folder_path} 폴더에 jpg 파일이 없습니다.")
        return

    # 최대 5장까지 순서대로 처리함.
    for i, img_path in enumerate(img_paths[:5]):
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        file_name = os.path.basename(img_path)
        
        # 1. 전처리만 된 사진 저장
        save_path_1 = os.path.join(dir_preprocessed, file_name)
        cv2.imwrite(save_path_1, image)
        
        # 2. 컬러맵 변환 사진 저장
        gray = generate_depth_map(image)
        colormap_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        save_path_2 = os.path.join(dir_colormap, file_name)
        cv2.imwrite(save_path_2, colormap_img)
        
        # 3. 3D 포인트 클라우드 캡처 저장
        save_path_3 = os.path.join(dir_pointcloud, file_name)
        save_rviz_style_pointcloud(image, gray, save_path_3)
        
        print(f"[{i+1}/5] {file_name} -> 3가지 버전으로 저장 완료함.")

# 이 파일이 직접 실행될 때만 main() 함수를 가동하도록 막아줌.
if __name__ == "__main__":
    main()
