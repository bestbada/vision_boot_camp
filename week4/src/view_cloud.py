import sys
import numpy as np
import open3d as o3d


def show_point_cloud_interactive(npy_path):
    # 의미: 저장된 .npy 포인트 클라우드 파일을 읽어서 3D 창으로 띄움
    # 사용 이유: matplotlib 정적 이미지와 달리, 마우스로 회전/확대하며
    #           직접 눈으로 확인하고 싶을 때 사용하는 별도의 뷰어 도구
    points_3d = np.load(npy_path)
    points_flat = points_3d.reshape(-1, 3).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)

    # 의미: Z값(밝기 기반 깊이)에 따라 색을 입혀서 높낮이를 시각적으로 구분
    z_vals = points_flat[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
    colors = np.stack([z_norm, 1 - z_norm, np.zeros_like(z_norm)], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # 의미: "python -m src.view_cloud output/depth/point_cloud_batch16.npy"처럼
    #      확인하고 싶은 .npy 경로를 인자로 받음
    if len(sys.argv) < 2:
        print("사용법: python -m src.view_cloud <point_cloud.npy 경로>")
        sys.exit(1)

    npy_path = sys.argv[1]
    show_point_cloud_interactive(npy_path)
