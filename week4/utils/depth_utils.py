import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map, grayscale


def generate_point_cloud(grayscale, z_scale=0.3):
    # 의미: 픽셀별 밝기 노이즈를 줄이기 위해 가벼운 가우시안 블러 적용,
    #      Z값에 z_scale을 곱해 높낮이 차이를 완화
    smoothed = cv2.GaussianBlur(grayscale, (15, 15), 0)

    h, w = smoothed.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = smoothed.astype(np.float32) * z_scale
    points_3d = np.dstack((X, Y, Z))
    return points_3d


def visualize_point_cloud_3d(points_3d, save_path=None, sample_step=10):
    # 의미: (H, W, 3) 포인트 클라우드를 3D 산점도 정적 이미지로 저장
    # 사용 이유: 전체 픽셀(수십만 개)을 다 그리면 너무 느리고 무거우므로,
    #           sample_step 간격으로 건너뛰며 일부만 뽑아 가볍게 시각화함.
    #           README/PPT 첨부용 정적 이미지가 필요할 때 사용
    sampled = points_3d[::sample_step, ::sample_step]

    X = sampled[:, :, 0].flatten()
    Y = sampled[:, :, 1].flatten()
    Z = sampled[:, :, 2].flatten()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='jet', s=1)

    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    ax.set_zlabel('Z (밝기 기반 깊이)')
    ax.set_title('3D Point Cloud (Depth Approximation)')
    fig.colorbar(scatter, shrink=0.5)

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close(fig)


def build_plotly_point_cloud(points_3d, sample_step=10):
    # 의미: (H, W, 3) 포인트 클라우드를 Plotly의 3D 산점도 Figure 객체로 변환
    # 사용 이유: matplotlib(정적 이미지)과 달리, 브라우저(Streamlit) 안에서
    #           마우스로 회전·확대가 가능한 인터랙티브 그래프가 필요할 때 사용
    sampled = points_3d[::sample_step, ::sample_step]

    X = sampled[:, :, 0].flatten()
    Y = sampled[:, :, 1].flatten()
    Z = sampled[:, :, 2].flatten()

    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=2, color=Z, colorscale='Jet', opacity=0.8)
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X (pixel)',
            yaxis_title='Y (pixel)',
            zaxis_title='Z (밝기 기반 깊이)',
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig
