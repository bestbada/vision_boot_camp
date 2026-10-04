import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from config.config import get_weight_path, CLASS_CONF_THRESHOLDS, DEFAULT_CONF
from utils.visualize import draw_detections
from src.depth_processing import process_full_image_to_3d
from utils.depth_utils import build_plotly_point_cloud

st.title("🍅 토마토 숙성도 탐지 서비스")
st.write("이미지를 업로드하면 숙성도를 탐지하고, depth map까지 함께 생성합니다.")

uploaded_file = st.file_uploader("토마토 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    model = YOLO(get_weight_path("batch16_augmented"))
    lowest_conf = min(list(CLASS_CONF_THRESHOLDS.values()) + [DEFAULT_CONF])
    results = model(image, conf=lowest_conf, iou=0.4, agnostic_nms=True)

    boxed_image = draw_detections(image.copy(), results, CLASS_CONF_THRESHOLDS)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("탐지 결과")
        st.image(cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB))

    depth_path, depth_boxed_path, cloud_path, cloud_viz_path = process_full_image_to_3d(
        image, results, "output/depth", CLASS_CONF_THRESHOLDS, run_name="streamlit"
    )
    with col2:
        st.subheader("Depth Map")
        st.image(depth_boxed_path)
    # 신규: 인터랙티브 3D 포인트 클라우드
    st.subheader("3D 포인트 클라우드 (마우스로 회전/확대 가능)")
    cloud_data = np.load(cloud_path)
    fig = build_plotly_point_cloud(cloud_data)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("탐지된 클래스 개수")
    counts = {}
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            counts[label] = counts.get(label, 0) + 1
    st.write(counts)
