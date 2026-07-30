import cv2


def draw_detections(image, results, class_conf_thresholds=None):
    # 의미: class_conf_thresholds가 주어지면 클래스별 기준으로 박스를 걸러내고,
    #      안 주어지면(None) 기존처럼 전부 그림
    # 사용 이유: 기존 코드(및 기존 테스트)와의 하위 호환을 유지하면서
    #           Colab에서 확인한 클래스별 임계값 기능을 추가하기 위함
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if class_conf_thresholds is not None:
                threshold = class_conf_thresholds.get(label, 0.5)
                if confidence < threshold:
                    continue  # 이 클래스 기준 미달이면 그리지 않고 건너뜀

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
