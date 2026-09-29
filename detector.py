import os
import cv2
import numpy as np

def _demo_detection(image):
    """
    Demo fallback used when a trained custom model is unavailable.
    It finds bright/high-contrast candidate regions so the complete
    dashboard can be demonstrated. It is NOT a real plastic classifier.
    """
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 100, 255]))
    mask2 = cv2.inRange(hsv, np.array([5, 20, 120]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = max(100, int(h * w * 0.0005))

    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > h * w * 0.30:
            continue

        area_percent = area_px / (h * w) * 100.0
        confidence = min(0.95, 0.55 + min(area_percent / 10.0, 0.35))

        candidates.append({
            "class": "possible_plastic",
            "confidence": round(confidence, 3),
            "x": x,
            "y": y,
            "width": bw,
            "height": bh,
            "area_percent": round(area_percent, 4),
            "latitude": round(10.0 + (1 - y / max(h, 1)) * 0.5, 6),
            "longitude": round(76.0 + (x / max(w, 1)) * 0.5, 6)
        })

    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)[:10]

    annotated = image.copy()
    for d in candidates:
        x, y, bw, bh = d["x"], d["y"], d["width"], d["height"]
        cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (255, 0, 0), 3)
        label = f'plastic {d["confidence"]:.2f}'
        cv2.putText(
            annotated, label, (x, max(20, y-8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

    return {
        "detections": candidates,
        "annotated_image": annotated,
        "mode": "demo"
    }


def detect_plastic(image_rgb, model_path="weights/best.pt"):
    """
    Run a custom Ultralytics YOLO model if weights/best.pt exists.
    The custom model should contain a class representing floating plastic.
    """
    if not os.path.exists(model_path):
        return _demo_detection(image_rgb)

    try:
        from ultralytics import YOLO
    except ImportError:
        return _demo_detection(image_rgb)

    model = YOLO(model_path)
    results = model.predict(source=image_rgb, conf=0.25, verbose=False)

    result = results[0]
    annotated = result.plot()

    h, w = image_rgb.shape[:2]
    detections = []

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            area_percent = (bw * bh) / (w * h) * 100.0

            # These coordinates are demo georeferencing.
            # Replace with real satellite georeferencing in deployment.
            lat = 10.0 + (1 - ((y1 + y2) / 2) / h) * 0.5
            lon = 76.0 + (((x1 + x2) / 2) / w) * 0.5

            detections.append({
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 3),
                "x": x1,
                "y": y1,
                "width": bw,
                "height": bh,
                "area_percent": round(area_percent, 4),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6)
            })

    return {
        "detections": detections,
        "annotated_image": annotated,
        "mode": "trained_model"
    }
