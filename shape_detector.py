import cv2
import numpy as np
import argparse

def preprocess_image(frame, blur_ksize=(5, 5), low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return gray, closed

def classify_shape(approx_curve):
    num_vertices = len(approx_curve)
    shape_name = "Polygon"

    if num_vertices == 3:
        shape_name = "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx_curve)
        aspect_ratio = float(w) / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif num_vertices == 5:
        shape_name = "Pentagon"
    elif num_vertices == 6:
        shape_name = "Hexagon"
    elif num_vertices > 6:
        area = cv2.contourArea(approx_curve)
        perimeter = cv2.arcLength(approx_curve, True)
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            shape_name = "Circle" if circularity > 0.8 else f"{num_vertices}-gon"
                
    return shape_name

def process_frame(frame, min_area=400):
    display_frame = frame.copy()
    _, processed_edges = preprocess_image(frame)
    contours, _ = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.035 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        shape_type = classify_shape(approx)
        x, y, w, h = cv2.boundingRect(approx)

        cv2.drawContours(display_frame, [approx], -1, (0, 255, 0), 2)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        label = f"{shape_type}"
        stats_label = f"A:{int(area)} P:{int(perimeter)}"
        cv2.putText(display_frame, label, (x, max(y - 12, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display_frame, stats_label, (x, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        detected_shapes.append({
            "shape": shape_type,
            "vertices": len(approx),
            "area": area,
            "perimeter": perimeter,
            "bbox": (x, y, w, h)
        })

    return display_frame, detected_shapes

def run_pipeline(source=0):
    if isinstance(source, str) and not source.isdigit():
        image = cv2.imread(source)
        if image is None:
            print(f"Error: Unable to load image at '{source}'")
            return
        processed, shapes = process_frame(image)
        print(f"Detected {len(shapes)} shapes:")
        for s in shapes:
            print(f" - {s["shape"]} | Area: {s["area"]:.1f} | Perimeter: {s["perimeter"]:.1f} | BBox: {s["bbox"]}")
        cv2.imshow("Detected Shapes", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cam_idx = int(source) if str(source).isdigit() else 0
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"Error: Camera index {cam_idx} not accessible.")
            return

        print("Press 'q' to exit live shape detection.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed, _ = process_frame(frame)
            cv2.imshow("Real-Time Shape Detection", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometric Shape Detector")
    parser.add_argument("--source", type=str, default="0", help="Camera index or image path")
    args = parser.parse_args()
    run_pipeline(args.source)