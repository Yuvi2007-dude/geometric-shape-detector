# Geometric Shape Detector & Geometric Measurements

A Computer Vision application built with Python and OpenCV that detects geometric shapes, classifies their types, traces contours, and computes spatial measurements in real-time or from static images.

## Features
- **Preprocessing:** Grayscale conversion, Gaussian blurring, Canny edge detection, and morphological gap closing.
- **Contour Filtering:** Eliminates noise contours based on area thresholding.
- **Polygon Classification:** Approximates contours using the Douglas-Peucker algorithm (`approxPolyDP`) and classifies triangles, squares, rectangles, pentagons, hexagons, and circles.
- **Geometric Measurements:** Computes area, perimeter, and bounding box dimensions.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application
- **Generate Sample Test Image:**
  ```bash
  python create_test_image.py
  ```
- **Static Image Mode:**
  ```bash
  python shape_detector.py --source sample_shapes.png
  ```
- **Live Video Mode (Webcam):**
  ```bash
  python shape_detector.py --source 0
  ```
