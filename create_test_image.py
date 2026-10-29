import cv2
import numpy as np

canvas = np.ones((600, 800, 3), dtype="uint8") * 255

# 1. Triangle
pts_tri = np.array([[120, 180], [60, 280], [180, 280]], np.int32)
cv2.drawContours(canvas, [pts_tri], 0, (0, 0, 0), -1)

# 2. Square
cv2.rectangle(canvas, (260, 180), (360, 280), (0, 0, 0), -1)

# 3. Rectangle
cv2.rectangle(canvas, (440, 200), (600, 260), (0, 0, 0), -1)

# 4. Hexagon
center, r = (700, 230), 50
hex_pts = np.array([[int(center[0] + r * np.cos(np.pi / 3 * i)),
                     int(center[1] + r * np.sin(np.pi / 3 * i))] for i in range(6)], np.int32)
cv2.drawContours(canvas, [hex_pts], 0, (0, 0, 0), -1)

# 5. Circle
cv2.circle(canvas, (200, 440), 55, (0, 0, 0), -1)

# 6. Pentagon
center_p, rp = (450, 440), 55
pent_pts = np.array([[int(center_p[0] + rp * np.cos(np.pi / 2.5 * i - np.pi/2)),
                      int(center_p[1] + rp * np.sin(np.pi / 2.5 * i - np.pi/2))] for i in range(5)], np.int32)
cv2.drawContours(canvas, [pent_pts], 0, (0, 0, 0), -1)

cv2.imwrite("sample_shapes.png", canvas)
print("sample_shapes.png created successfully!")