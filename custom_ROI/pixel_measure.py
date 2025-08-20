
import cv2
import numpy as np
import math

# === Parameters ===
VIDEO_PATH = "/Users/anthapuvivekanandareddy/Desktop/speed_module/Input_data/speed_video_1.mp4"
ROI_POLYGON = [(902, 599), (476, 669), (1186, 997), (1550, 765)]

# === Load First Frame ===
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read video frame.")
    exit()

# === Draw ROI Polygon ===
roi_np = np.array(ROI_POLYGON, np.int32)
cv2.polylines(frame, [roi_np], isClosed=True, color=(0, 255, 255), thickness=2)

# === Measure Pixel Distance Between Clicks ===
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if cv2.pointPolygonTest(roi_np, (x, y), False) < 0:
            print("❌ Point is outside ROI — please click inside the ROI.")
            return

        points.append((x, y))
        print(f"📍 Point {len(points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Measure Inside ROI", frame)

        if len(points) == 2:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            pixel_distance = int(math.hypot(dx, dy))
            print(f"\n📏 Measured Pixel Distance: {pixel_distance} pixels")
            print("👉 Use this value as PIXEL_DISTANCE_REF.")
            print("✅ Now you can relate it to your REAL_DISTANCE_METERS (e.g., 8.5 m)")

cv2.imshow("Measure Inside ROI", frame)
cv2.setMouseCallback("Measure Inside ROI", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()