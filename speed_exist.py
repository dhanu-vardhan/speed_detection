import cv2
import os
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO



# === Parameters ===
VIDEO_PATH = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/PT_MODELS/speed_video_1.mp4"
MODEL_PATH = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/PT_MODELS/yolov8n.pt"
CSV_OUTPUT = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/output"
SNAPSHOT_DIR = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/snapshots"

FPS = 59
FRAME_SKIP = 2
REAL_DISTANCE_METERS = 25
PIXEL_DISTANCE_REF = 414
SPEED_LIMIT_KMPH = 40
SPEED_MIN = 0
SPEED_MAX = 300

ROI_POLYGON = [(902, 599), (476, 669), (1186, 997), (1550, 765)]

CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
CLASS_IDS = list(CLASS_NAMES.keys())

CSV_HEADER = ['object_id', 'class_name', 'speed_kmph', 'direction', 'timestamp', 'date', 'snapshot_path']

# === Initialization ===
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
frame_num = 0

object_tracks = {}
object_speeds = {}
csv_rows = []

# === Main Processing Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if frame_num % FRAME_SKIP != 0:
        continue

    results = model.track(frame, persist=True, classes=CLASS_IDS, verbose=False)
    if not results or not results[0].boxes:
        continue

    boxes = results[0].boxes
    ids = boxes.id
    if ids is None:
        continue

    ids = ids.int().tolist()
    class_ids = boxes.cls.int().tolist()
    coords = boxes.xyxy.cpu().numpy()

    for i in range(len(coords)):
        obj_id = ids[i]
        class_id = class_ids[i]
        class_name = CLASS_NAMES.get(class_id)
        if class_name is None:
            continue

        x1, y1, x2, y2 = map(int, coords[i])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cv2.pointPolygonTest(np.array(ROI_POLYGON, np.int32), (cx, cy), False) < 0:
            continue

        if obj_id not in object_tracks:
            object_tracks[obj_id] = [(frame_num, cy)]
        else:
            object_tracks[obj_id].append((frame_num, cy))

        history = object_tracks[obj_id]
        if len(history) >= 5 and obj_id not in object_speeds:
            f1, y1_ = history[-5]
            f2, y2_ = history[-1]

            pixel_movement = abs(y2_ - y1_)
            if pixel_movement > 0:
                meters_moved = (pixel_movement / PIXEL_DISTANCE_REF) * REAL_DISTANCE_METERS
                time_elapsed = (f2 - f1) * (1 / FPS)
                if time_elapsed > 0:
                    speed_mps = meters_moved / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    if SPEED_MIN < speed_kmph < SPEED_MAX:
                        object_speeds[obj_id] = speed_kmph
                        direction = "down" if y2_ > y1_ else "up"
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        date = datetime.now().strftime('%Y-%m-%d')
                        snapshot_path = ""

                        if speed_kmph > SPEED_LIMIT_KMPH:
                            snap_name = f"{obj_id}_{int(speed_kmph)}kmph_{timestamp.replace(':','')}.jpg"
                            snapshot_path = os.path.join(SNAPSHOT_DIR, snap_name)
                            cv2.imwrite(snapshot_path, frame[y1:y2, x1:x2])
                            print(f"🚨 OVERSPEED: {class_name} ID:{obj_id} → {speed_kmph:.1f} km/h")

                        csv_rows.append([
                            obj_id, class_name, round(speed_kmph, 1),
                            direction, timestamp, date, snapshot_path
                        ])

        # === Draw Bounding Box & Speed ===
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ID:{obj_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if obj_id in object_speeds:
            cv2.putText(frame, f"{object_speeds[obj_id]:.1f} km/h", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.polylines(frame, [np.array(ROI_POLYGON, np.int32)], True, (0, 255, 255), 2)
    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Save CSV Log ===
with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    writer.writerows(csv_rows)

print(f"\n✅ Speed log saved to: {CSV_OUTPUT}")