import os
import cv2
import csv
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/input_data/speed_video_1.mp4"

# Polygon ROI (adjust manually for your scene)
POLYGON_POINTS = np.array([
    (300, 700),
    (1200, 1050),
    (1600, 760),
    (900, 600)
], np.int32)

# Entry & Exit lines (inside polygon)
ENTRY_LINE = ((1600, 760), (900, 600))   # Yellow
EXIT_LINE  = ((300, 700), (1200, 1050))  # Red

# Real-world distance between lines (meters)
DISTANCE_METERS_BETWEEN_LINES = 10.0

# Skip frames for performance
FRAME_SKIP = 1

# Vehicle classes (COCO: car, motorcycle, bus, truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Speed limit
SPEED_LIMIT_KMPH = 10.0

# Output paths
OUTPUT_DIR = "./violations"
CSV_LOG_PATH = "./vehicles_log.csv"

# Tracking
MAX_TRACK_LOSS_FRAMES = 40
MATCH_DISTANCE_PX = 80
NEAR_LINE_THRESHOLD_PX = 30

# =========================
# UTILS
# =========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["UID", "timestamp", "vehicle_class", "speed_kmph", "snapshot_path"])

def log_vehicle(uid, vehicle_class, speed_kmph, snapshot_path=""):
    ts = datetime.now().isoformat(timespec="seconds")
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([uid, ts, vehicle_class, f"{speed_kmph:.2f}", snapshot_path])

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def distance_point_to_segment(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    if np.allclose(ab, 0):
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def is_on_line(centroid, line, threshold=NEAR_LINE_THRESHOLD_PX):
    return distance_point_to_segment(centroid, line[0], line[1]) <= threshold

def side_of_line(point, line):
    (x1, y1), (x2, y2) = line
    (x, y) = point
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def draw_polygon(frame, polygon, color=(255, 255, 0)):
    cv2.polylines(frame, [polygon], True, color, 3)

def draw_line(frame, line, color, text):
    cv2.line(frame, line[0], line[1], color, 3)
    cv2.putText(frame, text, (line[0][0] - 50, line[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def safe_crop(frame, bbox, pad=4):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    return frame[y1:y2, x1:x2]

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()
    init_csv(CSV_LOG_PATH)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracks, next_uid, frame_idx = {}, 0, 0
    entry_dir = lambda c: np.sign(side_of_line(c, ENTRY_LINE))
    exit_dir  = lambda c: np.sign(side_of_line(c, EXIT_LINE))

    print("🚦 Vehicle Speed Detection + Logging + Violations")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % FRAME_SKIP != 0:
            continue

        results = model(frame, verbose=False)[0]
        dets = results.boxes.data.cpu().numpy() if results.boxes is not None else []

        # Filter vehicles inside ROI
        current = []
        for d in dets:
            x1, y1, x2, y2, conf, cls = d
            cls = int(cls)
            if conf < 0.5 or cls not in VEHICLE_CLASSES:
                continue
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            if not point_in_polygon((cx, cy), POLYGON_POINTS):
                continue
            current.append({"bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "centroid": (cx, cy), "class_id": cls})

        # Match detections to tracks
        unmatched_tracks = set(tracks.keys())
        for det in current:
            cx, cy = det["centroid"]
            best_id, best_dist = None, float("inf")
            for vid in list(unmatched_tracks):
                px, py = tracks[vid]["centroid"]
                dist = np.hypot(cx - px, cy - py)
                if dist < MATCH_DISTANCE_PX and dist < best_dist:
                    best_dist, best_id = dist, vid
            if best_id is None:
                next_uid += 1
                vid = next_uid
                tracks[vid] = {"uid": vid, "centroid": (cx, cy), "bbox": det["bbox"],
                               "class_id": det["class_id"], "last_seen": frame_idx,
                               "entry_frame": None, "exit_frame": None,
                               "entry_side": entry_dir((cx, cy)),
                               "exit_side": exit_dir((cx, cy)),
                               "has_logged": False, "has_violation": False}
            else:
                tracks[best_id].update({"centroid": (cx, cy), "bbox": det["bbox"],
                                        "class_id": det["class_id"], "last_seen": frame_idx})
                unmatched_tracks.discard(best_id)

        # Drop stale tracks
        for vid in list(tracks.keys()):
            if frame_idx - tracks[vid]["last_seen"] > MAX_TRACK_LOSS_FRAMES:
                tracks.pop(vid)

        # Crossing + speed calculation
        for vid, t in tracks.items():
            cx, cy, c = *t["centroid"], t["centroid"]

            if t["entry_frame"] is None:
                cur_side = entry_dir(c)
                if cur_side != 0 and cur_side != t["entry_side"] and is_on_line(c, ENTRY_LINE):
                    t["entry_frame"] = frame_idx
                    t["entry_side"] = cur_side

            if t["entry_frame"] and not t["exit_frame"]:
                cur_side = exit_dir(c)
                if cur_side != 0 and cur_side != t["exit_side"] and is_on_line(c, EXIT_LINE):
                    t["exit_frame"] = frame_idx
                    t["exit_side"] = cur_side

                    frames_taken = (t["exit_frame"] - t["entry_frame"])
                    time_sec = (frames_taken * FRAME_SKIP) / fps
                    if time_sec > 0:
                        speed_kmph = (DISTANCE_METERS_BETWEEN_LINES / time_sec) * 3.6
                        t["speed_kmph"] = speed_kmph
                        vclass = CLASS_NAMES.get(t["class_id"], str(t["class_id"]))

                        if not t["has_logged"]:
                            # Log every vehicle
                            log_vehicle(t["uid"], vclass, speed_kmph, "")
                            t["has_logged"] = True

                        if speed_kmph > SPEED_LIMIT_KMPH and not t["has_violation"]:
                            crop = safe_crop(frame, t["bbox"])
                            snap_name = f"violation_{t['uid']}_{int(time.time())}.jpg"
                            snap_path = os.path.join(OUTPUT_DIR, snap_name)
                            cv2.imwrite(snap_path, crop)
                            log_vehicle(t["uid"], vclass, speed_kmph, snap_path)
                            t["has_violation"] = True

        # ==== DRAWING ====
        draw_polygon(frame, POLYGON_POINTS)
        draw_line(frame, ENTRY_LINE, (0, 255, 255), "ENTRY")
        draw_line(frame, EXIT_LINE, (0, 0, 255), "EXIT")

        for vid, t in tracks.items():
            if frame_idx - t["last_seen"] > 5: continue
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = t["centroid"]
            vclass = CLASS_NAMES.get(t["class_id"], str(t["class_id"]))
            label = f"ID:{vid} {vclass}"
            color = (0, 255, 0)
            if "speed_kmph" in t:
                label += f" {t['speed_kmph']:.1f} km/h"
                color = (0, 0, 255) if t["speed_kmph"] > SPEED_LIMIT_KMPH else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

        cv2.putText(frame, f"Frame:{frame_idx} FPS:{fps:.1f} Limit:{SPEED_LIMIT_KMPH:.0f} km/h",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2)

        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done. Log: {os.path.abspath(CSV_LOG_PATH)} | Snapshots: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
