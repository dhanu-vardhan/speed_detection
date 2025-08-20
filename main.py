import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_PATH = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/PT_MODELS/yolov8n.pt"
VIDEO_PATH = "/Users/anthapuvivekanandareddy/Desktop/Speed_detection/input_data/speed_video_1.mp4"

# Entry & Exit Lines (two points each)
ENTRY_LINE = ((1607, 761), (929, 598))   # Yellow Line
EXIT_LINE = ((323, 685), (1160, 1036))   # Red Line

# Real-world distance (meters) between entry & exit
DISTANCE_METERS = 15.0

# Vehicle Classes in COCO: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def is_crossing_line(point, line, threshold=30):
    """Check if centroid is near a line segment"""
    p, p1, p2 = np.array(point), np.array(line[0]), np.array(line[1])
    line_vec, point_vec = p2 - p1, p - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return False
    line_unit = line_vec / line_len
    proj = np.dot(point_vec, line_unit)
    if proj < 0 or proj > line_len:
        return False
    nearest = p1 + proj * line_unit
    return np.linalg.norm(p - nearest) <= threshold


def draw_line(frame, line, color, text):
    cv2.line(frame, line[0], line[1], color, 3)
    cv2.putText(frame, text, (line[0][0] - 50, line[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ----------------------------
# MAIN
# ----------------------------
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    vehicle_tracks = {}   # {id: {entry_frame, exit_frame, class_id, centroid}}
    vehicle_speeds = {}
    next_id = 0
    frame_idx = 0

    print("Starting Vehicle Speed Detection...")
    print(f"FPS: {fps}, Distance: {DISTANCE_METERS} m")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run YOLO detection
        results = model(frame, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,class

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if cls not in VEHICLE_CLASSES or conf < 0.5:
                continue

            # Compute centroid
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Try to match with existing track
            matched_id = None
            for vid, data in vehicle_tracks.items():
                px, py = data["centroid"]
                if np.linalg.norm([cx - px, cy - py]) < 80:  # pixel threshold
                    matched_id = vid
                    break

            # New vehicle
            if matched_id is None:
                next_id += 1
                matched_id = next_id
                vehicle_tracks[matched_id] = {"centroid": (cx, cy),
                                              "entry_frame": None,
                                              "exit_frame": None,
                                              "class_id": cls}

            # Update centroid
            vehicle_tracks[matched_id]["centroid"] = (cx, cy)

            # ENTRY check
            if (vehicle_tracks[matched_id]["entry_frame"] is None and
                    is_crossing_line((cx, cy), ENTRY_LINE)):
                vehicle_tracks[matched_id]["entry_frame"] = frame_idx
                print(f"Vehicle {matched_id} entered at frame {frame_idx}")

            # EXIT check
            if (vehicle_tracks[matched_id]["entry_frame"] is not None and
                    vehicle_tracks[matched_id]["exit_frame"] is None and
                    is_crossing_line((cx, cy), EXIT_LINE)):
                vehicle_tracks[matched_id]["exit_frame"] = frame_idx
                frames_taken = (vehicle_tracks[matched_id]["exit_frame"] -
                                vehicle_tracks[matched_id]["entry_frame"])
                time_sec = frames_taken / fps
                if time_sec > 0:
                    speed_ms = DISTANCE_METERS / time_sec
                    speed_kmph = speed_ms * 3.6
                    vehicle_speeds[matched_id] = speed_kmph
                    cname = CLASS_NAMES[vehicle_tracks[matched_id]["class_id"]]
                    print(f"Vehicle {matched_id} ({cname}) speed: {speed_kmph:.2f} km/h")

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"ID:{matched_id} {CLASS_NAMES[cls]}"
            if matched_id in vehicle_speeds:
                label += f" {vehicle_speeds[matched_id]:.1f} km/h"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.putText(frame, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Draw ROI lines
        draw_line(frame, ENTRY_LINE, (0, 255, 255), "ENTRY")
        draw_line(frame, EXIT_LINE, (0, 0, 255), "EXIT")

        # Show frame
        cv2.imshow("Vehicle Speed Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("\n========== FINAL RESULTS ==========")
    for vid, spd in vehicle_speeds.items():
        cname = CLASS_NAMES[vehicle_tracks[vid]["class_id"]]
        print(f"Vehicle {vid} ({cname}): {spd:.2f} km/h")
    print("==================================")

if __name__ == "__main__":
    main()
