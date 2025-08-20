import cv2

# --- Load the first frame from video ---
video_path = r"/Users/anthapuvivekanandareddy/Desktop/Speed_detection/input_data/speed_video_1.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to load video or first frame.")
    exit()

# Clone the original frame
clone = frame.copy()
points = []

def draw_polygon(img, pts):
    # Draw all the points
    for pt in pts:
        cv2.circle(img, pt, 6, (0, 0, 255), -1)
    
    # Draw connecting lines
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], (0, 255, 0), 2)
    
    if len(pts) == 4:
        cv2.line(img, pts[3], pts[0], (0, 255, 0), 2)  # Close the polygon

def click_event(event, x, y, flags, param):
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        draw_polygon(frame, points)

        print(f"Clicked point {len(points)}: ({x}, {y})")

        if len(points) == 4:
            # Save the image with ROI
            cv2.imwrite("roi_drawn_from_video.jpg", frame)
            print("\n✅ ROI complete and saved as 'roi_drawn_from_video.jpg'")
            print("\n🧾 ROI Points:")
            for i, pt in enumerate(points, 1):
                print(f"Point {i}: {pt}")

cv2.namedWindow("Draw ROI - Click 4 Points")
cv2.setMouseCallback("Draw ROI - Click 4 Points", click_event)

while True:
    cv2.imshow("Draw ROI - Click 4 Points", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        frame = clone.copy()
        points = []
        print("🔁 Reset - Click 4 new points")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()