import cv2
import csv
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# -------------------------------
# 1. PROJECT SETTINGS
# -------------------------------

VIDEO_PATH = "input_videos/campus_cctv.mp4"
MODEL_PATH = "yolo26n.pt"

OUTPUT_DIR = Path("output")
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"

CONFIDENCE_THRESHOLD = 0.40
DISPLAY_VIDEO = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 2. LOAD YOLO MODEL
# -------------------------------

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# -------------------------------
# 3. OPEN CCTV VIDEO
# -------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Cannot open video: {VIDEO_PATH}. "
        "Check the filename and folder."
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps >= 100:
    fps = 100

print(f"Video resolution: {width} x {height}")
print(f"Video FPS: {fps:2f}")

# -------------------------------
# 4. DEFINE RESTRICTED ZONE
# -------------------------------
# These coordinates are percentages of the video size.
# Change them to match your restricted entrance.

zone = [
    (int(width * 0.65), int(height * 0.30)),
    (int(width * 0.95), int(height * 0.30)),
    (int(width * 0.95), int(height * 0.95)),
    (int(width * 0.65), int(height * 0.95))
]

zone_contour = __import__("numpy").array(
    zone, dtype="int32"
)

# -------------------------------
# 5. PREPARE OUTPUT FILES
# -------------------------------

video_output = str(OUTPUT_DIR / "annotated_video.mp4")
csv_output = OUTPUT_DIR / "intrusion_events.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    video_output, fourcc, fps, (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Cannot create output video. Check the codec or output path."
    )

log_file = open(
    csv_output, "w", newline="", encoding="utf-8"
)

csv_writer = csv.writer(log_file)
csv_writer.writerow([
    "Timestamp",
    "Frame",
    "Event",
    "Confidence",
    "Bounding_Box",
    "Snapshot"
])

# -------------------------------
# 6. PROCESS VIDEO FRAMES
# -------------------------------

frame_number = 0
intrusion_was_present = False
total_events = 0

print("Processing video. Press Q to stop early.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Run YOLO detection on this frame.
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],       # COCO class 0 = person
            verbose=False
        )

        result = results[0]
        intruders = []

        # Draw restricted area.
        cv2.polylines(
            frame,
            [zone_contour],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2
        )

        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            (zone[0][0], max(25, zone[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # ---------------------------
        # 7. CHECK EACH PERSON
        # ---------------------------

        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Approximate ground position of person.
            foot_x = (x1 + x2) // 2
            foot_y = y2

            # Is the person's ground point inside the zone?
            inside_zone = cv2.pointPolygonTest(
                zone_contour,
                (float(foot_x), float(foot_y)),
                False
            ) >= 0

            if inside_zone:
                intruders.append({
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "foot": (foot_x, foot_y)
                })

            # Draw bounding box and confidence.
            box_color = (0, 0, 255) if inside_zone else (0, 255, 0)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                box_color, 2
            )

            label = f"Person {confidence:.2f}"

            if inside_zone:
                label = f"INTRUSION {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            cv2.circle(
                frame,
                (foot_x, foot_y),
                4,
                box_color,
                -1
            )

        # ---------------------------
        # 8. GENERATE INTRUSION EVENT
        # ---------------------------

        intrusion_present = len(intruders) > 0

        if intrusion_present:
            cv2.putText(
                frame,
                "ALERT: PERSON IN RESTRICTED AREA",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Log once when the zone changes from empty to occupied.
        if intrusion_present and not intrusion_was_present:
            total_events += 1

            timestamp = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

            snapshot_name = (
                f"intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_frame{frame_number}.jpg"
            )

            snapshot_path = SNAPSHOT_DIR / snapshot_name

            cv2.imwrite(str(snapshot_path), frame)

            # Log the highest-confidence intruder in this frame.
            best = max(
                intruders,
                key=lambda item: item["confidence"]
            )

            csv_writer.writerow([
                timestamp,
                frame_number,
                "PERSON_ENTERED_RESTRICTED_ZONE",
                round(best["confidence"], 3),
                str(best["bbox"]),
                str(snapshot_path)
            ])

            log_file.flush()

            print(
                f"INTRUSION EVENT #{total_events} | "
                f"Confidence: {best['confidence']:.2f} | "
                f"Snapshot: {snapshot_path}"
            )

        intrusion_was_present = intrusion_present

        # Display status and save the processed frame.
        cv2.putText(
            frame,
            f"Frame: {frame_number} | Events: {total_events}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if DISPLAY_VIDEO:
            cv2.imshow("CCTV Intrusion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

finally:
    cap.release()
    writer.release()
    log_file.close()
    cv2.destroyAllWindows()

print("\nProcessing finished.")
print("Annotated video:", video_output)
print("Intrusion log:", csv_output)
print("Snapshots folder:", SNAPSHOT_DIR)
print("Total intrusion episodes:", total_events)
