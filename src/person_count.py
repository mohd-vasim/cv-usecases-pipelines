# requirements:
# pip install ultralytics opencv-python supervision numpy

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = "data/person_count.mp4"
OUTPUT_PATH = "data/runs/person_count.avi"  # Changed to AVI format for better compatibility
MODEL_PATH = "yolov8s.pt"

CONFIDENCE = 0.3
IOU = 0.5
CLASSES = [0]  # person only

FRAME_SIZE = (1280, 720)
READ_FPS = 2  # Process 1 or 2 frames per second (set 1 for 1 FPS, 2 for 2 FPS)

# Define ROI polygon (EDIT THIS BASED ON YOUR VIDEO)
ROI_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1280, 720],
    [0, 720]
], dtype=np.int64)
# ─────────────────────────────────────────────────────────────────────────────


def build_annotators():
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=4
    )
    return box_annotator, label_annotator


def run():
    # ── Load model ────────────────────────────────────────────────────────────
    model = YOLO(MODEL_PATH)

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = sv.ByteTrack(
        track_activation_threshold=CONFIDENCE,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    # ── ROI Zone ──────────────────────────────────────────────────────────────
    polygon_zone = sv.PolygonZone(polygon=ROI_POLYGON)

    polygon_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color.GREEN,
        thickness=2
    )

    box_annotator, label_annotator = build_annotators()

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise Exception("Error opening video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Calculate frame skip rate
    frame_skip = int(fps / READ_FPS)
    print(f"Video FPS: {fps}, Reading {READ_FPS} frames/sec (skipping every {frame_skip} frames)")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"MJPG"),  # Motion JPEG codec - highly compatible
        READ_FPS,  # Output FPS matches the reading rate
        FRAME_SIZE
    )
    
    if not writer.isOpened():
        raise Exception(f"Failed to open video writer for {OUTPUT_PATH}")

    # ── Counting state ─────────────────────────────────────────────────────────
    counted_ids = set()

    frame_idx = 0
    read_frame_count = 0

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Skip frames based on READ_FPS
        if (frame_idx - 1) % frame_skip != 0:
            continue
        
        read_frame_count += 1
        frame = cv2.resize(frame, FRAME_SIZE)

        # ── 1. Detection ──────────────────────────────────────────────────────
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            iou=IOU,
            classes=CLASSES,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # ── 2. Tracking ───────────────────────────────────────────────────────
        detections = tracker.update_with_detections(detections)

        # ── 3. ROI Filtering (using bottom-center point) ──────────────────────
        mask = polygon_zone.trigger(detections=detections)
        detections_in_roi = detections[mask]

        # ── 4. Unique Counting ────────────────────────────────────────────────
        for tracker_id in detections_in_roi.tracker_id:
            if tracker_id not in counted_ids:
                counted_ids.add(tracker_id)

        person_count = len(counted_ids)

        # ── 5. Labels ─────────────────────────────────────────────────────────
        labels = [
            f"id:{tid} person {conf:.2f}"
            for tid, conf in zip(
                detections_in_roi.tracker_id,
                detections_in_roi.confidence
            )
        ]

        # ── 6. Annotation ─────────────────────────────────────────────────────
        annotated = frame.copy()

        # Draw ROI
        annotated = polygon_annotator.annotate(scene=annotated)

        # Draw boxes
        annotated = box_annotator.annotate(
            scene=annotated,
            detections=detections_in_roi
        )

        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections_in_roi,
            labels=labels
        )

        # Draw count with professional styling
        count_text = f"Total Count: {person_count}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.5
        font_thickness = 2
        text_color = (0, 0, 0)  # Black
        bg_color = (255, 255, 255)  # White
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            count_text, font, font_scale, font_thickness
        )
        
        # Define position and padding
        x, y = 50, 60
        padding = 10
        
        # Draw white background rectangle
        cv2.rectangle(
            annotated,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1  # Filled rectangle
        )
        
        # Draw black border for better definition
        cv2.rectangle(
            annotated,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            (0, 0, 0),  # Black border
            2
        )
        
        # Draw text
        cv2.putText(
            annotated,
            count_text,
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness
        )

        # ── 7. Write + Display ────────────────────────────────────────────────
        writer.write(annotated)

        # cv2.imshow("Person Counting (ROI)", annotated)
        try:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            pass  # Headless environment, skip display

        if read_frame_count % 10 == 0:
            print(f"Processed {read_frame_count} read frames (frame #{frame_idx} from video) | Count: {person_count}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # Headless environment

    print(f"\n✅ Done. Output saved to: {OUTPUT_PATH}")
    print(f"Final Count: {person_count}")


if __name__ == "__main__":
    run()