# requirements:
# pip install ultralytics opencv-python supervision numpy

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = "../data/person_count.mp4"
OUTPUT_PATH = "../data/runs/person_count.mp4"
MODEL_PATH = "yolov8n.pt"

CONFIDENCE = 0.3
IOU = 0.5
CLASSES = [0]  # person only

FRAME_SIZE = (1280, 720)

# Define ROI polygon (EDIT THIS BASED ON YOUR VIDEO)
ROI_POLYGON = np.array([
    [0, 0],
    [0, 1280],
    [1280, 720],
    [300, 700]
])
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
    polygon_zone = sv.PolygonZone(
        polygon=ROI_POLYGON,
        frame_resolution_wh=FRAME_SIZE
    )

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

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        FRAME_SIZE
    )

    # ── Counting state ─────────────────────────────────────────────────────────
    counted_ids = set()

    frame_idx = 0

    # ── Main Loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        mask = polygon_zone.trigger(points=points)
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

        # Draw count
        cv2.putText(
            annotated,
            f"Total Count: {person_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ── 7. Write + Display ────────────────────────────────────────────────
        writer.write(annotated)

        cv2.imshow("Person Counting (ROI)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames | Count: {person_count}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Done. Output saved to: {OUTPUT_PATH}")
    print(f"Final Count: {person_count}")


if __name__ == "__main__":
    run()