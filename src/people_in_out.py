# requirements:
# pip install ultralytics opencv-python supervision numpy

import argparse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Default config
# ----------------------------------------------------------------------------
DEFAULT_VIDEO_PATH = "data/people_in_out.mp4"
DEFAULT_OUTPUT_PATH = "data/runs/people_in_out.avi"
DEFAULT_MODEL_PATH = "yolov8s.pt"

DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.5
DEFAULT_CLASSES = [0]  # person only

DEFAULT_FRAME_SIZE = (1280, 720)
DEFAULT_READ_FPS = 10

# line is middle horizontal by default; set to H/2
DEFAULT_LINE_COLOR = (0, 255, 255)
DEFAULT_LINE_THICKNESS = 2
DEFAULT_IN_COLOR = (0, 255, 0)
DEFAULT_OUT_COLOR = (0, 0, 255)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="People In/Out counting with YOLO+ByteTrack")

    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output video path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")

    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IOU threshold")
    parser.add_argument("--fps", type=int, default=DEFAULT_READ_FPS, help="Read FPS")
    parser.add_argument("--w", type=int, default=DEFAULT_FRAME_SIZE[0], help="Output width")
    parser.add_argument("--h", type=int, default=DEFAULT_FRAME_SIZE[1], help="Output height")

    parser.add_argument("--line_y", type=int, default=None, help="Y coordinate for virtual line")

    return parser.parse_args()


def build_annotators():
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=4,
    )
    return box_annotator, label_annotator


def run(
    video_path: str = DEFAULT_VIDEO_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    classes=None,
    read_fps: int = DEFAULT_READ_FPS,
    frame_size=DEFAULT_FRAME_SIZE,
    line_y=None,
):

    if classes is None:
        classes = DEFAULT_CLASSES

    frame_width, frame_height = frame_size
    line_y = int(line_y if line_y is not None else frame_height / 2)

    model = YOLO(model_path)

    tracker = sv.ByteTrack(
        track_activation_threshold=confidence,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    box_annotator, label_annotator = build_annotators()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(fps / read_fps))
    print(f"Video FPS: {fps:.2f}, reading {read_fps} fps (skipping every {frame_skip} frames)")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        read_fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    track_states = {}  # tracker_id -> dict(last_y, last_side, lock)
    count_in = 0
    count_out = 0

    frame_idx = 0
    frames_processed = 0

    # Crossing padding to avoid jitter near line
    crossing_padding = max(5, int(frame_height * 0.025))

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if (frame_idx - 1) % frame_skip != 0:
            continue

        frames_processed += 1

        frame = cv2.resize(frame, (frame_width, frame_height))

        results = model.predict(
            frame,
            conf=confidence,
            iou=iou,
            classes=classes,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Track crossing on middle line using centroid and anti-jitter lock
        for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            cy = int((y1 + y2) / 2)
            current_side = "above" if cy < line_y else "below"

            prev_state = track_states.get(tracker_id)
            if prev_state is None:
                track_states[tracker_id] = {
                    "last_y": cy,
                    "last_side": current_side,
                    "lock_until": None,
                }
                continue

            last_side = prev_state["last_side"]
            lock_until = prev_state.get("lock_until")

            # if currently in a lock period, don't re-count unless far from line
            if lock_until is not None and abs(cy - line_y) <= crossing_padding:
                prev_state["last_y"] = cy
                continue

            if last_side != current_side:
                if last_side == "above" and current_side == "below":
                    count_out += 1
                    prev_state["lock_until"] = line_y + crossing_padding
                elif last_side == "below" and current_side == "above":
                    count_in += 1
                    prev_state["lock_until"] = line_y - crossing_padding

            prev_state.update({"last_y": cy, "last_side": current_side})
            track_states[tracker_id] = prev_state

        # Draw line
        annotated_frame = frame.copy()
        cv2.line(
            annotated_frame,
            (0, line_y),
            (frame_width, line_y),
            DEFAULT_LINE_COLOR,
            DEFAULT_LINE_THICKNESS,
        )

        # Draw boxes and labels
        labels = [f"id:{tid:.0f}" for tid in detections.tracker_id]

        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Draw counts and status with white background and styled foreground
        text = f"IN: {count_in}  OUT: {count_out}  TRACKS: {len(track_states)}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.9
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        padding = 12

        # White background box
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (10 + text_width + 2 * padding, 10 + text_height + 2 * padding),
            (255, 255, 255),
            -1,
        )

        # Draw black outline for text for contrast
        cv2.putText(
            annotated_frame,
            text,
            (10 + padding, 10 + text_height + padding),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness + 2,
            cv2.LINE_AA,
        )

        # Draw white text on top
        cv2.putText(
            annotated_frame,
            text,
            (10 + padding, 10 + text_height + padding),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        writer.write(annotated_frame)

        if frames_processed % 10 == 0:
            print(f"Processed {frames_processed} frames - IN: {count_in}, OUT: {count_out}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Output saved: {output_path}")
    print(f"Final IN: {count_in}, OUT: {count_out}, TOTAL TRACKS: {len(track_states)}")


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        classes=DEFAULT_CLASSES,
        read_fps=args.fps,
        frame_size=(args.w, args.h),
        line_y=args.line_y,
    )
