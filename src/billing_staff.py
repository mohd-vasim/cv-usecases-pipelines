# requirements:
# pip install ultralytics opencv-python supervision numpy

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class CounterROI:
    id: int
    customer_roi: List[Tuple[int, int]]
    staff_roi: List[Tuple[int, int]]
    customer_present: bool = False
    staff_present: bool = False

    def reset(self):
        self.customer_present = False
        self.staff_present = False

    def get_status(self) -> str:
        if self.customer_present:
            if self.staff_present:
                return "SERVING"
            return "WAITING STAFF"
        return "IDLE"

    def draw(self, frame, roi_color_customer, roi_color_staff):
        # customer area
        customer_poly = np.array(self.customer_roi, dtype=np.int32)
        cv2.polylines(frame, [customer_poly], True, roi_color_customer, 2)
        cv2.putText(
            frame,
            f"C{self.id}",
            (self.customer_roi[0][0] + 5, self.customer_roi[0][1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            roi_color_customer,
            2,
            cv2.LINE_AA,
        )

        # staff area
        staff_poly = np.array(self.staff_roi, dtype=np.int32)
        cv2.polylines(frame, [staff_poly], True, roi_color_staff, 2)
        cv2.putText(
            frame,
            f"S{self.id}",
            (self.staff_roi[0][0] + 5, self.staff_roi[0][1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            roi_color_staff,
            2,
            cv2.LINE_AA,
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Billing staff detection with paired ROIs")

    parser.add_argument("--config", default="config.json", help="Config JSON file path")

    # Load config to set defaults
    config_path = parser.parse_known_args()[0].config
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print(f"Config file {config_path} not found, using command line defaults")
        config = {}

    parser.add_argument("--video", default=config.get("video_path", "data/people_in_out.mp4"), help="Input video path")
    parser.add_argument("--output", default=config.get("output_path", "data/runs/billing_staff.avi"), help="Output video path")
    parser.add_argument("--model", default=config.get("model_path", "yolov8s.pt"), help="YOLO model path")

    parser.add_argument("--conf", type=float, default=config.get("confidence", 0.3), help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=config.get("iou", 0.5), help="NMS IOU threshold")
    parser.add_argument("--fps", type=int, default=config.get("read_fps", 5), help="Read FPS")
    parser.add_argument("--w", type=int, default=config.get("frame_size", [1280, 720])[0], help="Output width")
    parser.add_argument("--h", type=int, default=config.get("frame_size", [1280, 720])[1], help="Output height")

    parser.add_argument("--rois", default=None, help="JSON file containing counter ROI pairs (overrides config)")

    return parser.parse_args()


def load_counter_rois(rois_path: Optional[str], config: dict) -> List[CounterROI]:
    if rois_path is not None:
        if not os.path.exists(rois_path):
            raise FileNotFoundError(f"ROI file does not exist: {rois_path}")
        with open(rois_path, "r") as f:
            roi_config = json.load(f)
        raw = roi_config.get("counters", [])
        if not raw:
            raise ValueError("ROI config file must contain a 'counters' list")
    else:
        raw = config.get("counters", [])
        if not raw:
            raise ValueError("Config file must contain a 'counters' list")

    counters = []
    for entry in raw:
        counters.append(
            CounterROI(
                id=int(entry["id"]),
                customer_roi=[tuple(p) for p in entry["customer_roi"]],
                staff_roi=[tuple(p) for p in entry["staff_roi"]],
            )
        )
    return counters


def _point_in_polygon(point_x: float, point_y: float, polygon: List[Tuple[int, int]]) -> bool:
    # Convert to numpy array for cv2
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, (point_x, point_y), False) >= 0


def classify_person_in_counter(counter: CounterROI, bbox: Tuple[float, float, float, float], track_id: int):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if _point_in_polygon(cx, cy, counter.customer_roi):
        counter.customer_present = True
    if _point_in_polygon(cx, cy, counter.staff_roi):
        counter.staff_present = True


def make_status_text(counters: List[CounterROI]) -> str:
    parts = []
    for counter in counters:
        if counter.customer_present:
            if counter.staff_present:
                status = "Customer present, Staff present"
            else:
                status = "Customer present, Staff absent"
        else:
            status = "No customer"
        parts.append(f"Counter {counter.id}: {status}")
    return " | ".join(parts)


def build_annotators():
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)
    return box_annotator, label_annotator


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(
    config: dict,
    video_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    rois_path: Optional[str] = None,
    confidence: Optional[float] = None,
    iou: Optional[float] = None,
    classes=None,
    read_fps: Optional[int] = None,
    frame_size=None,
):
    # Use config defaults, override with args
    video_path = video_path or config.get("video_path", "data/people_in_out.mp4")
    output_path = output_path or config.get("output_path", "data/runs/billing_staff.avi")
    model_path = model_path or config.get("model_path", "yolov8s.pt")
    confidence = confidence if confidence is not None else config.get("confidence", 0.3)
    iou = iou if iou is not None else config.get("iou", 0.5)
    read_fps = read_fps if read_fps is not None else config.get("read_fps", 5)
    frame_size = frame_size or tuple(config.get("frame_size", [1280, 720]))
    if classes is None:
        classes = config.get("classes", [0])

    frame_width, frame_height = frame_size

    # Colors from config
    ROI_COLOR_CUSTOMER = tuple(config.get("roi_color_customer", [0, 165, 255]))
    ROI_COLOR_STAFF = tuple(config.get("roi_color_staff", [255, 0, 0]))
    ROI_COLOR_TEXT = tuple(config.get("roi_color_text", [255, 255, 255]))

    # Load model and tracker
    model = YOLO(model_path)
    tracker = sv.ByteTrack(
        track_activation_threshold=confidence,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    box_annotator, label_annotator = build_annotators()

    # Load or initialize ROI pairs
    counters = load_counter_rois(rois_path, config)

    # Video I/O
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

    frame_idx = 0
    processed_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if (frame_idx - 1) % frame_skip != 0:
            continue

        processed_frames += 1
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

        # reset counter occupancy per frame
        for counter in counters:
            counter.reset()

        # assign each tracked person to counter ROIs
        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            for counter in counters:
                classify_person_in_counter(counter, xyxy, track_id)

        annotated_frame = frame.copy()

        # Draw ROIs
        for counter in counters:
            counter.draw(annotated_frame, ROI_COLOR_CUSTOMER, ROI_COLOR_STAFF)

        # Draw detection boxes & IDs
        labels = [f"id:{int(tid)}" for tid in detections.tracker_id]
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Draw counter statuses
        status_text = make_status_text(counters)
        cv2.rectangle(annotated_frame, (10, 10), (frame_width - 10, 70), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            status_text,
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ROI_COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated_frame)

        if processed_frames % 10 == 0:
            print(f"Processed {processed_frames} frames | {status_text}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Output saved: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    # Config is already loaded in parse_args, but we need to load it again for run
    config = load_config(args.config)
    run(
        config=config,
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        rois_path=args.rois,
        confidence=args.conf,
        iou=args.iou,
        read_fps=args.fps,
        frame_size=(args.w, args.h),
    )
