"""
Sport Activity Detection — IMX500 RPK Inference Script
=======================================================
Fixed version — resolves empty detections issue by:
  1. Correct bbox_normalization (True for our exported model)
  2. Correct bbox_order (xy for our YOLO export)
  3. Lower threshold (0.25 vs default 0.55)
  4. Debug mode to inspect raw outputs

Usage:
    python3 sport_detect.py \
        --model /home/tagore/sport_project/rpk_output/network.rpk \
        --labels /home/tagore/sport_project/labels.txt \
        --fps 25

Author: Chandu | THD AISD
"""

import argparse
import sys
from functools import lru_cache

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# ─── Globals ─────────────────────────────────────────────────────────────────
last_detections = []
last_results     = None
frame_count      = 0

# ─── Sport colors (BGR) ──────────────────────────────────────────────────────
SPORT_COLORS = {
    "basketball_shoot": (0,  165, 255),   # Orange
    "cricket_batting":  (0,  255,   0),   # Green
    "tennis_swing":     (255,  0, 255),   # Purple
    "volleyball_spike": (0,  255, 255),   # Cyan
}


# ─── Detection class ─────────────────────────────────────────────────────────
class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf     = conf
        self.box      = imx500.convert_inference_coords(coords, metadata, picam2)


# ─── Parse detections ────────────────────────────────────────────────────────
def parse_detections(metadata: dict):
    global last_detections, frame_count
    frame_count += 1

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        if frame_count % 30 == 0:
            print("[DEBUG] np_outputs is None — camera not returning tensors yet")
        return last_detections

    input_w, input_h = imx500.get_input_size()

    # ── Debug: print raw output shapes every 60 frames ──────────────────────
    if frame_count % 60 == 0:
        print(f"\n[DEBUG] frame={frame_count}")
        print(f"  input_w={input_w}  input_h={input_h}")
        print(f"  np_outputs length: {len(np_outputs)}")
        for i, o in enumerate(np_outputs):
            print(f"  output[{i}] shape={o.shape}  dtype={o.dtype}")

    # ── Handle nanodet postprocess ───────────────────────────────────────────
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=args.threshold,
            iou_thres=args.iou,
            max_out_dets=args.max_detections,
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)

    else:
        # Standard YOLO output: [boxes, scores, classes]
        boxes   = np_outputs[0][0]
        scores  = np_outputs[1][0]
        classes = np_outputs[2][0]

        # ── Debug: show top-5 raw scores ────────────────────────────────────
        if frame_count % 60 == 0:
            top5 = sorted(zip(scores, classes), reverse=True)[:5]
            print(f"  top-5 scores: {[(f'{s:.3f}', int(c)) for s,c in top5]}")
            print(f"  bbox_normalization : {intrinsics.bbox_normalization}")
            print(f"  bbox_order         : {intrinsics.bbox_order}")

        # ── Apply normalization ──────────────────────────────────────────────
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h

        # ── Reorder coordinates if needed ───────────────────────────────────
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    # ── Build detection list ─────────────────────────────────────────────────
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > args.threshold
    ]

    if frame_count % 60 == 0:
        print(f"  detections above threshold ({args.threshold}): {len(last_detections)}")

    return last_detections


# ─── Labels ──────────────────────────────────────────────────────────────────
@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [l for l in labels if l and l != "-"]
    return labels


# ─── Draw detections ─────────────────────────────────────────────────────────
def draw_detections(request, stream="main"):
    detections = last_results
    if not detections:
        return

    labels = get_labels()

    with MappedArray(request, stream) as m:

        # Title banner
        cv2.rectangle(m.array, (0, 0), (260, 45), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            m.array, "SPORT DETECTOR",
            (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2,
        )

        for det in detections:
            x, y, w, h = det.box
            idx   = int(det.category)
            sport = labels[idx] if idx < len(labels) else f"cls{idx}"
            label = f"{sport}: {det.conf*100:.0f}%"
            color = SPORT_COLORS.get(sport, (255, 255, 255))

            # Bounding box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), color, thickness=2)

            # Label background
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tx, ty = x + 4, y + th + 4
            overlay = m.array.copy()
            cv2.rectangle(overlay, (tx - 2, ty - th - 2), (tx + tw + 2, ty + bl),
                          (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, m.array, 0.5, 0, m.array)

            # Label text
            cv2.putText(m.array, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ROI indicator
        if intrinsics.preserve_aspect_ratio:
            bx, by, bw, bh = imx500.get_roi_scaled(request)
            cv2.rectangle(m.array, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
            cv2.putText(m.array, "ROI", (bx + 5, by + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


# ─── Args ─────────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Sport Activity Detector — IMX500")
    parser.add_argument("--model",
        default="/home/tagore/sport_project/rpk_output/network.rpk",
        help="Path to network.rpk")
    parser.add_argument("--labels",
        default="/home/tagore/sport_project/labels.txt",
        help="Path to labels.txt")
    parser.add_argument("--fps",            type=int,   default=25)
    parser.add_argument("--threshold",      type=float, default=0.25,
        help="Detection confidence threshold (default 0.25)")
    parser.add_argument("--iou",            type=float, default=0.65)
    parser.add_argument("--max-detections", type=int,   default=10)
    parser.add_argument("--bbox-normalization",
        action=argparse.BooleanOptionalAction, default=True,
        help="Normalize bboxes by input height (default: True for YOLO exports)")
    parser.add_argument("--bbox-order",
        choices=["yx", "xy"], default="xy",
        help="Coordinate order: xy=(x0,y0,x1,y1) yx=(y0,x0,y1,x1) (default: xy)")
    parser.add_argument("--ignore-dash-labels",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess",
        choices=["", "nanodet"], default=None)
    parser.add_argument("--preserve-aspect-ratio", "-r",
        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-intrinsics",
        action="store_true", help="Print model intrinsics and exit")
    parser.add_argument("--debug",
        action="store_true", help="Print debug output every 60 frames")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()

    print(f"Loading model: {args.model}")
    print(f"Labels:        {args.labels}")
    print(f"Threshold:     {args.threshold}")
    print(f"BBox order:    {args.bbox_order}")
    print(f"BBox norm:     {args.bbox_normalization}")

    # Must be called before Picamera2
    imx500 = IMX500(args.model)

    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    # Apply args → intrinsics
    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value) as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Fallback labels
    if intrinsics.labels is None:
        intrinsics.labels = [
            "basketball_shoot",
            "cricket_batting",
            "tennis_swing",
            "volleyball_spike",
        ]

    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    print("\n✅ Intrinsics loaded:")
    print(f"  labels            : {intrinsics.labels}")
    print(f"  bbox_normalization: {intrinsics.bbox_normalization}")
    print(f"  bbox_order        : {intrinsics.bbox_order}")
    print(f"  inference_rate    : {intrinsics.inference_rate}")
    print(f"  postprocess       : {intrinsics.postprocess}")
    print("\nStarting camera... press Ctrl+C to stop\n")

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": args.fps},
        buffer_count=12,
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if args.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    picam2.pre_callback = draw_detections

    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata())
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        picam2.stop()
