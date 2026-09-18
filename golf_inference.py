#!/usr/bin/env python3
"""
Mini Caddie — Golf Inference (Pi-side NMS)
Uses hailort Python bindings directly — no GStreamer pipeline.
Runs our custom YOLOv8n HEF and does DFL decode + NMS on the Pi.

Usage:
  cd ~/hailo-apps && source setup_env.sh
  python ~/mini-caddie/golf_inference.py
"""
import os
import sys
import json
import time
import numpy as np
import cv2

# Hailo imports
from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    ConfigureParams,
    FormatType,
    HailoStream,
    InputVStreamParams,
    OutputVStreamParams,
)

LABELS = [
    "background", "golf_ball", "golf_club", "golf_club_head",
    "golf_hole", "golf_mat", "person", "player_not_ready", "player_ready"
]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_CLASSES = 8
INPUT_SIZE = 640

# Anchor config for YOLOv8 (strides 8, 16, 32)
ANCHOR_STRIDES = [8, 16, 32]
ANCHOR_GRIDS = [80, 40, 20]  # 640/stride
TOTAL_ANCHORS = sum(g * g for g in ANCHOR_GRIDS)  # 8400


def build_anchors():
    """Build anchor centers and strides for all 8400 YOLOv8 anchors."""
    centers = []
    strides = []
    for stride, grid in zip(ANCHOR_STRIDES, ANCHOR_GRIDS):
        for y in range(grid):
            for x in range(grid):
                centers.append([x * stride + stride // 2, y * stride + stride // 2])
                strides.append(stride)
    return np.array(centers, dtype=np.float32), np.array(strides, dtype=np.float32)


ANCHOR_CENTERS, ANCHOR_STRIDES_ARR = build_anchors()


def dfl_decode(box_pred):
    """Decode DFL: (N, 64) -> (N, 4) [l, t, r, b] in grid units."""
    N = box_pred.shape[0]
    box_pred = box_pred.reshape(N, 4, 16)
    exp = np.exp(box_pred - box_pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    bins = np.arange(16, dtype=np.float32)
    return (prob * bins).sum(axis=-1)


def nms(boxes, scores, iou_threshold):
    """Simple NMS returning kept indices."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(area_i + area_j - inter, 1e-9)
        idx = np.where(iou <= iou_threshold)[0]
        order = order[idx + 1]
    return keep


def postprocess(raw_output, img_w, img_h):
    """Full post-processing: DFL decode + NMS. Returns list of detections."""
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    expected_dim = 4 * 16 + NUM_CLASSES  # 72
    if raw_output.shape[-1] != expected_dim:
        if raw_output.shape[0] == expected_dim:
            raw_output = raw_output.T
        else:
            print(f"  [WARN] Output shape {raw_output.shape}, expected (*, {expected_dim})")
            return []

    N = raw_output.shape[0]
    box_dim = 4 * 16

    box_pred = raw_output[:, :box_dim]
    cls_pred = raw_output[:, box_dim:]

    decoded = dfl_decode(box_pred)

    if N != TOTAL_ANCHORS:
        print(f"  [WARN] Got {N} anchors, expected {TOTAL_ANCHORS}")

    anchors = ANCHOR_CENTERS[:N]
    strides = ANCHOR_STRIDES_ARR[:N]

    l = decoded[:, 0] * strides
    t = decoded[:, 1] * strides
    r = decoded[:, 2] * strides
    b = decoded[:, 3] * strides

    cx = anchors[:, 0]
    cy = anchors[:, 1]

    x1 = (cx - l) / INPUT_SIZE * img_w
    y1 = (cy - t) / INPUT_SIZE * img_h
    x2 = (cx + r) / INPUT_SIZE * img_w
    y2 = (cy + b) / INPUT_SIZE * img_h

    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    cls_scores = 1.0 / (1.0 + np.exp(-cls_pred))
    class_ids = cls_scores.argmax(axis=-1)
    max_scores = cls_scores.max(axis=-1)

    mask = max_scores > CONF_THRESHOLD
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    if len(boxes) == 0:
        return []

    results = []
    for cls_id in range(NUM_CLASSES):
        cls_mask = class_ids == cls_id
        if not cls_mask.any():
            continue
        cb = boxes[cls_mask]
        cs = max_scores[cls_mask]
        keep = nms(cb, cs, IOU_THRESHOLD)
        for idx in keep:
            results.append({
                "label": LABELS[cls_id + 1],
                "confidence": float(cs[idx]),
                "bbox": [float(v) for v in cb[idx]],
            })

    return results


def main():
    hef_path = os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef")
    print("⛳ Mini Caddie — Golf Inference")
    print(f"  Model: {hef_path}")
    print("  Press Ctrl+C to stop\n")

    hef = HEF(hef_path)
    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStream.ACCELERATOR)
    network_group = target.configure(hef, configure_params)

    input_vstreams_info = hef.get_input_vstream_infos()
    output_vstreams_info = hef.get_output_vstream_infos()

    print(f"  Inputs: {[(i.name, i.shape) for i in input_vstreams_info]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in output_vstreams_info]}")

    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group)

    for params in output_vstreams_params.values():
        params.user_buffer_format = FormatType.FLOAT32

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n  Camera opened. Running inference...\n")

    frame_count = 0
    fps_start = time.time()

    with network_group:
        with InferVStreams(target, network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera read failed!")
                    break

                frame_count += 1
                img_h, img_w = frame.shape[:2]

                input_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_img = input_img.astype(np.float32) / 255.0
                input_img = np.transpose(input_img, (2, 0, 1))
                input_img = np.expand_dims(input_img, axis=0)

                input_dict = {input_vstreams_info[0].name: input_img}
                results = infer_pipeline.infer(input_dict)

                output_name = output_vstreams_info[0].name
                raw_output = results[output_name]

                detections = postprocess(raw_output, img_w, img_h)

                for det in detections:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    label = det["label"]
                    conf = det["confidence"]

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed
                    det_str = ", ".join(f"{d['label']}:{d['confidence']:.0%}" for d in detections)
                    print(f"  FPS: {fps:.1f} | Frame {frame_count} | Detections: {det_str or 'none'}")

                cv2.imshow("Mini Caddie", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("\n Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Stopped!")