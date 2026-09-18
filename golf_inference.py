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
    HailoStreamInterface,
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
    """Full post-processing: NMS on decoded output.
    
    Output shape from HEF: (1, 8400, 8) — DFL already decoded by Hailo.
    8 values per anchor: 4 bbox (cx, cy, w, h in pixels relative to 640) + 4 class scores? 
    Or: 4 bbox (l, t, r, b) + 4 class scores.
    We'll figure out the exact format from the data.
    """
    if raw_output.ndim == 3:
        raw_output = raw_output[0]  # (8400, 8)

    N = raw_output.shape[0]
    print(f"  [DEBUG] Output shape: {raw_output.shape}, sample[0]: {raw_output[0]}")
    
    # With 8 values and 8 classes, this is likely just class scores (no bbox)
    # Or 4 bbox + 4 classes = 8 total
    # Let's check: if first 4 values are in a different range than last 4
    # For now, assume 4 bbox + 4 classes (but we have 8 classes, not 4!)
    
    # Actually with 8 classes and output dim=8, this might be:
    # Option A: just 8 class scores, no bbox (NMS-only output)
    # Option B: 4 bbox + 4 class scores (only 4 classes made it through?)
    
    # Most likely: the compilation merged DFL and the output is 
    # [cx, cy, w, h, cls0, cls1, cls2, cls3] — but that's only 4 classes
    # OR: the 8 values ARE the 8 class scores and bbox is separate
    
    # Let's just try: 4 bbox + 4 classes
    if raw_output.shape[-1] == 8:
        # Could be 4 bbox + 4 classes, or 8 classes only
        # Print sample for debugging
        sample = raw_output[0]
        print(f"  [DEBUG] Sample anchor 0: {sample}")
        print(f"  [DEBUG] Sample anchor 100: {raw_output[100]}")
        print(f"  [DEBUG] Sample anchor 4000: {raw_output[4000]}")
        print(f"  [DEBUG] Min/Max per col: {raw_output.min(axis=0)} / {raw_output.max(axis=0)}")
        
        # If first 4 cols look like coords (larger range) and last 4 like scores (0-1)
        # then it's 4 bbox + 4 classes
        # If all 8 look like scores (0-1 range), it's 8 classes only
        
        col_ranges = raw_output.max(axis=0) - raw_output.min(axis=0)
        print(f"  [DEBUG] Range per col: {col_ranges}")
        
        # Try treating as 4 bbox + 4 classes first
        boxes_raw = raw_output[:, :4]
        cls_pred = raw_output[:, 4:]
        
        # If cls_pred has more than 4 classes, we need different split
        # For now just return empty and debug
        print(f"  [DEBUG] Assuming 4 bbox + 4 classes (may be wrong)")
        
        # Bbox might be [cx, cy, w, h] in pixel coords (relative to 640)
        # Scale to image size
        cx = boxes_raw[:, 0] / INPUT_SIZE * img_w
        cy = boxes_raw[:, 1] / INPUT_SIZE * img_h
        w = boxes_raw[:, 2] / INPUT_SIZE * img_w
        h = boxes_raw[:, 3] / INPUT_SIZE * img_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        
        # Sigmoid class scores
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
        for cls_id in range(cls_pred.shape[-1]):
            cls_mask = class_ids == cls_id
            if not cls_mask.any():
                continue
            cb = boxes[cls_mask]
            cs = max_scores[cls_mask]
            keep = nms(cb, cs, IOU_THRESHOLD)
            for idx in keep:
                label_idx = cls_id + 1  # +1 for background
                if label_idx < len(LABELS):
                    label = LABELS[label_idx]
                else:
                    label = f"class_{cls_id}"
                results.append({
                    "label": label,
                    "confidence": float(cs[idx]),
                    "bbox": [float(v) for v in cb[idx]],
                })
        
        return results
    
    # Fallback: try the original 72-dim approach
    expected_dim = 4 * 16 + NUM_CLASSES  # 72
    if raw_output.shape[-1] == expected_dim:
        return _postprocess_dfl(raw_output, img_w, img_h)
    
    print(f"  [WARN] Unexpected output shape: {raw_output.shape}")
    return []


def _postprocess_dfl(raw_output, img_w, img_h):
    """Original DFL-based postprocess for 72-dim output."""
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

    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)

    input_vstreams_info = hef.get_input_vstream_infos()
    output_vstreams_info = hef.get_output_vstream_infos()

    print(f"  Inputs: {[(i.name, i.shape) for i in input_vstreams_info]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in output_vstreams_info]}")

    network_group_params = network_group.create_params()
    input_vstreams_params = InputVStreamParams.make(network_group_params)
    output_vstreams_params = OutputVStreamParams.make(network_group_params)

    for params in output_vstreams_params:
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
                # Input shape is (640, 640, 3) = HWC, keep as-is
                input_img = np.expand_dims(input_img, axis=0)  # Add batch dim

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