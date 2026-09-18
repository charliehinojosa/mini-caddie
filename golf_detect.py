#!/usr/bin/env python3
"""
Mini Caddie — Custom Golf Detection
Runs our custom YOLOv8n HEF model on Hailo AI HAT+ with Pi-side NMS post-processing.

The model was compiled with end_node at /model.22/Concat_1 (before DFL/NMS),
so NMS must be done on the Pi side.

Usage:
  cd ~/hailo-apps
  source setup_env.sh
  python ~/mini-caddie/golf_detect.py --input rpi
"""

import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import sys
import json
import argparse
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import hailo
from hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline import (
    GStreamerDetectionSimpleApp,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)

# ── Labels ──────────────────────────────────────────────────────────────────
LABELS = [
    "background",
    "golf_ball",
    "golf_club",
    "golf_club_head",
    "golf_hole",
    "golf_mat",
    "person",
    "player_not_ready",
    "player_ready",
]

# ── NMS Parameters ──────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_CLASSES = 8  # excluding background
INPUT_SIZE = 640


def dfl_decode(box_pred):
    """Decode DFL (Distribution Focal Loss) box predictions.
    
    box_pred shape: (num_anchors, 4, num_bins) or (num_anchors, 4*num_bins)
    Returns: (num_anchors, 4) in [cx, cy, w, h] format
    """
    if box_pred.ndim == 2:
        # Reshape from (N, 4*num_bins) to (N, 4, num_bins)
        num_bins = box_pred.shape[-1] // 4
        box_pred = box_pred.reshape(-1, 4, num_bins)
    
    # Softmax over bins dimension
    exp = np.exp(box_pred - box_pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    
    # Expected value: sum(bin_idx * prob)
    num_bins = prob.shape[-1]
    bins = np.arange(num_bins, dtype=np.float32)
    decoded = (prob * bins).sum(axis=-1)  # (N, 4)
    
    return decoded


def xywh_to_xyxy(boxes, img_w, img_h):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] scaled to image size."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        # Compute IoU with rest
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_j - inter
        iou = inter / np.maximum(union, 1e-9)
        
        idx = np.where(iou <= iou_threshold)[0]
        order = order[idx + 1]
    
    return keep


def postprocess_detections(raw_output, img_w, img_h):
    """
    Post-process raw YOLOv8 output (cut at Concat_1 — before DFL/NMS).
    
    The output is the concatenation of 3 detection heads:
    - P3 (80x80): 80*80*80 = 512,000 values (4 box coords * 16 bins + 8 classes)
    - P4 (40x40): 40*40*80 = 128,000 values  
    - P5 (20x20): 20*20*80 = 32,000 values
    
    Each anchor: (4*16 + 8) = 72 values for YOLOv8n with 8 classes
    Total anchors: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
    """
    # The raw output shape depends on how Hailo arranges it
    # Typical: (1, 8400, 72) or (8400, 72) — 4*16 DFL bins + 8 classes
    if raw_output.ndim == 3:
        raw_output = raw_output[0]  # remove batch dim
    
    num_anchors = raw_output.shape[0]
    box_dim = 4 * 16  # 64 DFL values
    total_dim = box_dim + NUM_CLASSES
    
    if raw_output.shape[-1] != total_dim:
        # Try to figure out the layout
        print(f"  [DEBUG] Raw output shape: {raw_output.shape}")
        print(f"  [DEBUG] Expected last dim: {total_dim} (64 box + 8 classes)")
        # Maybe the output is transposed or different layout
        if raw_output.shape[0] == total_dim:
            raw_output = raw_output.T
            num_anchors = raw_output.shape[0]
        else:
            print(f"  [WARNING] Unexpected output shape: {raw_output.shape}")
            return []
    
    # Split into box predictions and class scores
    box_pred = raw_output[:, :box_dim]  # (N, 64)
    cls_pred = raw_output[:, box_dim:]  # (N, 8)
    
    # DFL decode: (N, 64) -> (N, 4, 16) -> softmax -> (N, 4) [l, t, r, b] offsets
    box_pred = box_pred.reshape(-1, 4, 16)
    exp = np.exp(box_pred - box_pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    bins = np.arange(16, dtype=np.float32)
    decoded = (prob * bins).sum(axis=-1)  # (N, 4) — l, t, r, b in grid units * stride
    
    # Convert ltbr to xywh
    # Anchor grid centers depend on which head — but since Hailo concatenates,
    # we need the anchor positions. For now, use a simplified approach.
    # The 3 heads have strides 8, 16, 32 for 80x80, 40x40, 20x20
    # Anchors are ordered: P3 first (6400), then P4 (1600), then P5 (400)
    
    anchor_centers = []
    for stride, grid_size in [(8, 80), (16, 40), (32, 20)]:
        for y in range(grid_size):
            for x in range(grid_size):
                anchor_centers.append([x * stride + stride // 2, y * stride + stride // 2])
    anchor_centers = np.array(anchor_centers, dtype=np.float32)  # (8400, 2)
    
    if num_anchors != len(anchor_centers):
        print(f"  [WARNING] Anchor count mismatch: {num_anchors} vs {len(anchor_centers)}")
        # Try transposing
        if raw_output.shape[0] == 72 and raw_output.shape[1] == 8400:
            raw_output = raw_output.T
            num_anchors = raw_output.shape[0]
            box_pred = raw_output[:, :box_dim]
            cls_pred = raw_output[:, box_dim:]
            box_pred = box_pred.reshape(-1, 4, 16)
            exp = np.exp(box_pred - box_pred.max(axis=-1, keepdims=True))
            prob = exp / exp.sum(axis=-1, keepdims=True)
            decoded = (prob * bins).sum(axis=-1)
    
    # decoded is (N, 4) — [l, t, r, b] distances from anchor center
    # Convert to [x1, y1, x2, y2]
    cx = anchor_centers[:, 0]
    cy = anchor_centers[:, 1]
    
    # Get stride for each anchor
    strides = np.concatenate([
        np.full(6400, 8),
        np.full(1600, 16),
        np.full(400, 32),
    ])
    
    l = decoded[:, 0] * strides
    t = decoded[:, 1] * strides
    r = decoded[:, 2] * strides
    b = decoded[:, 3] * strides
    
    x1 = (cx - l) / INPUT_SIZE * img_w
    y1 = (cy - t) / INPUT_SIZE * img_h
    x2 = (cx + r) / INPUT_SIZE * img_w
    y2 = (cy + b) / INPUT_SIZE * img_h
    
    boxes = np.stack([x1, y1, x2, y2], axis=-1)
    
    # Class scores — sigmoid
    cls_scores = 1 / (1 + np.exp(-cls_pred))  # sigmoid
    class_ids = cls_scores.argmax(axis=-1)
    max_scores = cls_scores.max(axis=-1)
    
    # Filter by confidence
    mask = max_scores > CONF_THRESHOLD
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]
    
    if len(boxes) == 0:
        return []
    
    # NMS per class
    results = []
    for cls_id in range(NUM_CLASSES):
        cls_mask = class_ids == cls_id
        if not cls_mask.any():
            continue
        cls_boxes = boxes[cls_mask]
        cls_scores = max_scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)
        for idx in keep:
            results.append({
                "label": LABELS[cls_id + 1],  # +1 for background offset
                "confidence": float(cls_scores[idx]),
                "bbox": [float(v) for v in cls_boxes[idx]],
            })
    
    return results


class GolfDetectCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_count = 0


def app_callback(element, buffer, user_data):
    frame_idx = user_data.get_count()
    user_data.frame_count += 1

    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get raw Hailo output from the buffer
    ros = hailo.get_roi_from_buffer(buffer)
    
    # Try to get raw tensor data — our model doesn't have NMS on chip,
    # so we get raw detection head output, not HAILO_DETECTION objects
    try:
        objects = ros.get_objects_typed(hailo.HAILO_DETECTION)
        if objects:
            # If Hailo did provide detection objects, use them directly
            golf_objects = []
            for det in objects:
                label = det.get_label()
                conf = det.get_confidence()
                if conf < CONF_THRESHOLD:
                    continue
                if label in LABELS:
                    golf_objects.append((label, conf))
            
            if golf_objects:
                output = f"─── Frame {frame_idx} ───\n"
                for label, conf in golf_objects:
                    output += f"  {label}: {conf:.0%}\n"
                print(output)
            return Gst.PadProbeReturn.OK
    except Exception:
        pass
    
    # Try to get raw tensors for Pi-side post-processing
    try:
        tensors = ros.get_objects_typed(hailo.HAILO_TENSOR)
        if tensors:
            for tensor in tensors:
                name = tensor.name()
                shape = tensor.shape()
                print(f"  [DEBUG] Tensor: {name}, shape: {shape}")
                
                # Get tensor data
                data = np.frombuffer(tensor.data(), dtype=np.float32)
                data = data.reshape(shape)
                
                # Post-process
                detections = postprocess_detections(data, 640, 640)
                
                if detections:
                    output = f"─── Frame {frame_idx} ───\n"
                    for det in detections:
                        output += f"  {det['label']}: {det['confidence']:.0%}  bbox={det['bbox']}\n"
                    print(output)
                elif user_data.frame_count % 30 == 0:
                    print(f"  Frame {frame_idx}: no detections")
    except Exception as e:
        if user_data.frame_count % 30 == 0:
            print(f"  [DEBUG] Tensor access error: {e}")

    return Gst.PadProbeReturn.OK


def main():
    parser = argparse.ArgumentParser(description="Mini Caddie Golf Detection")
    parser.add_argument("--input", "-i", default="rpi", help="Input source (rpi, usb, or file path)")
    parser.add_argument("--hef-path", default=os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"),
                        help="Path to HEF model file")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  ⛳ MINI CADDIE — GOLF DETECTION")
    print("  Custom YOLOv8n + Pi-side NMS")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    hailo_logger.info("🏌️ Starting golf detection...")

    user_data = GolfDetectCallback()
    
    # Pass HEF path to the app
    import hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline as dsp
    original_init = dsp.GStreamerDetectionSimpleApp.__init__
    
    def patched_init(self, callback, user_data):
        original_init(self, callback, user_data)
        self._hef_path = args.hef_path
    
    dsp.GStreamerDetectionSimpleApp.__init__ = patched_init
    
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()