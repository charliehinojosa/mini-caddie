#!/usr/bin/env python3
"""
Mini Caddie Cam — Golf-focused live detection with custom YOLOv8n model.
Uses the NMS-compiled HEF (mini_caddie_golf_nms.hef) for on-chip NMS.
Filters to 8 golf classes, shows clean overlay, logs detections.

Usage:
  cd ~/hailo-apps
  source setup_env.sh
  python ~/mini-caddie/mini_caddie_cam.py --input rpi \
    --hef-path ~/mini-caddie/mini_caddie_golf_nms.hef \
    --labels-json ~/mini-caddie/labels.json
"""

import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import argparse
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import hailo
from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

hailo_logger = get_logger(__name__)

# ── Golf class labels (background at index 0 from Hailo compilation) ────────
GOLF_CLASSES = {
    "golf_ball":       {"emoji": "⚪", "color": "\033[97m",  "min_conf": 0.30},
    "golf_club":       {"emoji": "🏌️", "color": "\033[93m",  "min_conf": 0.35},
    "golf_club_head":  {"emoji": "🔨", "color": "\033[92m",  "min_conf": 0.35},
    "golf_hole":       {"emoji": "🟤", "color": "\033[33m",  "min_conf": 0.35},
    "golf_mat":        {"emoji": "🟩", "color": "\033[32m",  "min_conf": 0.35},
    "person":          {"emoji": "🧑", "color": "\033[96m",  "min_conf": 0.40},
    "player_not_ready":{"emoji": "🙅", "color": "\033[91m",  "min_conf": 0.40},
    "player_ready":    {"emoji": "✅", "color": "\033[92m",  "min_conf": 0.40},
}

RESET = "\033[0m"


class MiniCaddieCallback(app_callback_class):
    """Track golf detections across frames."""

    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.detection_counts = {}  # label → total count
        self.last_ball_frame = None
        self.last_ball_conf = None
        self.start_time = time.time()
        self.fps_log_interval = 100


def app_callback(element, buffer, user_data):
    """Process each frame — filter for golf classes, log stats."""
    frame_idx = user_data.get_count()
    user_data.frame_count += 1

    # FPS logging
    if user_data.frame_count % user_data.fps_log_interval == 0:
        elapsed = time.time() - user_data.start_time
        fps = user_data.frame_count / elapsed
        print(f"\n  📊 {user_data.frame_count} frames | {fps:.1f} FPS | "
              f"elapsed: {elapsed:.0f}s\n")

    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get ALL detections for debugging
    all_detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(
        hailo.HAILO_DETECTION
    )

    # Debug: print raw labels every 30 frames
    if user_data.frame_count % 30 == 0:
        labels_seen = set()
        for det in all_detections:
            labels_seen.add(det.get_label())
        print(f"  [DEBUG] Frame {frame_idx}: {len(all_detections)} detections, labels: {labels_seen}")

    detections = all_detections

    golf_objects = []
    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()

        # Skip background and non-golf labels
        if label not in GOLF_CLASSES:
            continue

        class_cfg = GOLF_CLASSES[label]
        if confidence < class_cfg["min_conf"]:
            continue

        golf_objects.append((label, confidence, class_cfg))
        user_data.detection_counts[label] = user_data.detection_counts.get(label, 0) + 1

        if label == "golf_ball":
            user_data.last_ball_frame = frame_idx
            user_data.last_ball_conf = confidence

    # Print detections
    if golf_objects:
        print(f"  ── Frame {frame_idx} ──")
        for label, conf, cfg in golf_objects:
            color = cfg["color"]
            emoji = cfg["emoji"]
            print(f"    {emoji} {color}{label:<16}{RESET} {conf:.0%}")

        if user_data.last_ball_frame and user_data.last_ball_frame != frame_idx:
            frames_ago = frame_idx - user_data.last_ball_frame
            print(f"    ⚪ Last ball: {frames_ago} frames ago ({user_data.last_ball_conf:.0%})")

    return Gst.PadProbeReturn.OK


def main():
    print("\n" + "=" * 55)
    print("  ⛳ MINI CADDIE CAM")
    print("  Custom YOLOv8n — 8 golf classes")
    print("  On-chip NMS — live inference")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    print("\n  Classes:")
    for label, cfg in GOLF_CLASSES.items():
        print(f"    {cfg['emoji']} {label:<16} min conf: {cfg['min_conf']:.0%}")
    print()

    # GStreamerDetectionApp uses get_pipeline_parser() which adds --hef-path,
    # --input, --arch, etc. automatically. We also add --labels-json.
    user_data = MiniCaddieCallback()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()

    # ── Summary on exit ─────────────────────────────────────────────────────
    elapsed = time.time() - user_data.start_time
    print("\n" + "=" * 55)
    print("  📊 SESSION SUMMARY")
    print(f"  Frames: {user_data.frame_count}")
    print(f"  Duration: {elapsed:.0f}s")
    if user_data.frame_count > 0:
        print(f"  Avg FPS: {user_data.frame_count / elapsed:.1f}")
    print()
    if user_data.detection_counts:
        print("  Detections by class:")
        for label, count in sorted(user_data.detection_counts.items(),
                                    key=lambda x: -x[1]):
            cfg = GOLF_CLASSES.get(label, {"emoji": "❓"})
            print(f"    {cfg['emoji']} {label:<16} {count}")
    else:
        print("  No golf objects detected.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()