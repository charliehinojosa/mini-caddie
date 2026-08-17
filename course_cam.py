#!/usr/bin/env python3
"""
Mini Caddie — Course Cam
Golf-focused object detection using Hailo AI HAT+ on Raspberry Pi 5.

Filters the standard YOLO detection pipeline to show golf-relevant objects:
- Sports balls (golf ball detection)
- Persons (players, caddies)
- Other relevant course objects

Future: will be replaced with a custom-trained golf model (flags, holes, clubs, hazards).
"""

import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

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

# ── Golf-relevant COCO classes ──────────────────────────────────────────────
# These are the standard YOLO/COCO classes that map to golf scene objects.
# We'll expand this once we train a custom golf model.
GOLF_RELEVANT = {
    "sports ball": "⛳ Ball",
    "person": "🏌️ Player",
    "umbrella": "☂️ Umbrella",
    "bottle": "🍶 Bottle",
    "cup": "🥤 Cup",
    "chair": "🪑 Chair",
    "bench": "🪑 Bench",
    "backpack": "🎒 Bag",
    "tie": "👔 Tie",
}

# Confidence threshold for display
MIN_CONFIDENCE = 0.40


class CourseCamCallback(app_callback_class):
    """Track golf-relevant detections across frames."""

    def __init__(self):
        super().__init__()
        self.golf_detections = 0
        self.total_frames = 0
        self.last_sport_ball = None  # Track last ball position


def app_callback(element, buffer, user_data):
    """Process each frame — filter for golf-relevant objects."""
    frame_idx = user_data.get_count()
    user_data.total_frames += 1

    if buffer is None:
        return Gst.PadProbeReturn.OK

    detections = hailo.get_roi_from_buffer(buffer).get_objects_typed(
        hailo.HAILO_DETECTION
    )

    golf_objects = []
    for det in detections:
        label = det.get_label()
        confidence = det.get_confidence()

        # Skip low confidence
        if confidence < MIN_CONFIDENCE:
            continue

        # Check if this is a golf-relevant object
        if label in GOLF_RELEVANT:
            display_name = GOLF_RELEVANT[label]
            golf_objects.append((display_name, label, confidence))

            # Track ball specifically
            if label == "sports ball":
                user_data.last_sport_ball = (frame_idx, confidence)
                user_data.golf_detections += 1

    # Print golf-relevant detections only
    if golf_objects:
        output = f"─── Course Cam Frame {frame_idx} ───\n"
        for display_name, label, conf in golf_objects:
            output += f"  {display_name}  ({label})  conf: {conf:.0%}\n"
        if user_data.last_sport_ball:
            ball_frame, ball_conf = user_data.last_sport_ball
            frames_ago = frame_idx - ball_frame
            output += f"  ⛳ Last ball seen: {frames_ago} frames ago ({ball_conf:.0%})\n"
        output += f"  Total golf detections: {user_data.golf_detections}\n"
        print(output)

    return Gst.PadProbeReturn.OK


def main():
    hailo_logger.info("🏌️ Mini Caddie — Course Cam starting...")
    print("\n" + "=" * 50)
    print("  ⛳ MINI CADDIE — COURSE CAM")
    print("  Golf-focused object detection")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    user_data = CourseCamCallback()
    app = GStreamerDetectionSimpleApp(app_callback, user_data)
    app.run()


if __name__ == "__main__":
    main()