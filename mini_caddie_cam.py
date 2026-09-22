#!/usr/bin/env python3
"""
Mini Caddie Cam — Live golf detection using direct Hailo API.
Reads camera frames with Picamera2, runs inference on the NMS HEF,
and prints golf class detections.

NMS output shape: (1, 8, 5, 100) = (batch, classes, 5_vals, max_boxes)
Each box: [y_min, x_min, y_max, x_max, score] (normalized 0-1)

Usage:
  cd ~/hailo-apps && source setup_env.sh
  python ~/mini-caddie/mini_caddie_cam.py
"""

import os
import time
import numpy as np

# ── Golf class labels (index 0 = background, NOT in NMS output) ─────────────
# NMS output index 0-7 = classes 1-8 from training (background is removed)
LABELS = [
    "golf_ball",
    "golf_club",
    "golf_club_head",
    "golf_hole",
    "golf_mat",
    "person",
    "player_not_ready",
    "player_ready",
]

EMOJI = {
    "golf_ball": "⚪", "golf_club": "🏌️", "golf_club_head": "🔨",
    "golf_hole": "🟤", "golf_mat": "🟩", "person": "🧑",
    "player_not_ready": "🙅", "player_ready": "✅",
}

MIN_CONF = 0.30
HEF_PATH = os.path.expanduser("~/mini-caddie/mini_caddie_golf_nms.hef")


def detect(camera, hef, target, ng, input_params, output_params, input_name, output_name):
    """Run one inference frame and return detections."""
    # Capture frame
    frame = camera.capture_array()

    # Resize to 640x640 (HWC, RGB, uint8)
    import cv2
    frame_resized = cv2.resize(frame, (640, 640))
    if frame_resized.shape[2] == 4:  # RGBA → RGB
        frame_resized = frame_resized[:, :, :3]
    input_array = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

    # Infer
    with InferVStreams(ng, input_params, output_params, target) as pipe:
        results = pipe.infer({input_name: input_array})
        nms_output = results[output_name]  # (1, 8, 5, 100)

    # Decode NMS output
    detections = []
    nms = nms_output[0]  # (8, 5, 100)
    for cls_idx in range(8):
        for box_idx in range(100):
            score = nms[cls_idx, 4, box_idx]
            if score >= MIN_CONF:
                y_min = nms[cls_idx, 0, box_idx]
                x_min = nms[cls_idx, 1, box_idx]
                y_max = nms[cls_idx, 2, box_idx]
                x_max = nms[cls_idx, 3, box_idx]
                detections.append({
                    "label": LABELS[cls_idx],
                    "confidence": float(score),
                    "bbox": [float(y_min), float(x_min), float(y_max), float(x_max)],
                })
    return detections, frame


def main():
    from hailo_platform import (
        HEF, VDevice, InferVStreams, ConfigureParams,
        HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
        FormatType,
    )
    from picamera2 import Picamera2
    import cv2

    print("\n" + "=" * 55)
    print("  ⛳ MINI CADDIE CAM")
    print("  Direct Hailo API — NMS HEF")
    print("  8 golf classes | Press Ctrl+C to stop")
    print("=" * 55 + "\n")

    # ── Init Hailo ──────────────────────────────────────────────────────────
    hef = HEF(HEF_PATH)
    target = VDevice()
    cp = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = target.configure(hef, cp)[0]

    input_info = hef.get_input_vstream_infos()
    output_info = hef.get_output_vstream_infos()
    input_name = input_info[0].name
    output_name = output_info[0].name

    input_params = InputVStreamParams.make_from_network_group(ng)
    output_params = OutputVStreamParams.make_from_network_group(ng)
    for p in output_params.values():
        p.user_buffer_format.type = FormatType.FLOAT32

    activated = ng.activate()

    # ── Init camera ─────────────────────────────────────────────────────────
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    camera.start()

    # ── Live loop ───────────────────────────────────────────────────────────
    frame_count = 0
    start_time = time.time()
    det_counts = {}

    with activated:
        try:
            while True:
                frame_count += 1
                detections, frame = detect(
                    camera, hef, target, ng,
                    input_params, output_params,
                    input_name, output_name
                )

                # Print detections
                if detections:
                    print(f"  ── Frame {frame_count} ───")
                    for det in detections:
                        label = det["label"]
                        conf = det["confidence"]
                        emoji = EMOJI.get(label, "❓")
                        print(f"    {emoji} {label:<16} {conf:.0%}")
                        det_counts[label] = det_counts.get(label, 0) + 1
                elif frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"  [Frame {frame_count}] {fps:.1f} FPS — no detections")

        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 55)
    print("  📊 SESSION SUMMARY")
    print(f"  Frames: {frame_count} | Duration: {elapsed:.0f}s | FPS: {frame_count/elapsed:.1f}")
    if det_counts:
        print("  Detections:")
        for label, count in sorted(det_counts.items(), key=lambda x: -x[1]):
            print(f"    {EMOJI.get(label, '❓')} {label:<16} {count}")
    else:
        print("  No golf objects detected.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()