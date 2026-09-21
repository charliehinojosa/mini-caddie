#!/usr/bin/env python3
"""
Mini Caddie — Golf Inference (DEBUG MODE)
Uses hailo_platform Python bindings directly — no GStreamer pipeline.

This version runs ONE frame, dumps the raw output tensor values, and exits.
No camera display loop — just debug output to figure out the (8400, 8) format.

Usage:
  cd ~/hailo-apps && source setup_env.sh
  python ~/mini-caddie/golf_inference.py
"""
import os
import sys
import numpy as np
import cv2

from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,  # NOT HailoStream
    InputVStreamParams,
    OutputVStreamParams,
)

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_SIZE = 640
HEF_PATH = os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef")

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("⛳ Mini Caddie — DEBUG MODE")
    print(f"  Model: {HEF_PATH}")
    print("=" * 60)

    # Load HEF
    hef = HEF(HEF_PATH)
    target = VDevice()

    # Configure — configure() returns a LIST
    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    # Get I/O info
    input_vstreams_info = hef.get_input_vstream_infos()
    output_vstreams_info = hef.get_output_vstream_infos()

    print(f"\n📥 Inputs:")
    for i in input_vstreams_info:
        print(f"  {i.name}: shape={i.shape}")

    print(f"\n📤 Outputs:")
    for o in output_vstreams_info:
        print(f"  {o.name}: shape={o.shape}")

    # Create vstream params — CORRECTED API
    network_group_params = network_group.create_params()
    input_vstreams_params = InputVStreamParams.make(network_group_params)
    output_vstreams_params = OutputVStreamParams.make(network_group_params)

    # Set output format to FLOAT32 — output_vstreams_params is a LIST, not dict
    for params in output_vstreams_params:
        params.format_type = FormatType.FLOAT32

    # Create a test image (640x640, HWC, RGB, normalized)
    # Use a simple gradient so we get non-trivial output
    test_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    # Add a gradient pattern
    for y in range(INPUT_SIZE):
        for x in range(INPUT_SIZE):
            test_img[y, x, 0] = x / INPUT_SIZE  # R gradient
            test_img[y, x, 1] = y / INPUT_SIZE  # G gradient
            test_img[y, x, 2] = 0.5              # B constant
    test_img = np.expand_dims(test_img, axis=0)  # (1, 640, 640, 3)

    print(f"\n🖼️  Test input shape: {test_img.shape}")
    print(f"  Input range: [{test_img.min():.3f}, {test_img.max():.3f}]")

    # Run inference on ONE frame
    print("\n🧠 Running inference on test image...")

    with network_group:
        with InferVStreams(
            target, network_group, input_vstreams_params, output_vstreams_params
        ) as infer_pipeline:
            input_dict = {input_vstreams_info[0].name: test_img}
            results = infer_pipeline.infer(input_dict)

            # Get the output
            output_name = output_vstreams_info[0].name
            raw_output = results[output_name]

            print(f"\n{'=' * 60}")
            print(f"📊 RAW OUTPUT ANALYSIS")
            print(f"{'=' * 60}")
            print(f"  Output name: {output_name}")
            print(f"  Output shape: {raw_output.shape}")
            print(f"  Output dtype: {raw_output.dtype}")
            print(f"  Output size: {raw_output.size} elements")

            # Squeeze batch dim if present
            if raw_output.ndim == 3:
                flat = raw_output[0]  # (8400, 8)
            else:
                flat = raw_output

            N, D = flat.shape
            print(f"  Flat shape: ({N}, {D})")
            print(f"  N (anchors): {N}")
            print(f"  D (values per anchor): {D}")

            print(f"\n📈 Per-column statistics:")
            print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
            for col in range(D):
                col_data = flat[:, col]
                print(
                    f"  {col:4d}  {col_data.min():10.4f}  {col_data.max():10.4f}  "
                    f"{col_data.mean():10.4f}  {col_data.std():10.4f}"
                )

            print(f"\n🔢 Sample anchors (first 10):")
            for i in range(min(10, N)):
                print(f"  anchor[{i:4d}]: {flat[i]}")

            print(f"\n🔢 Sample anchors (middle 10):")
            mid = N // 2
            for i in range(mid, mid + 10):
                print(f"  anchor[{i:4d}]: {flat[i]}")

            print(f"\n🔢 Sample anchors (last 10):")
            for i in range(max(0, N - 10), N):
                print(f"  anchor[{i:4d}]: {flat[i]}")

            # Check if values look like they need sigmoid (0-1 range after sigmoid)
            print(f"\n🔬 Analysis:")
            print(f"  All values in [0, 1]?  {bool((flat >= 0).all() and (flat <= 1).all())}")
            print(f"  All values in [-1, 1]? {bool((flat >= -1).all() and (flat <= 1).all())}")
            print(f"  Any negative values?   {bool((flat < 0).any())}")
            print(f"  Any values > 1?        {bool((flat > 1).any())}")
            print(f"  Any values > 10?       {bool((flat > 10).any())}")
            print(f"  Any values > 100?      {bool((flat > 100).any())}")
            print(f"  Any values > 640?      {bool((flat > 640).any())}")

            # Check if first 4 cols look like coordinates (larger range)
            if D >= 4:
                bbox_range = flat[:, :4].max() - flat[:, :4].min()
                cls_range = flat[:, 4:].max() - flat[:, 4:].min()
                print(f"\n  First 4 cols range: {bbox_range:.4f}")
                print(f"  Last {D - 4} cols range: {cls_range:.4f}")
                if bbox_range > cls_range * 2:
                    print(f"  → First 4 cols look like COORDINATES (larger range)")
                else:
                    print(f"  → Ranges are similar — could all be scores or all coords")

            # If all values are in 0-1, they might already be sigmoid'd
            if (flat >= 0).all() and (flat <= 1).all():
                print(f"\n  → Values in [0,1] — likely already sigmoid'd class scores")
                print(f"  → 8 values = 8 class scores (NO bbox in output)")
                print(f"  → Bbox may need to be decoded from a different output layer")
            elif (flat[:, :4] > 1).any():
                print(f"\n  → First 4 cols have values > 1 — likely raw coordinates")
                print(f"  → Format: [cx, cy, w, h, cls0, cls1, cls2, cls3]")

            # Also try with a real camera frame if available
            print(f"\n{'=' * 60}")
            print(f"📷 Attempting camera frame...")
            print(f"{'=' * 60}")

            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  Camera frame shape: {frame.shape}")
                    img_h, img_w = frame.shape[:2]

                    # Preprocess: HWC format
                    input_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)

                    input_dict = {input_vstreams_info[0].name: input_img}
                    cam_results = infer_pipeline.infer(input_dict)

                    cam_output = cam_results[output_name]
                    if cam_output.ndim == 3:
                        cam_flat = cam_output[0]
                    else:
                        cam_flat = cam_output

                    print(f"\n  Camera output shape: {cam_output.shape}")
                    print(f"  Camera flat shape: {cam_flat.shape}")

                    print(f"\n  Per-column stats (camera frame):")
                    print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
                    for col in range(cam_flat.shape[-1]):
                        col_data = cam_flat[:, col]
                        print(
                            f"  {col:4d}  {col_data.min():10.4f}  {col_data.max():10.4f}  "
                            f"{col_data.mean():10.4f}  {col_data.std():10.4f}"
                        )

                    # Show top-10 highest-scoring anchors
                    max_scores = cam_flat.max(axis=-1)
                    top10 = max_scores.argsort()[-10:][::-1]
                    print(f"\n  Top 10 anchors by max value:")
                    for idx in top10:
                        print(
                            f"  anchor[{idx:4d}]: max={max_scores[idx]:.4f}  "
                            f"values={cam_flat[idx]}"
                        )

                    # If 8 values, check top anchors per column
                    if cam_flat.shape[-1] == 8:
                        print(f"\n  Top 5 anchors per column (camera frame):")
                        for col in range(8):
                            top5 = cam_flat[:, col].argsort()[-5:][::-1]
                            print(f"  col {col}: ", end="")
                            for idx in top5:
                                print(f"[{idx}:{cam_flat[idx, col]:.3f}] ", end="")
                            print()
                else:
                    print("  ❌ Camera read failed")
                cap.release()
            else:
                print("  ⚠️  No camera available (that's OK — test image output above is enough)")

    print(f"\n{'=' * 60}")
    print("✅ Debug complete! Copy this output and send it to Caddie.")
    print("   We'll use it to determine the exact output format.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()