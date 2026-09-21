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
    HailoStreamInterface,
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

    hef = HEF(HEF_PATH)
    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    input_vstreams_info = hef.get_input_vstream_infos()
    output_vstreams_info = hef.get_output_vstream_infos()

    print(f"\n📥 Inputs:")
    for i in input_vstreams_info:
        print(f"  {i.name}: shape={i.shape}")

    print(f"\n📤 Outputs:")
    for o in output_vstreams_info:
        print(f"  {o.name}: shape={o.shape}")

    # Test image: 640x640 gradient, HWC, RGB, normalized
    test_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    for y in range(INPUT_SIZE):
        for x in range(INPUT_SIZE):
            test_img[y, x, 0] = x / INPUT_SIZE
            test_img[y, x, 1] = y / INPUT_SIZE
            test_img[y, x, 2] = 0.5
    test_img = np.expand_dims(test_img, axis=0)  # (1, 640, 640, 3)

    print(f"\n🖼️  Test input shape: {test_img.shape}")

    # Create vstream params — need ActivatedNetworkGroup
    # ConfiguredNetwork.activate() returns ActivatedNetworkGroup
    activated = network_group.activate()
    with activated:
        input_vstreams_params = InputVStreamParams.make(activated)
        output_vstreams_params = OutputVStreamParams.make(activated)

        # Set output to FLOAT32 — try both attr names
        for params in output_vstreams_params:
            try:
                params.format_type = FormatType.FLOAT32
            except AttributeError:
                params.user_buffer_format = FormatType.FLOAT32

        with InferVStreams(
            target, activated, input_vstreams_params, output_vstreams_params
        ) as infer_pipeline:

            # ── Run 1: Test gradient image ──
            print("\n🧠 Running inference on test gradient...")
            input_dict = {input_vstreams_info[0].name: test_img}
            results = infer_pipeline.infer(input_dict)

            output_name = output_vstreams_info[0].name
            raw_output = results[output_name]

            print(f"\n{'=' * 60}")
            print(f"📊 RAW OUTPUT ANALYSIS (test image)")
            print(f"{'=' * 60}")
            print(f"  Output name: {output_name}")
            print(f"  Output shape: {raw_output.shape}")
            print(f"  Output dtype: {raw_output.dtype}")

            if raw_output.ndim == 3:
                flat = raw_output[0]
            else:
                flat = raw_output

            N, D = flat.shape
            print(f"  Flat shape: ({N}, {D})")

            print(f"\n📈 Per-column statistics:")
            print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
            for col in range(D):
                c = flat[:, col]
                print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")

            print(f"\n🔢 Sample anchors (first 10):")
            for i in range(min(10, N)):
                print(f"  [{i:4d}]: {flat[i]}")

            print(f"\n🔢 Sample anchors (middle 10):")
            mid = N // 2
            for i in range(mid, mid + 10):
                print(f"  [{i:4d}]: {flat[i]}")

            print(f"\n🔢 Sample anchors (last 10):")
            for i in range(max(0, N - 10), N):
                print(f"  [{i:4d}]: {flat[i]}")

            print(f"\n🔬 Analysis:")
            in01 = bool((flat >= 0).all() and (flat <= 1).all())
            has_neg = bool((flat < 0).any())
            print(f"  All in [0,1]?     {in01}")
            print(f"  Any negative?     {has_neg}")
            print(f"  Any > 1?          {bool((flat > 1).any())}")
            print(f"  Any > 10?         {bool((flat > 10).any())}")
            print(f"  Any > 640?        {bool((flat > 640).any())}")

            if D >= 4:
                r1 = flat[:, :4].max() - flat[:, :4].min()
                r2 = flat[:, 4:].max() - flat[:, 4:].min()
                print(f"  First 4 cols range: {r1:.4f}")
                print(f"  Last {D-4} cols range: {r2:.4f}")

            # ── Run 2: Camera frame ──
            print(f"\n{'=' * 60}")
            print(f"📷 Camera frame...")
            print(f"{'=' * 60}")

            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  Camera shape: {frame.shape}")
                    img_h, img_w = frame.shape[:2]

                    input_img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)

                    cam_results = infer_pipeline.infer(
                        {input_vstreams_info[0].name: input_img}
                    )
                    cam_out = cam_results[output_name]
                    cam_flat = cam_out[0] if cam_out.ndim == 3 else cam_out

                    print(f"  Camera output shape: {cam_out.shape}")
                    print(f"  Flat shape: {cam_flat.shape}")

                    print(f"\n  Per-column stats (camera):")
                    print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
                    for col in range(cam_flat.shape[-1]):
                        c = cam_flat[:, col]
                        print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")

                    # Top 10 highest-scoring anchors
                    max_vals = cam_flat.max(axis=-1)
                    top10 = max_vals.argsort()[-10:][::-1]
                    print(f"\n  Top 10 anchors by max value:")
                    for idx in top10:
                        print(f"  [{idx:4d}]: max={max_vals[idx]:.4f}  vals={cam_flat[idx]}")

                    # Top 5 per column
                    D2 = cam_flat.shape[-1]
                    print(f"\n  Top 5 anchors per column:")
                    for col in range(D2):
                        top5 = cam_flat[:, col].argsort()[-5:][::-1]
                        vals = " ".join(f"[{i}:{cam_flat[i,col]:.3f}]" for i in top5)
                        print(f"  col {col}: {vals}")
                else:
                    print("  ❌ Camera read failed")
                cap.release()
            else:
                print("  ⚠️  No camera — test image output above is enough")

    print(f"\n{'=' * 60}")
    print("✅ Debug complete! Copy this output and send it to Caddie.")
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