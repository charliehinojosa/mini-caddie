#!/usr/bin/env python3
"""
Mini Caddie — Golf Inference DEBUG (WORKING — final fix)
InferVStreams(ng, inp, out, target) ✅
Input dtype: uint8 (NOT float32)
"""
import os
import numpy as np
import cv2
from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

INPUT_SIZE = 640
HEF_PATH = os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef")

hef = HEF(HEF_PATH)
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
ng = network_groups[0]

input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()

print(f"📥 Inputs: {[(i.name, i.shape) for i in input_info]}")
print(f"📤 Outputs: {[(o.name, o.shape) for o in output_info]}")

# Create vstream params — CONFIRMED WORKING
inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)

# Skip setting output format — use default

# Test image: uint8, HWC, RGB
test_img = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
for y in range(INPUT_SIZE):
    for x in range(INPUT_SIZE):
        test_img[0, y, x, 0] = int(x / INPUT_SIZE * 255)
        test_img[0, y, x, 1] = int(y / INPUT_SIZE * 255)
        test_img[0, y, x, 2] = 128

print(f"\n🖼️ Test input: {test_img.shape}, dtype: {test_img.dtype}")

activated = ng.activate()
with activated:
    with InferVStreams(ng, inp_params, out_params, target) as pipeline:
        # Run 1: test gradient
        print("\n🧠 Inference on test gradient...")
        input_dict = {input_info[0].name: test_img}
        results = pipeline.infer(input_dict)

        out_name = output_info[0].name
        raw = results[out_name]

        print(f"\n{'='*60}")
        print(f"📊 RAW OUTPUT (test image)")
        print(f"{'='*60}")
        print(f"  Type: {type(raw)}")
        print(f"  Shape: {raw.shape}")
        print(f"  Dtype: {raw.dtype}")

        # Handle any number of dims
        flat = np.array(raw)
        while flat.ndim > 2:
            flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])

        N, D = flat.shape
        print(f"  Flat: ({N}, {D})")

        print(f"\n📈 Per-column stats:")
        print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
        for col in range(D):
            c = flat[:, col]
            print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")

        print(f"\n🔢 Samples:")
        for i in [0, 1, 100, 1000, 4000, 8000, 8399]:
            if i < N:
                print(f"  [{i:4d}]: {flat[i]}")

        print(f"\n🔬 Analysis:")
        print(f"  All in [0,1]?  {bool((flat >= 0).all() and (flat <= 1).all())}")
        print(f"  Any > 1?       {bool((flat > 1).any())}")
        print(f"  Any > 10?      {bool((flat > 10).any())}")
        print(f"  Any > 640?     {bool((flat > 640).any())}")
        print(f"  Any negative?  {bool((flat < 0).any())}")

        # Run 2: camera
        print(f"\n{'='*60}")
        print(f"📷 Camera frame")
        print(f"{'='*60}")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera: {frame.shape}")
                inp = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(inp, axis=0)  # keep uint8

                cam_results = pipeline.infer({input_info[0].name: inp})
                cam_raw = cam_results[out_name]
                cam_flat = np.array(cam_raw)
                while cam_flat.ndim > 2:
                    cam_flat = cam_flat[0] if cam_flat.shape[0] == 1 else cam_flat.reshape(-1, cam_flat.shape[-1])

                print(f"  Output shape: {cam_raw.shape}")
                print(f"  Flat: {cam_flat.shape}")

                print(f"\n  Per-col stats:")
                print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
                for col in range(cam_flat.shape[-1]):
                    c = cam_flat[:, col]
                    print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")

                # Top 10 anchors
                max_vals = cam_flat.max(axis=-1)
                top10 = max_vals.argsort()[-10:][::-1]
                print(f"\n  Top 10 anchors by max value:")
                for idx in top10:
                    print(f"  [{idx:4d}]: max={max_vals[idx]:.4f}  vals={cam_flat[idx]}")

                # Top 5 per column
                D2 = cam_flat.shape[-1]
                print(f"\n  Top 5 per column:")
                for col in range(D2):
                    top5 = cam_flat[:, col].argsort()[-5:][::-1]
                    vals = " ".join(f"[{i}:{cam_flat[i,col]:.3f}]" for i in top5)
                    print(f"  col {col}: {vals}")
            else:
                print("  ❌ Camera read failed")
            cap.release()
        else:
            print("  ⚠️ No camera")

print(f"\n{'='*60}")
print("✅ Debug complete! Send this to Caddie.")
print("=" * 60)