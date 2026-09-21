#!/usr/bin/env python3
"""
Mini Caddie — Golf Inference DEBUG (WORKING API)
Approach 1: make_from_network_group(ng) ✅
InferVStreams order: (ng, input_params, output_params, target) — ng has .name
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

# Set output format to FLOAT32 — output is a DICT, use .values()
# user_buffer_format needs HailoFormat, not FormatType
try:
    from hailo_platform import HailoFormat
    for p in out_params.values():
        p.user_buffer_format = HailoFormat(
            FormatType.FLOAT32
        )
except ImportError:
    # Skip — use default format
    print("  (HailoFormat not available, using default output format)")

# Test image
test_img = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
for y in range(INPUT_SIZE):
    for x in range(INPUT_SIZE):
        test_img[0, y, x, 0] = x / INPUT_SIZE
        test_img[0, y, x, 1] = y / INPUT_SIZE
        test_img[0, y, x, 2] = 0.5

print(f"\n🖼️ Test input: {test_img.shape}")

# Try different InferVStreams argument orders
orders = [
    ("ng, inp, out, target", lambda: InferVStreams(ng, inp_params, out_params, target)),
    ("ng, inp, out, target, ng", lambda: InferVStreams(ng, inp_params, out_params, target, ng)),
    ("target, ng, inp, out", lambda: InferVStreams(target, ng, inp_params, out_params)),
]

activated = ng.activate()
with activated:
    for label, factory in orders:
        print(f"\n--- Trying InferVStreams({label}) ---")
        try:
            with factory() as pipeline:
                input_dict = {input_info[0].name: test_img}
                results = pipeline.infer(input_dict)
                out_name = output_info[0].name
                raw = results[out_name]

                flat = raw[0] if raw.ndim == 3 else raw
                N, D = flat.shape

                print(f"✅✅✅ INFERENCE WORKED!")
                print(f"   Output shape: {raw.shape}, dtype: {raw.dtype}")
                print(f"   Flat: ({N}, {D})")
                print(f"\n   Per-col min: {flat.min(axis=0)}")
                print(f"   Per-col max: {flat.max(axis=0)}")
                print(f"   Per-col mean: {flat.mean(axis=0)}")
                print(f"   Per-col std: {flat.std(axis=0)}")
                print(f"\n   Sample[0]: {flat[0]}")
                print(f"   Sample[100]: {flat[100]}")
                print(f"   Sample[4000]: {flat[4000]}")
                print(f"   Sample[8399]: {flat[8399]}")

                # Camera frame
                print(f"\n📷 Camera frame...")
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        img_h, img_w = frame.shape[:2]
                        inp = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                        inp = inp.astype(np.float32) / 255.0
                        inp = np.expand_dims(inp, axis=0)

                        cam_results = pipeline.infer({input_info[0].name: inp})
                        cam_raw = cam_results[out_name]
                        cam_flat = cam_raw[0] if cam_raw.ndim == 3 else cam_raw

                        print(f"   Camera output: {cam_raw.shape}")
                        print(f"\n   Per-col min: {cam_flat.min(axis=0)}")
                        print(f"   Per-col max: {cam_flat.max(axis=0)}")
                        print(f"   Per-col mean: {cam_flat.mean(axis=0)}")
                        print(f"   Per-col std: {cam_flat.std(axis=0)}")

                        # Top 10 anchors
                        max_vals = cam_flat.max(axis=-1)
                        top10 = max_vals.argsort()[-10:][::-1]
                        print(f"\n   Top 10 anchors:")
                        for idx in top10:
                            print(f"   [{idx:4d}]: max={max_vals[idx]:.4f}  vals={cam_flat[idx]}")

                        # Top 5 per column
                        print(f"\n   Top 5 per column:")
                        for col in range(D):
                            top5 = cam_flat[:, col].argsort()[-5:][::-1]
                            vals = " ".join(f"[{i}:{cam_flat[i,col]:.3f}]" for i in top5)
                            print(f"   col {col}: {vals}")
                    else:
                        print("   ❌ Camera read failed")
                    cap.release()
                else:
                    print("   ⚠️ No camera")

                print(f"\n✅ Debug complete! Send this output to Caddie.")
                break  # Don't try other orders if one works
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue