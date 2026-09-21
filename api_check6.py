#!/usr/bin/env python3
"""
Check ALL output stream infos + quant_info + try sigmoid on the 8 values.
Also check if there are multiple output layers.
"""
import os
import numpy as np
import cv2
from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

hef = HEF(os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"))
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng = target.configure(hef, configure_params)[0]

# Check ALL output stream infos (not just vstream)
print("📊 ALL output stream infos:")
output_streams = hef.get_output_stream_infos()
for o in output_streams:
    print(f"  {o.name}: shape={o.shape}")
    for attr in sorted(dir(o)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(o, attr)
            if not callable(val):
                print(f"    {attr} = {val}")
        except:
            print(f"    {attr} = <error>")

print(f"\n📊 ALL output vstream infos:")
output_vstreams = hef.get_output_vstream_infos()
for o in output_vstreams:
    print(f"  {o.name}: shape={o.shape}")
    # Check quant_info
    qi = o.quant_info
    print(f"    quant_info attrs: {[a for a in dir(qi) if not a.startswith('_')]}")
    for attr in dir(qi):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(qi, attr)
            if not callable(val):
                print(f"    quant_info.{attr} = {val}")
        except:
            print(f"    quant_info.{attr} = <error>")

# Check get_output_shapes
print(f"\n📊 get_output_shapes():")
try:
    shapes = ng.get_output_shapes()
    print(f"  {shapes}")
except Exception as e:
    print(f"  Error: {e}")

# Now run inference with camera + apply sigmoid
print("\n🧠 Running inference with camera...")
input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()

inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)

# Set FLOAT32 on output
for p in out_params.values():
    p.user_buffer_format.type = FormatType.FLOAT32

activated = ng.activate()
with activated:
    with InferVStreams(ng, inp_params, out_params, target) as pipeline:
        # Try camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera: {frame.shape}")
                inp = cv2.resize(frame, (640, 640))
                inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(inp, axis=0)
                
                results = pipeline.infer({input_info[0].name: inp})
                out_name = output_info[0].name
                raw = results[out_name]
                flat = np.array(raw)
                while flat.ndim > 2:
                    flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
                
                print(f"  Output: {flat.shape}, dtype: {flat.dtype}")
                
                # Apply sigmoid
                sigmoid = 1.0 / (1.0 + np.exp(-flat))
                print(f"\n  After sigmoid:")
                print(f"  Per-col stats:")
                for col in range(sigmoid.shape[-1]):
                    c = sigmoid[:, col]
                    print(f"  col {col}: min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f}")
                
                # Top 10 anchors by max sigmoid score
                max_scores = sigmoid.max(axis=-1)
                top10 = max_scores.argsort()[-10:][::-1]
                print(f"\n  Top 10 anchors (after sigmoid):")
                for idx in top10:
                    print(f"  [{idx:4d}]: max={max_scores[idx]:.4f}  vals={sigmoid[idx]}")
                
                # Top 5 per column
                print(f"\n  Top 5 per column (after sigmoid):")
                for col in range(sigmoid.shape[-1]):
                    top5 = sigmoid[:, col].argsort()[-5:][::-1]
                    vals = " ".join(f"[{i}:{sigmoid[i,col]:.4f}]" for i in top5)
                    print(f"  col {col}: {vals}")
            else:
                print("  ❌ Camera read failed — using test image")
                # Fall back to test image
                test = np.zeros((1, 640, 640, 3), dtype=np.uint8)
                results = pipeline.infer({input_info[0].name: test})
                raw = results[out_name]
                flat = np.array(raw)
                while flat.ndim > 2:
                    flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
                
                sigmoid = 1.0 / (1.0 + np.exp(-flat))
                print(f"\n  Test image after sigmoid:")
                for col in range(sigmoid.shape[-1]):
                    c = sigmoid[:, col]
                    print(f"  col {col}: min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f}")
                
                max_scores = sigmoid.max(axis=-1)
                top10 = max_scores.argsort()[-10:][::-1]
                print(f"\n  Top 10 anchors:")
                for idx in top10:
                    print(f"  [{idx:4d}]: max={max_scores[idx]:.4f}  vals={sigmoid[idx]}")
            cap.release()
        else:
            print("  ⚠️ No camera — using test image")
            test = np.zeros((1, 640, 640, 3), dtype=np.uint8)
            results = pipeline.infer({input_info[0].name: test})
            raw = results[out_name]
            flat = np.array(raw)
            while flat.ndim > 2:
                flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
            
            sigmoid = 1.0 / (1.0 + np.exp(-flat))
            print(f"\n  Test image after sigmoid:")
            for col in range(sigmoid.shape[-1]):
                c = sigmoid[:, col]
                print(f"  col {col}: min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f}")
            
            max_scores = sigmoid.max(axis=-1)
            top10 = max_scores.argsort()[-10:][::-1]
            print(f"\n  Top 10 anchors:")
            for idx in top10:
                print(f"  [{idx:4d}]: max={max_scores[idx]:.4f}  vals={sigmoid[idx]}")

print("\n✅ Done!")