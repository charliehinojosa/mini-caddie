#!/usr/bin/env python3
"""
Final analysis: apply sigmoid to the 8 output values, check per-column behavior.
Also try pointing camera at something to see if values change.
"""
import os
import numpy as np
import cv2
from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

HEF_PATH = os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef")
LABELS = ["background", "golf_ball", "golf_club", "golf_club_head",
          "golf_hole", "golf_mat", "person", "player_not_ready", "player_ready"]

hef = HEF(HEF_PATH)
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng = target.configure(hef, configure_params)[0]

input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()
out_name = output_info[0].name

inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)
for p in out_params.values():
    p.user_buffer_format.type = FormatType.FLOAT32

def run_inference(pipeline, img, label=""):
    """Run inference and return sigmoid'd output."""
    inp = np.expand_dims(img, axis=0) if img.ndim == 3 else img
    results = pipeline.infer({input_info[0].name: inp})
    raw = results[out_name]
    flat = np.array(raw)
    while flat.ndim > 2:
        flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
    
    sigmoid = 1.0 / (1.0 + np.exp(-flat))
    
    print(f"\n  {label} raw logits stats:")
    print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}")
    for col in range(flat.shape[-1]):
        c = flat[:, col]
        print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}")
    
    print(f"\n  {label} sigmoid stats:")
    print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'label':>20}")
    for col in range(sigmoid.shape[-1]):
        c = sigmoid[:, col]
        lbl = LABELS[col + 1] if col + 1 < len(LABELS) else f"class_{col}"
        print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {lbl:>20}")
    
    # Top 10 anchors
    max_scores = sigmoid.max(axis=-1)
    top10 = max_scores.argsort()[-10:][::-1]
    print(f"\n  {label} Top 10 anchors:")
    for idx in top10:
        cls = sigmoid[idx].argmax()
        lbl = LABELS[cls + 1] if cls + 1 < len(LABELS) else f"class_{cls}"
        print(f"  [{idx:4d}]: {lbl} score={max_scores[idx]:.4f}  all={sigmoid[idx]}")
    
    return sigmoid

activated = ng.activate()
with activated:
    with InferVStreams(ng, inp_params, out_params, target) as pipeline:
        
        # Test 1: black image
        print("="*60)
        print("TEST 1: Black image")
        print("="*60)
        black = np.zeros((640, 640, 3), dtype=np.uint8)
        run_inference(pipeline, black, "black")
        
        # Test 2: white image
        print("\n" + "="*60)
        print("TEST 2: White image")
        print("="*60)
        white = np.ones((640, 640, 3), dtype=np.uint8) * 255
        run_inference(pipeline, white, "white")
        
        # Test 3: random noise
        print("\n" + "="*60)
        print("TEST 3: Random noise")
        print("="*60)
        noise = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        run_inference(pipeline, noise, "noise")
        
        # Test 4: camera
        print("\n" + "="*60)
        print("TEST 4: Camera")
        print("="*60)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for attempt in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"  Camera frame: {frame.shape}")
                    inp = cv2.resize(frame, (640, 640))
                    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                    run_inference(pipeline, inp, f"camera_{attempt}")
                    break
            else:
                print("  ❌ All camera attempts failed")
            cap.release()
        else:
            print("  ⚠️ No camera")

print("\n✅ Done! Send this to Caddie.")