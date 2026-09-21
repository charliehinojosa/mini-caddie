#!/usr/bin/env python3
"""Quick sigmoid check — save results to file."""
import os, numpy as np, cv2
from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

LABELS = ["background", "golf_ball", "golf_club", "golf_club_head",
          "golf_hole", "golf_mat", "person", "player_not_ready", "player_ready"]

hef = HEF(os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"))
target = VDevice()
ng = target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()
out_name = output_info[0].name

inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)
for p in out_params.values():
    p.user_buffer_format.type = FormatType.FLOAT32

lines = []

def log(msg):
    print(msg)
    lines.append(msg)

def run(pipeline, img, label):
    inp = np.expand_dims(img, axis=0) if img.ndim == 3 else img
    raw = pipeline.infer({input_info[0].name: inp})[out_name]
    flat = np.array(raw)
    while flat.ndim > 2:
        flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
    sig = 1.0 / (1.0 + np.exp(-flat))
    
    log(f"\n=== {label} ===")
    log(f"  {'col':>4} {'min':>8} {'max':>8} {'mean':>8} {'label'}")
    for col in range(sig.shape[-1]):
        c = sig[:, col]
        lbl = LABELS[col + 1] if col + 1 < len(LABELS) else f"class_{col}"
        log(f"  {col:4d} {c.min():8.4f} {c.max():8.4f} {c.mean():8.4f} {lbl}")
    
    # Top 5 anchors
    mx = sig.max(axis=-1)
    top5 = mx.argsort()[-5:][::-1]
    log(f"  Top 5: ")
    for idx in top5:
        cls = sig[idx].argmax()
        log(f"    [{idx}] {LABELS[cls+1]} {mx[idx]:.4f}")
    
    # Check if cols 0-3 differ from 4-7 (bbox vs classes?)
    first4 = sig[:, :4]
    last4 = sig[:, 4:]
    log(f"  cols 0-3 max: {first4.max():.4f}, cols 4-7 max: {last4.max():.4f}")
    
    return sig

activated = ng.activate()
with activated:
    with InferVStreams(ng, inp_params, out_params, target) as pipeline:
        log("MINI CADDIE — SIGMOID ANALYSIS")
        log("=" * 50)
        
        run(pipeline, np.zeros((640, 640, 3), dtype=np.uint8), "BLACK")
        run(pipeline, np.ones((640, 640, 3), dtype=np.uint8) * 255, "WHITE")
        run(pipeline, np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8), "NOISE")
        
        # Camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    inp = cv2.resize(frame, (640, 640))
                    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                    run(pipeline, inp, "CAMERA")
                    break
            else:
                log("CAMERA: failed")
            cap.release()
        else:
            log("CAMERA: not available")

# Save to file
out_path = os.path.expanduser("~/mini-caddie/debug_output.txt")
with open(out_path, 'w') as f:
    f.write('\n'.join(lines))
log(f"\nSaved to {out_path}")
log("Run: cat ~/mini-caddie/debug_output.txt")