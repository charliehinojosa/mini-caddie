#!/usr/bin/env python3
"""
Compile script v3 — YOLOv8 → Hailo HEF with on-chip NMS.

This is v2 + NMS postprocess config so hailo-detect works without Pi-side NMS.
Run with: /content/dfc_env/bin/python /content/compile_hailo_v3.py

Key change from v2:
  - runner.load_model_script() with NMS config before compile()
  - end_node_names changed to include the detection head outputs for NMS input
"""
import os
os.environ['USER'] = 'colab'

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner
import glob, shutil

# --- 1. Copy fresh ONNX (re-exported with simplify=True) ---
shutil.copy('/content/drive/MyDrive/mini_caddie_golf_best.onnx',
            '/content/mini_caddie_golf_best.onnx')
print(f"ONNX copied: {os.path.getsize('/content/mini_caddie_golf_best.onnx')} bytes")

# --- 2. Create runner and translate ONNX ---
runner = ClientRunner(hw_arch='hailo8l')

# For NMS, we need to cut at the detection head outputs BEFORE DFL
# But DFL is unsupported by the parser. With simplify=True, the graph
# may have different node names. Let's try cutting at Concat_1 first
# and adding NMS via model script.
runner.translate_onnx_model(
    '/content/mini_caddie_golf_best.onnx',
    'mini_caddie_golf',
    start_node_names=['images'],
    end_node_names=['/model.22/Concat_1'],
    net_input_shapes={'images': [1, 3, 640, 640]}
)
print("✅ ONNX translated!")

# --- 3. Load NMS model script ---
# The Hailo NMS postprocess config tells the compiler to add an NMS op
# after the detection head output. This makes the HEF output ready-to-use
# detections that hailo-detect can read directly.
#
# For YOLOv8 with 8 classes (excluding background), the NMS script needs:
# - score_threshold: minimum confidence
# - iou_threshold: NMS IoU threshold  
# - number of classes
# - bbox decoding params
#
# The model script format is Hailo's .alls format:
nms_script = """
nms_postprocess(
    score_threshold=0.25,
    iou_threshold=0.45,
    num_classes=8,
    max_proposals_per_class=100,
    max_proposals_total=300,
    bbox_decoding={
        "yolov8",
        "8400",
        {"scores_type": "sigmoid", "classes": 8, "bbox_format": "ltbr"}
    }
)
"""

# Try loading the script — if the API differs, we'll adjust
try:
    runner.load_model_script(nms_script)
    print("✅ NMS model script loaded!")
except Exception as e:
    print(f"⚠️  Script load failed: {e}")
    print("Trying alternative NMS config format...")
    
    # Alternative: use the Hailo SDK's built-in YOLOv8 NMS config
    try:
        runner.load_model_script("""
nms_postprocess(
    score_threshold=0.25,
    iou_threshold=0.45,
    num_classes=8
)
""")
        print("✅ Alternative NMS script loaded!")
    except Exception as e2:
        print(f"⚠️  Alt script also failed: {e2}")
        print("Proceeding without NMS — will need Pi-side post-processing")

# --- 4. Load calibration images ---
calib_dir = '/content/dataset/unified/valid/images'
calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))[:100]
print(f"Found {len(calib_files)} calibration images")

def preprocess(file):
    img = Image.open(file).resize((640, 640))
    return np.array(img, dtype=np.float32) / 255.0

calib_data = np.array([preprocess(f) for f in calib_files])
print(f"Calibration shape: {calib_data.shape}")

# --- 5. Optimize (quantize) ---
runner.optimize(calib_data)
print("✅ Optimization/Quantization complete!")

# --- 6. Compile to HEF ---
hef = runner.compile()
print("✅ Compile complete!")

# --- 7. Save HEF ---
hef_path = '/content/mini_caddie_golf_nms.hef'
with open(hef_path, 'wb') as f:
    f.write(hef)
print(f"✅ HEF saved! Size: {os.path.getsize(hef_path)} bytes")

# --- 8. Copy to Drive ---
shutil.copy(hef_path, '/content/drive/MyDrive/mini_caddie_golf_nms.hef')
print("✅ HEF copied to Drive!")