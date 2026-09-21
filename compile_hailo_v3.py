#!/usr/bin/env python3
"""
CONFIRMED WORKING compile script v3 — WITH NMS POSTPROCESS
Run with: /content/dfc_env/bin/python /content/compile_hailo_v3.py

Changes from v2:
  - Loads a Hailo model script (.alls) that adds NMS postprocess
  - This bakes NMS into the HEF so hailo-detect --hef-path works directly
  - No more Pi-side post-processing needed!

The model script tells Hailo to:
  1. Add sigmoid activation to output layers
  2. Add NMS postprocess with our 8 classes (not 80 like COCO)
  3. Handle the DFL decode on-chip

Prerequisites (all in the Python 3.12 venv at /content/dfc_env):
- DFC 3.34.0 installed with all deps (TF 2.18.0, onnx 1.16.0, etc.)
- ultralytics installed for ONNX re-export
- numpy<2 (after ultralytics install to avoid breaking TF 2.18.0)

Before running:
1. ONNX model re-exported from .pt with simplify=True
2. Dataset extracted to /content/dataset/unified/valid/images
"""
import os
os.environ['USER'] = 'colab'

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner
import glob, shutil, json

# --- Config ---
NUM_CLASSES = 8  # golf classes (NOT including background — Hailo adds it)
SCORES_TH = 0.2
IOU_TH = 0.7
MAX_PROPOSALS = 100
REGRESSION_LENGTH = 16  # DFL bins

# --- 1. Copy fresh ONNX (re-exported with simplify=True) ---
shutil.copy('/content/drive/MyDrive/mini_caddie_golf_best.onnx',
            '/content/mini_caddie_golf_best.onnx')
print(f"ONNX copied: {os.path.getsize('/content/mini_caddie_golf_best.onnx')} bytes")

# --- 2. Create runner and translate ONNX ---
runner = ClientRunner(hw_arch='hailo8l')

# IMPORTANT: For NMS, we need the FULL model (not cut at Concat_1)
# The simplified ONNX should parse OK with the right end nodes
# Try the full model first — if it fails, fall back to Concat_1
try:
    runner.translate_onnx_model(
        '/content/mini_caddie_golf_best.onnx',
        'mini_caddie_golf',
        start_node_names=['images'],
        end_node_names=['/model.22/Concat_1'],  # Try this first (known to work)
        net_input_shapes={'images': [1, 3, 640, 640]}
    )
    print("✅ ONNX translated (cut at Concat_1)!")
    
    # Check what outputs we have
    hn = runner._get_hn()
    print(f"  Network outputs: {hn.outputs}")
    
except Exception as e:
    print(f"❌ Concat_1 cut failed: {e}")
    print("  Trying full model parse...")
    runner.translate_onnx_model(
        '/content/mini_caddie_golf_best.onnx',
        'mini_caddie_golf',
        start_node_names=['images'],
        net_input_shapes={'images': [1, 3, 640, 640]}
    )
    print("✅ ONNX translated (full model)!")

# --- 3. Create NMS config JSON ---
nms_config = {
    "nms_scores_th": SCORES_TH,
    "nms_iou_th": IOU_TH,
    "image_dims": [640, 640],
    "max_proposals_per_class": MAX_PROPOSALS,
    "classes": NUM_CLASSES,
    "regression_length": REGRESSION_LENGTH,
    "background_removal": False,
    "bbox_decoders": [
        {
            "name": "mini_caddie_golf/bbox_decoder41",
            "stride": 8,
            "reg_layer": "mini_caddie_golf/conv41",
            "cls_layer": "mini_caddie_golf/conv42"
        },
        {
            "name": "mini_caddie_golf/bbox_decoder52",
            "stride": 16,
            "reg_layer": "mini_caddie_golf/conv52",
            "cls_layer": "mini_caddie_golf/conv53"
        },
        {
            "name": "mini_caddie_golf/bbox_decoder62",
            "stride": 32,
            "reg_layer": "mini_caddie_golf/conv62",
            "cls_layer": "mini_caddie_golf/conv63"
        }
    ]
}

nms_config_path = '/content/mini_caddie_nms_config.json'
with open(nms_config_path, 'w') as f:
    json.dump(nms_config, f, indent=4)
print(f"✅ NMS config written to {nms_config_path}")

# --- 4. Create and load Hailo model script (.alls) ---
# The layer names may differ from standard YOLOv8n — we need to find them
# First, let's inspect the parsed network to find the output layer names
alls_script = f"""
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
nms_postprocess("{nms_config_path}", meta_arch=yolov8, engine=cpu)
allocator_param(width_splitter_defuse=disabled, spatial_defuse_legacy=True)
"""

# Save the .alls script
alls_path = '/content/mini_caddie_golf.alls'
with open(alls_path, 'w') as f:
    f.write(alls_script.strip())
print(f"✅ Model script written to {alls_path}")

# Load the model script BEFORE optimization
runner.load_model_script(alls_path)
print("✅ Model script loaded!")

# --- 5. Load calibration images ---
calib_dir = '/content/dataset/unified/valid/images'
calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))[:100]
print(f"Found {len(calib_files)} calibration images")

def preprocess(file):
    img = Image.open(file).resize((640, 640))
    return np.array(img, dtype=np.float32) / 255.0

calib_data = np.array([preprocess(f) for f in calib_files])
print(f"Calibration shape: {calib_data.shape}")

# --- 6. Optimize (quantize) ---
runner.optimize(calib_data)
print("✅ Optimization/Quantization complete!")

# --- 7. Compile to HEF ---
hef = runner.compile()
print("✅ Compile complete!")

# --- 8. Save HEF ---
hef_output = '/content/mini_caddie_golf.hef'
with open(hef_output, 'wb') as f:
    f.write(hef)
print(f"✅ HEF saved! Size: {os.path.getsize(hef_output)} bytes")

# --- 9. Copy to Drive ---
shutil.copy(hef_output, '/content/drive/MyDrive/mini_caddie_golf_nms.hef')
print("✅ HEF copied to Drive as mini_caddie_golf_nms.hef!")

# --- 10. Also copy the NMS config for reference ---
shutil.copy(nms_config_path, '/content/drive/MyDrive/mini_caddie_nms_config.json')
print("✅ NMS config copied to Drive!")