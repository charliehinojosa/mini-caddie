#!/usr/bin/env python3
"""
Compile script v3 — WITH NMS POSTPROCESS baked into HEF.
Run with: /content/dfc_env/bin/python /content/compile_hailo_v3.py

After compile, hailo-detect --hef-path works directly — no Pi-side post-processing.
"""
import os
os.environ['USER'] = 'colab'

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner
import glob, shutil, json

NUM_CLASSES = 8
SCORES_TH = 0.2
IOU_TH = 0.7
MAX_PROPOSALS = 100
REGRESSION_LENGTH = 16

# --- 1. Copy ONNX ---
shutil.copy('/content/drive/MyDrive/mini_caddie_golf_best.onnx', '/content/mini_caddie_golf_best.onnx')
print(f"ONNX copied: {os.path.getsize('/content/mini_caddie_golf_best.onnx')} bytes")

# --- 2. Create runner and translate ONNX ---
runner = ClientRunner(hw_arch='hailo8l')

runner.translate_onnx_model(
    '/content/mini_caddie_golf_best.onnx',
    'mini_caddie_golf',
    start_node_names=['images'],
    end_node_names=['/model.22/Concat_1'],
    net_input_shapes={'images': [1, 3, 640, 640]}
)
print("ONNX translated!")

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
print("NMS config written!")

# --- 4. Create and load Hailo model script (.alls) ---
alls_script = (
    'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'
    'nms_postprocess("' + nms_config_path + '", meta_arch=yolov8, engine=cpu)\n'
    'allocator_param(width_splitter_defuse=disabled, spatial_defuse_legacy=True)\n'
)

alls_path = '/content/mini_caddie_golf.alls'
with open(alls_path, 'w') as f:
    f.write(alls_script)
print("Model script written!")

runner.load_model_script(alls_path)
print("Model script loaded!")

# --- 5. Load calibration images ---
calib_dir = '/content/dataset/unified/valid/images'
calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))[:100]
print(f"Found {len(calib_files)} calibration images")

def preprocess(file):
    img = Image.open(file).resize((640, 640))
    return np.array(img, dtype=np.float32) / 255.0

calib_data = np.array([preprocess(f) for f in calib_files])
print(f"Calibration shape: {calib_data.shape}")

# --- 6. Optimize (quantize) — ~17 min ---
runner.optimize(calib_data)
print("Optimization complete!")

# --- 7. Compile to HEF ---
hef = runner.compile()
print("Compile complete!")

# --- 8. Save HEF ---
hef_output = '/content/mini_caddie_golf_nms.hef'
with open(hef_output, 'wb') as f:
    f.write(hef)
print(f"HEF saved! Size: {os.path.getsize(hef_output)} bytes")

# --- 9. Copy to Drive ---
shutil.copy(hef_output, '/content/drive/MyDrive/mini_caddie_golf_nms.hef')
shutil.copy(nms_config_path, '/content/drive/MyDrive/mini_caddie_nms_config.json')
print("Copied to Drive!")