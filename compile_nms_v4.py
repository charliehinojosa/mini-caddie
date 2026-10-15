import os
os.environ['USER'] = 'colab'
import json
import numpy as np
from PIL import Image
import glob
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch='hailo8l')
runner.translate_onnx_model('/content/drive/MyDrive/mini_caddie_golf_best.onnx', 'mini_caddie_golf', start_node_names=['images'], end_node_names=['/model.22/cv2.0/cv2.0.2/Conv', '/model.22/cv3.0/cv3.0.2/Conv', '/model.22/cv2.1/cv2.1.2/Conv', '/model.22/cv3.1/cv3.1.2/Conv', '/model.22/cv2.2/cv2.2.2/Conv', '/model.22/cv3.2/cv3.2.2/Conv'], net_input_shapes={'images': [1, 3, 640, 640]})

# Lower score threshold to 0.001 to see if ANYTHING passes
nms_config = {"nms_scores_th": 0.001, "nms_iou_th": 0.7, "image_dims": [640, 640], "max_proposals_per_class": 100, "classes": 8, "regression_length": 16, "background_removal": False, "bbox_decoders": [{"name": "mini_caddie_golf/bbox_decoder41", "stride": 8, "reg_layer": "mini_caddie_golf/conv41", "cls_layer": "mini_caddie_golf/conv42"}, {"name": "mini_caddie_golf/bbox_decoder52", "stride": 16, "reg_layer": "mini_caddie_golf/conv52", "cls_layer": "mini_caddie_golf/conv53"}, {"name": "mini_caddie_golf/bbox_decoder62", "stride": 32, "reg_layer": "mini_caddie_golf/conv62", "cls_layer": "mini_caddie_golf/conv63"}]}

nms_config_path = '/content/mini_caddie_nms_config.json'
with open(nms_config_path, 'w') as f:
    json.dump(nms_config, f, indent=4)

# NO change_output_activation — let nms_postprocess handle sigmoid
alls = 'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\nnms_postprocess("' + nms_config_path + '", meta_arch=yolov8, engine=cpu)\n'
print("=== .alls script ===")
print(alls)
runner.load_model_script(alls)
print("Script loaded!")

# Find calibration images
calib_dirs = ['/content/dataset/unified/valid/images', '/content/dataset/unified/images/val', '/content/dataset/valid/images', '/content/dataset/images']
calib_dir = None
for d in calib_dirs:
    if os.path.isdir(d):
        calib_dir = d
        break

if calib_dir is None:
    import subprocess
    result = subprocess.run(['find', '/content/dataset', '-name', '*.jpg', '-print0'], capture_output=True, text=True)
    if result.stdout:
        all_imgs = result.stdout.strip('\0').split('\0')
        if all_imgs and all_imgs[0]:
            calib_dir = os.path.dirname(all_imgs[0])

if calib_dir is None:
    print("ERROR: No calibration images found!")
    exit(1)

calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')))[:100]
print("Calibration dir: " + str(calib_dir))
print("Found " + str(len(calib_files)) + " calibration images")

def preprocess(file):
    img = Image.open(file).resize((640, 640))
    return np.array(img, dtype=np.float32) / 255.0

calib_data = np.array([preprocess(f) for f in calib_files])
print("Calibration shape: " + str(calib_data.shape))

print("")
print("=== Starting optimize (quantize) — ~17 min ===")
runner.optimize(calib_data)
print("Optimization complete!")

# Check activations AFTER optimize
hn = runner.get_hn_model()
print("")
print("=== Conv activations AFTER optimize ===")
for name in ['mini_caddie_golf/conv42', 'mini_caddie_golf/conv53', 'mini_caddie_golf/conv63']:
    layer = hn.get_layer_by_name(name)
    act = getattr(layer, 'activation', 'NO ATTR')
    print("  " + name + ": activation=" + str(act))

# SDKPaths fix
try:
    from hailo_sdk_common.paths_manager.paths import SDKPaths
    paths = SDKPaths()
    paths._is_release = True
    print("SDKPaths fix applied!")
except Exception as e:
    print("SDKPaths fix skipped: " + str(e))

print("")
print("=== Compiling HEF ===")
hef = runner.compile()
print("Compile complete!")

hef_output = '/content/mini_caddie_golf_nms_v4.hef'
with open(hef_output, 'wb') as f:
    f.write(hef)
print("HEF saved! Size: " + str(os.path.getsize(hef_output)) + " bytes")

import shutil
shutil.copy(hef_output, '/content/drive/MyDrive/mini_caddie_golf_nms_v4.hef')
shutil.copy(nms_config_path, '/content/drive/MyDrive/mini_caddie_nms_config_v4.json')
print("Copied to Drive!")
print("DONE!")