import os
os.environ['USER'] = 'colab'
import json
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch='hailo8l')
runner.translate_onnx_model('/content/drive/MyDrive/mini_caddie_golf_best.onnx', 'mini_caddie_golf', start_node_names=['images'], end_node_names=['/model.22/cv2.0/cv2.0.2/Conv', '/model.22/cv3.0/cv3.0.2/Conv', '/model.22/cv2.1/cv2.1.2/Conv', '/model.22/cv3.1/cv3.1.2/Conv', '/model.22/cv2.2/cv2.2.2/Conv', '/model.22/cv3.2/cv3.2.2/Conv'], net_input_shapes={'images': [1, 3, 640, 640]})

nms_config = {"nms_scores_th": 0.2, "nms_iou_th": 0.7, "image_dims": [640, 640], "max_proposals_per_class": 100, "classes": 8, "regression_length": 16, "background_removal": False, "bbox_decoders": [{"name": "mini_caddie_golf/bbox_decoder41", "stride": 8, "reg_layer": "mini_caddie_golf/conv41", "cls_layer": "mini_caddie_golf/conv42"}, {"name": "mini_caddie_golf/bbox_decoder52", "stride": 16, "reg_layer": "mini_caddie_golf/conv52", "cls_layer": "mini_caddie_golf/conv53"}, {"name": "mini_caddie_golf/bbox_decoder62", "stride": 32, "reg_layer": "mini_caddie_golf/conv62", "cls_layer": "mini_caddie_golf/conv63"}]}

nms_config_path = '/content/mini_caddie_nms_config.json'
with open(nms_config_path, 'w') as f:
    json.dump(nms_config, f, indent=4)

alls = 'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\nnms_postprocess("' + nms_config_path + '", meta_arch=yolov8, engine=cpu)\n'
print("=== Loading .alls ===")
print(alls)
runner.load_model_script(alls)

print("(skipping get_detected_nms_config - not supported for yolov8)")

hn = runner.get_hn_model()
print("")
print("=== Output layer activations after NMS ===")
for ol in hn.get_real_output_layers():
    act = getattr(ol, 'activation', 'NO ATTR')
    print("  " + ol.name + ": activation=" + str(act))

print("")
print("=== Conv layer activations after NMS ===")
for name in ['mini_caddie_golf/conv42', 'mini_caddie_golf/conv53', 'mini_caddie_golf/conv63']:
    layer = hn.get_layer_by_name(name)
    act = getattr(layer, 'activation', 'NO ATTR')
    print("  " + name + ": activation=" + str(act))