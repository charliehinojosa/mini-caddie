"""Find actual layer names in the parsed Hailo network."""
import os
os.environ['USER'] = 'colab'

import shutil
from hailo_sdk_client import ClientRunner

shutil.copy('/content/drive/MyDrive/mini_caddie_golf_best.onnx', '/content/mini_caddie_golf_best.onnx')

runner = ClientRunner(hw_arch='hailo8l')
runner.translate_onnx_model(
    '/content/mini_caddie_golf_best.onnx',
    'mini_caddie_golf',
    start_node_names=['images'],
    end_node_names=['/model.22/Concat_1'],
    net_input_shapes={'images': [1, 3, 640, 640]}
)
print("ONNX translated!")

hn = runner.get_hn()
print("\n=== HN type ===")
print(type(hn))
print("\n=== HN attrs ===")
print([a for a in dir(hn) if not a.startswith('_')])

print("\n=== ALL LAYERS ===")
if hasattr(hn, 'layers'):
    for layer in hn.layers:
        print(f"  {layer.name}  shape={getattr(layer, 'shape', '?')}")
elif isinstance(hn, dict):
    for name, layer in hn.items():
        print(f"  {name}  type={type(layer).__name__}  attrs={[a for a in dir(layer) if not a.startswith('_')][:10]}")