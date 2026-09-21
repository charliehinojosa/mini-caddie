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

hn = runner._get_hn()
print("\n=== ALL LAYERS ===")
for layer in hn.layers:
    print(f"  {layer.name}  shape={getattr(layer, 'shape', '?')}  type={getattr(layer, 'type', '?')}")

print("\n=== OUTPUT LAYERS ===")
for out in hn.outputs:
    print(f"  {out.name}  shape={getattr(out, 'shape', '?')}")

print("\n=== INPUT LAYERS ===")
for inp in hn.inputs:
    print(f"  {inp.name}  shape={getattr(inp, 'shape', '?')}")