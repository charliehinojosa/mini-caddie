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
    net_input_shapes={'images': [1, 3, 640, 640]}
)
print("ONNX translated!")

hn = runner.get_hn()
print("\n=== HN keys ===")
print(list(hn.keys()))

layers = hn.get('layers', hn)
lines = []
lines.append(f"=== LAYERS type: {type(layers).__name__} ===")
if isinstance(layers, dict):
    for name, layer in layers.items():
        lines.append(f"  {name}  type={type(layer).__name__}  attrs={[a for a in dir(layer) if not a.startswith('_')][:8]}")
elif hasattr(layers, '__iter__'):
    for layer in layers:
        lines.append(f"  {getattr(layer, 'name', '?')}  type={type(layer).__name__}")

output = "\n".join(lines)
print(output)
with open('/content/drive/MyDrive/layer_names.txt', 'w') as f:
    f.write(output)
print("\nSaved to Drive as layer_names.txt!")