#!/usr/bin/env python3
"""
Find the correct layer names for NMS bbox_decoders.
Translates ONNX with the 6 conv outputs, then prints all layers
with their output shapes so we can identify reg vs cls per stride.
"""
import os
os.environ['USER'] = 'colab'

from hailo_sdk_client import ClientRunner
import json

runner = ClientRunner(hw_arch='hailo8l')

runner.translate_onnx_model(
    '/content/drive/MyDrive/mini_caddie_golf_best.onnx',
    'mini_caddie_golf',
    start_node_names=['images'],
    end_node_names=[
        '/model.22/cv2.0/cv2.0.2/Conv',
        '/model.22/cv3.0/cv3.0.2/Conv',
        '/model.22/cv2.1/cv2.1.2/Conv',
        '/model.22/cv3.1/cv3.1.2/Conv',
        '/model.22/cv2.2/cv2.2.2/Conv',
        '/model.22/cv3.2/cv3.2.2/Conv',
    ],
    net_input_shapes={'images': [1, 3, 640, 640]}
)
print("Translated!\n")

# Get the parsed Hailo Network
hn = runner._hn

# Print all layers with their shapes
print("=== ALL LAYERS ===")
layers = hn.get_layers()
for layer in layers:
    name = layer
    # Try to get shape info
    try:
        layer_obj = layers[layer]
        shape = layer_obj.get('shape', 'unknown')
        print(f"  {name}  shape={shape}")
    except:
        print(f"  {name}")

# Also print just the output layers
print("\n=== OUTPUT LAYERS ===")
try:
    outputs = hn.get_output_layers()
    for o in outputs:
        print(f"  {o}")
except:
    pass

# Try to save the hn to inspect
print("\n=== SAVING .hn FILE ===")
runner.save_model('/content/mini_caddie_golf.hn')
print("Saved! Checking file...")
print(f"File size: {os.path.getsize('/content/mini_caddie_golf.hn')} bytes")

# Read and print the output layer section
with open('/content/mini_caddie_golf.hn', 'r') as f:
    hn_data = json.load(f)

print("\n=== .hn OUTPUT LAYERS ===")
for layer in hn_data.get('layers', []):
    if layer.get('is_output', False) or 'output' in layer.get('name', '').lower():
        print(f"  name: {layer.get('name')}")
        print(f"  shape: {layer.get('shape')}")
        print()

print("\n=== .hn ALL LAYER NAMES + SHAPES ===")
for layer in hn_data.get('layers', []):
    name = layer.get('name', '???')
    shape = layer.get('shape', '???')
    is_output = layer.get('is_output', False)
    marker = " [OUTPUT]" if is_output else ""
    print(f"  {name:<45} shape={shape}{marker}")