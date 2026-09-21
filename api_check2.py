#!/usr/bin/env python3
"""Try different approaches to create vstream params."""
import os
import numpy as np
from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

hef = HEF(os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"))
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
ng = network_groups[0]

input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()
print(f"Inputs: {[(i.name, i.shape) for i in input_info]}")
print(f"Outputs: {[(o.name, o.shape) for o in output_info]}")

# Try Approach 1: make_from_network_group with ConfiguredNetwork
print("\n--- Approach 1: make_from_network_group(ng) ---")
try:
    inp = InputVStreamParams.make_from_network_group(ng)
    out = OutputVStreamParams.make_from_network_group(ng)
    print(f"✅ Success! inp type: {type(inp)}, len: {len(inp)}")
    print(f"   out type: {type(out)}, len: {len(out)}")
    if isinstance(out, dict):
        for k, v in out.items():
            print(f"   out[{k}]: {type(v)}")
            print(f"   attrs: {[a for a in dir(v) if not a.startswith('_')]}")
    elif isinstance(out, list):
        for i, v in enumerate(out):
            print(f"   out[{i}]: {type(v)}")
            print(f"   attrs: {[a for a in dir(v) if not a.startswith('_')]}")
    APPROACH = 1
except Exception as e:
    print(f"❌ Failed: {e}")

# Try Approach 2: make with create_params()
print("\n--- Approach 2: make(ng.create_params()) ---")
try:
    params = ng.create_params()
    print(f"   create_params() type: {type(params)}")
    print(f"   params attrs: {[a for a in dir(params) if not a.startswith('_')]}")
    inp2 = InputVStreamParams.make(params)
    out2 = OutputVStreamParams.make(params)
    print(f"✅ Success!")
    APPROACH = 2
except Exception as e:
    print(f"❌ Failed: {e}")

# Try Approach 3: make with ng, passing format_type explicitly
print("\n--- Approach 3: make(ng, format_type=FormatType.FLOAT32) ---")
try:
    inp3 = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    out3 = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    print(f"✅ Success!")
    APPROACH = 3
except Exception as e:
    print(f"❌ Failed: {e}")

# Try Approach 4: use _configured_network directly
print("\n--- Approach 4: make(ng._configured_network) ---")
try:
    inp4 = InputVStreamParams.make(ng._configured_network)
    out4 = OutputVStreamParams.make(ng._configured_network)
    print(f"✅ Success!")
    APPROACH = 4
except Exception as e:
    print(f"❌ Failed: {e}")

# If any approach worked, try a full inference run
print(f"\n--- Attempting inference with best approach ---")

# Re-try approach 1 with full inference
print("\n--- Full test: make_from_network_group + activate + InferVStreams ---")
try:
    inp = InputVStreamParams.make_from_network_group(ng)
    out = OutputVStreamParams.make_from_network_group(ng)
    
    # Check output params type and set format
    out_list = list(out.values()) if isinstance(out, dict) else out
    for p in out_list:
        for attr in ['format_type', 'user_buffer_format']:
            if hasattr(p, attr):
                try:
                    setattr(p, attr, FormatType.FLOAT32)
                    print(f"   Set {attr} = FLOAT32 ✅")
                    break
                except:
                    pass

    activated = ng.activate()
    with activated:
        with InferVStreams(target, ng, inp, out) as pipeline:
            test = np.zeros((1, 640, 640, 3), dtype=np.float32)
            input_dict = {input_info[0].name: test}
            results = pipeline.infer(input_dict)
            out_name = output_info[0].name
            raw = results[out_name]
            print(f"\n✅✅✅ INFERENCE WORKED!")
            print(f"   Output shape: {raw.shape}")
            print(f"   Output dtype: {raw.dtype}")
            flat = raw[0] if raw.ndim == 3 else raw
            print(f"   Flat shape: {flat.shape}")
            print(f"   Sample[0]: {flat[0]}")
            print(f"   Sample[100]: {flat[100]}")
            print(f"   Sample[4000]: {flat[4000]}")
            print(f"   Sample[8399]: {flat[8399]}")
            print(f"\n   Per-col min: {flat.min(axis=0)}")
            print(f"   Per-col max: {flat.max(axis=0)}")
            print(f"   Per-col mean: {flat.mean(axis=0)}")
            print(f"   Per-col std: {flat.std(axis=0)}")
except Exception as e:
    print(f"❌ Full test failed: {e}")
    import traceback
    traceback.print_exc()