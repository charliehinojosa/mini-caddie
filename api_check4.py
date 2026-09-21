#!/usr/bin/env python3
"""Check what format-related classes are available + try to get FLOAT32 output."""
import os
import numpy as np
import hailo_platform

# List everything in hailo_platform related to format
all_attrs = [a for a in dir(hailo_platform) if 'format' in a.lower() or 'Format' in a or 'Hailo' in a]
print(f"Format/Hailo attrs in hailo_platform:")
for a in all_attrs:
    obj = getattr(hailo_platform, a)
    print(f"  {a}: {type(obj)}")

# Check FormatType values
print(f"\nFormatType values:")
for ft in hailo_platform.FormatType:
    print(f"  {ft.name} = {ft.value}")

# Check if HailoFormat exists anywhere
print(f"\nSearching for HailoFormat...")
try:
    from hailo_platform import HailoFormat
    print(f"  HailoFormat found! {HailoFormat}")
    # Check its constructor
    import inspect
    print(f"  signature: {inspect.signature(HailoFormat.__init__)}")
except ImportError:
    print("  Not in hailo_platform top-level")

# Check pyhailort submodule
try:
    from hailo_platform.pyhailort._pyhailort import HailoFormat
    print(f"  Found in _pyhailort! {HailoFormat}")
except ImportError:
    print("  Not in _pyhailort")

# Try to find it in pyhailort
try:
    import hailo_platform.pyhailort.pyhailort as pyh
    fmt_attrs = [a for a in dir(pyh) if 'format' in a.lower() or 'Format' in a or 'Hailo' in a]
    print(f"\n  pyhailort format attrs: {fmt_attrs}")
except:
    pass

# Now try the actual inference with different format approaches
print("\n" + "="*60)
print("TRYING FLOAT32 OUTPUT")
print("="*60)

from hailo_platform import (
    HEF, VDevice, InferVStreams, ConfigureParams, FormatType,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
)

hef = HEF(os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"))
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng = target.configure(hef, configure_params)[0]

input_info = hef.get_input_vstream_infos()
output_info = hef.get_output_vstream_infos()

inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)

# Try different ways to set FLOAT32
print("\nOutput params type:", type(out_params))
out_p = list(out_params.values())[0]
print(f"Output param attrs: {[a for a in dir(out_p) if not a.startswith('_')]}")
print(f"user_buffer_format type: {type(out_p.user_buffer_format)}")
print(f"user_buffer_format value: {out_p.user_buffer_format}")

# Try setting via HailoFormat if available
try:
    from hailo_platform import HailoFormat
    fmt = HailoFormat(FormatType.FLOAT32)
    out_p.user_buffer_format = fmt
    print(f"✅ Set via HailoFormat(FormatType.FLOAT32)")
except Exception as e:
    print(f"❌ HailoFormat(FormatType.FLOAT32): {e}")
    
    # Try just the enum
    try:
        out_p.user_buffer_format = FormatType.FLOAT32
        print(f"✅ Set via FormatType.FLOAT32 directly")
    except Exception as e2:
        print(f"❌ FormatType directly: {e2}")
        
        # Try float32 string
        try:
            from hailo_platform.pyhailort._pyhailort import HailoFormat as HF2
            fmt2 = HF2(FormatType.FLOAT32)
            out_p.user_buffer_format = fmt2
            print(f"✅ Set via _pyhailort.HailoFormat")
        except Exception as e3:
            print(f"❌ _pyhailort.HailoFormat: {e3}")

# Run inference
test_img = np.zeros((1, 640, 640, 3), dtype=np.uint8)

activated = ng.activate()
with activated:
    with InferVStreams(ng, inp_params, out_params, target) as pipeline:
        results = pipeline.infer({input_info[0].name: test_img})
        out_name = output_info[0].name
        raw = results[out_name]
        
        print(f"\n📊 Output: shape={raw.shape}, dtype={raw.dtype}")
        flat = np.array(raw)
        while flat.ndim > 2:
            flat = flat[0] if flat.shape[0] == 1 else flat.reshape(-1, flat.shape[-1])
        
        print(f"  Flat: {flat.shape}")
        print(f"\n  Per-col stats:")
        print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
        for col in range(flat.shape[-1]):
            c = flat[:, col]
            print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")
        
        print(f"\n  Samples:")
        for i in [0, 100, 4000, 8399]:
            print(f"  [{i:4d}]: {flat[i]}")