#!/usr/bin/env python3
"""
Get quantization info from HEF output + try to get float32.
Also check VStreamInfo for scale/zero-point.
"""
import os
import numpy as np
import hailo_platform
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

# Check ALL attributes on output vstream info
print("📊 Output VStreamInfo attributes:")
for o in output_info:
    print(f"\n  {o.name}:")
    for attr in sorted(dir(o)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(o, attr)
            if not callable(val):
                print(f"    {attr} = {val}")
        except:
            print(f"    {attr} = <error>")

# Try to find HailoFormat
print("\n\n🔍 Looking for HailoFormat...")
# Check pyhailort module
import hailo_platform.pyhailort.pyhailort as pyh
pyh_attrs = [a for a in dir(pyh) if 'format' in a.lower() or 'Format' in a or 'Hailo' in a]
print(f"  pyhailort format attrs: {pyh_attrs}")

# Try to construct HailoFormat
for cls_name in ['HailoFormat']:
    if hasattr(pyh, cls_name):
        cls = getattr(pyh, cls_name)
        print(f"  Found {cls_name}: {cls}")
        import inspect
        try:
            print(f"  init signature: {inspect.signature(cls.__init__)}")
        except:
            print(f"  (no inspect signature)")
        # Try constructing with FormatType.FLOAT32
        try:
            fmt = cls(FormatType.FLOAT32)
            print(f"  ✅ Constructed: {fmt}")
        except Exception as e:
            print(f"  ❌ cls(FormatType.FLOAT32): {e}")
        # Try with int
        try:
            fmt = cls(3)  # FLOAT32 = 3
            print(f"  ✅ Constructed with int(3): {fmt}")
        except Exception as e:
            print(f"  ❌ cls(3): {e}")

# Now try setting output format and running inference
print("\n\n🧠 Running inference with format attempts...")

inp_params = InputVStreamParams.make_from_network_group(ng)
out_params = OutputVStreamParams.make_from_network_group(ng)

out_p = list(out_params.values())[0]
print(f"  Default user_buffer_format: {out_p.user_buffer_format}")
print(f"  Type: {type(out_p.user_buffer_format)}")

# Try setting format
fmt_set = False
try:
    from hailo_platform.pyhailort.pyhailort import HailoFormat
    fmt = HailoFormat(FormatType.FLOAT32)
    out_p.user_buffer_format = fmt
    print(f"  ✅ Set HailoFormat(FLOAT32) from pyhailort")
    fmt_set = True
except Exception as e:
    print(f"  ❌ pyhailort HailoFormat: {e}")

if not fmt_set:
    # Try using the existing format object but changing its type
    try:
        existing = out_p.user_buffer_format
        print(f"  Existing format attrs: {[a for a in dir(existing) if not a.startswith('_')]}")
        if hasattr(existing, 'type'):
            print(f"  Existing format.type: {existing.type}")
            existing.type = FormatType.FLOAT32
            print(f"  ✅ Set .type = FLOAT32")
            fmt_set = True
    except Exception as e:
        print(f"  ❌ Set .type: {e}")

# Run inference regardless
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

        print(f"  Flat: {flat.shape}, dtype: {flat.dtype}")

        # If uint8, check if we can dequantize
        if flat.dtype == np.uint8:
            print(f"\n  ⚠️  Output is uint8 — checking quantization params from vstream info...")
            for o in output_info:
                print(f"  {o.name} attrs:")
                for attr in dir(o):
                    if attr.startswith('_'):
                        continue
                    try:
                        val = getattr(o, attr)
                        if not callable(val):
                            print(f"    {attr} = {val}")
                    except:
                        pass

            # Check if results has quantization info
            print(f"\n  Results type: {type(results)}")
            print(f"  Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")

        print(f"\n  Per-col stats:")
        print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
        for col in range(flat.shape[-1]):
            c = flat[:, col]
            print(f"  {col:4d}  {c.min():10.4f}  {c.max():10.4f}  {c.mean():10.4f}  {c.std():10.4f}")

        print(f"\n  Samples:")
        for i in [0, 100, 4000, 8399]:
            print(f"  [{i:4d}]: {flat[i]}")

        # Camera
        print(f"\n📷 Camera...")
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                inp = cv2.resize(frame, (640, 640))
                inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(inp, axis=0)
                cam_res = pipeline.infer({input_info[0].name: inp})
                cam_raw = cam_res[out_name]
                cam_flat = np.array(cam_raw)
                while cam_flat.ndim > 2:
                    cam_flat = cam_flat[0] if cam_flat.shape[0] == 1 else cam_flat.reshape(-1, cam_flat.shape[-1])
                print(f"  Camera output: {cam_raw.shape}, dtype: {cam_raw.dtype}")
                print(f"  Flat: {cam_flat.shape}")
                print(f"\n  Per-col stats:")
                for col in range(cam_flat.shape[-1]):
                    c = cam_flat[:, col]
                    print(f"  col {col}: min={c.min():.1f} max={c.max():.1f} mean={c.mean():.1f}")
                # Top 10
                max_vals = cam_flat.max(axis=-1)
                top10 = max_vals.argsort()[-10:][::-1]
                print(f"\n  Top 10 anchors:")
                for idx in top10:
                    print(f"  [{idx:4d}]: max={max_vals[idx]}  vals={cam_flat[idx]}")
            else:
                print("  ❌ read failed")
            cap.release()
        else:
            print("  ⚠️ no camera")

print("\n✅ Done!")