#!/usr/bin/env python3
"""Quick API introspection — figure out what methods/attributes are available."""
import os
from hailo_platform import HEF, VDevice, ConfigureParams, HailoStreamInterface, InputVStreamParams, OutputVStreamParams

hef = HEF(os.path.expanduser("~/mini-caddie/mini_caddie_golf.hef"))
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_groups = target.configure(hef, configure_params)
ng = network_groups[0]

print(f"network_group type: {type(ng)}")
print(f"network_group attrs: {[a for a in dir(ng) if not a.startswith('__')]}")

# Check if activate() exists and what it returns
if hasattr(ng, 'activate'):
    print("\n→ ng has activate()")
    try:
        activated = ng.activate()
        print(f"  activate() returned type: {type(activated)}")
        print(f"  activated attrs: {[a for a in dir(activated) if not a.startswith('__')]}")
    except Exception as e:
        print(f"  activate() failed: {e}")
else:
    print("\n→ ng does NOT have activate()")

# Check InputVStreamParams.make signature
import inspect
print(f"\nInputVStreamParams.make signature: {inspect.signature(InputVStreamParams.make)}")

# Try make_from_network_group
if hasattr(InputVStreamParams, 'make_from_network_group'):
    print(f"  make_from_network_group signature: {inspect.signature(InputVStreamParams.make_from_network_group)}")