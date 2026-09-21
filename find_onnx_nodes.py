"""Find the ONNX node names for the detection head conv outputs."""
import onnx

model = onnx.load('/content/drive/MyDrive/mini_caddie_golf_best.onnx')

print("=== Last 30 nodes ===")
for node in model.graph.node[-30:]:
    print(f"  {node.op_type:20s}  {node.name}  inputs={[i[:30] for i in node.input]}  outputs={[o[:30] for o in node.output]}")

print("\n=== Output nodes ===")
for out in model.graph.output:
    print(f"  {out.name}")

# Find Conv nodes that feed into Concat_1
print("\n=== Nodes feeding /model.22/Concat_1 ===")
for node in model.graph.node:
    if 'Concat' in node.name and '22' in node.name:
        print(f"  {node.name}  inputs={[i[:40] for i in node.input]}")