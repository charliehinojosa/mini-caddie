"""Find the ONNX node names for the detection head conv outputs."""
import onnx

model = onnx.load('/content/drive/MyDrive/mini_caddie_golf_best.onnx')

lines = []
lines.append("=== Last 30 nodes ===")
for node in model.graph.node[-30:]:
    lines.append(f"  {node.op_type:20s}  {node.name}  inputs={[i[:30] for i in node.input]}  outputs={[o[:30] for o in node.output]}")

lines.append("\n=== Output nodes ===")
for out in model.graph.output:
    lines.append(f"  {out.name}")

lines.append("\n=== Nodes feeding /model.22/Concat_1 ===")
for node in model.graph.node:
    if 'Concat' in node.name and '22' in node.name:
        lines.append(f"  {node.name}  inputs={[i[:40] for i in node.input]}")

output = "\n".join(lines)
with open('/content/drive/MyDrive/onnx_nodes.txt', 'w') as f:
    f.write(output)
print("Saved to Drive as onnx_nodes.txt!")