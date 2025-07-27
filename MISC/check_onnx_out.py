import onnx

model = onnx.load('/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/pth_sources/yolov5m_3class_250725/weights/best.onnx')

# 모든 노드 출력
print("=== All Nodes ===")
for i, node in enumerate(model.graph.node):
    print(f"Node {i}: {node.name}, Op: {node.op_type}, Output: {node.output}")

# 모델의 최종 출력들 확인
print("\n=== Model Outputs ===")
for output in model.graph.output:
    print(f"Final Output: {output.name}")

# 마지막 몇 개 노드들만 확인 (보통 여기에 출력 레이어들이 있음)
print("\n=== Last 10 Nodes ===")
for node in model.graph.node[-10:]:
    print(f"Layer: {node.name}, Op: {node.op_type}, Output: {node.output}")
    