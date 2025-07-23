import onnx

# 생성된 ONNX 모델 확인
model = onnx.load('runs/train/exp3/weights/best.onnx')

print("=== ONNX 모델 정보 ===")
for inp in model.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"입력 '{inp.name}': {shape}")

for out in model.graph.output:
    shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
    print(f"출력 '{out.name}': {shape}")