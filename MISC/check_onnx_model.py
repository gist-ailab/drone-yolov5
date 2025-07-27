import onnx
import onnxruntime as ort
import numpy as np

# ONNX 파일 로드 및 검증
onnx_path = "/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/pth_sources/yolov5m_3class_250725/weights/best.onnx"

try:
    # ONNX 모델 로드
    model = onnx.load(onnx_path)
    print("✅ ONNX 파일 로드 성공")
    
    # 모델 검증
    onnx.checker.check_model(model)
    print("✅ ONNX 모델 유효성 검사 통과")
    
    # 입력/출력 정보 확인
    print(f"\n=== 입력 정보 ===")
    for input_tensor in model.graph.input:
        print(f"이름: {input_tensor.name}")
        print(f"형태: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
    
    print(f"\n=== 출력 정보 ===")
    for output_tensor in model.graph.output:
        print(f"이름: {output_tensor.name}")
        print(f"형태: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")
    
    # ONNX Runtime으로 테스트
    ort_session = ort.InferenceSession(onnx_path)
    print("✅ ONNX Runtime 세션 생성 성공")
    
    # 더미 입력으로 추론 테스트
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    outputs = ort_session.run(None, {input_name: dummy_input})
    print(f"✅ 추론 테스트 성공")
    print(f"출력 개수: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"출력 {i} 형태: {output.shape}")
        
except Exception as e:
    print(f"❌ ONNX 검증 실패: {e}")