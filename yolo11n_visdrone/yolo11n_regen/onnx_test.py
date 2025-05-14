import onnx

# 이전에 로드한 onnx_model 객체를 그대로 사용합니다.
onnx_model = onnx.load("/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/yolo11n_regen/best.onnx")

print("--- Model Inputs ---")
for input_node in onnx_model.graph.input:
    print(f"Name: {input_node.name}")
    # 입력 노드의 차원 정보 출력
    dim_info = []
    try:
        for dim in input_node.type.tensor_type.shape.dim:
            if dim.dim_value:
                dim_info.append(str(dim.dim_value))
            elif dim.dim_param:
                dim_info.append(dim.dim_param) # 동적 차원 (예: 'batch')
            else:
                dim_info.append('?')
        print(f"Shape: [{', '.join(dim_info)}]")
    except AttributeError:
        print("Shape: (Could not determine shape directly)")
    print(f"Type: {input_node.type.tensor_type.elem_type}") # 데이터 타입 (예: 1 for float32)
    print("-" * 10)

print("\n--- Model Outputs ---")
for output_node in onnx_model.graph.output:
    print(f"Name: {output_node.name}")
    # 출력 노드의 차원 정보 출력
    dim_info = []
    try:
        for dim in output_node.type.tensor_type.shape.dim:
            if dim.dim_value:
                dim_info.append(str(dim.dim_value))
            elif dim.dim_param:
                dim_info.append(dim.dim_param) # 동적 차원
            else:
                dim_info.append('?')
        print(f"Shape: [{', '.join(dim_info)}]")
    except AttributeError:
        print("Shape: (Could not determine shape directly)")
    print(f"Type: {output_node.type.tensor_type.elem_type}") # 데이터 타입
    print("-" * 10)