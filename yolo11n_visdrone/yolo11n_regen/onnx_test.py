import onnx_graphsurgeon as gs
import onnx
import numpy as np

# --- 사용자 설정 시작 ---
INPUT_ONNX_FILE = "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/yolo11n_regen/best.onnx"
OUTPUT_ONNX_FILE = "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/yolo11n_regen/best_transposed_fixed.onnx"
TARGET_OUTPUT_NAME = "output0" # 최종적으로 원하는 출력 텐서의 이름
# --- 사용자 설정 끝 ---

print(f"ONNX GraphSurgeon 스크립트 시작: 출력 텐서 Transpose (수정됨 v2)")
print(f"입력 ONNX 모델: {INPUT_ONNX_FILE}")
print(f"출력 ONNX 모델: {OUTPUT_ONNX_FILE}")
print(f"대상 출력 텐서 이름: {TARGET_OUTPUT_NAME}")

try:
    graph = gs.import_onnx(onnx.load(INPUT_ONNX_FILE))
    print("ONNX 모델 로드 성공.")
except Exception as e:
    print(f"오류: ONNX 모델 로드 실패 - {e}")
    exit()

# 1. 기존 그래프의 출력 텐서(Variable)를 찾습니다.
original_tensor_to_modify = None
# graph.outputs는 gs.Variable 객체들의 리스트입니다.
# 이 리스트를 직접 수정하기보다는, 작업 후 새로 구성하는 것이 안전할 수 있습니다.
# 하지만 여기서는 찾아낸 Variable 객체의 name 속성을 변경할 것입니다.
for tensor_var in graph.outputs:
    if tensor_var.name == TARGET_OUTPUT_NAME:
        original_tensor_to_modify = tensor_var
        break

if original_tensor_to_modify is None:
    print(f"오류: 그래프 출력에서 '{TARGET_OUTPUT_NAME}' 이름의 텐서를 찾을 수 없습니다.")
    print(f"사용 가능한 출력 텐서: {[out.name for out in graph.outputs]}")
    exit()

print(f"찾은 원본 출력 텐서 (수정 전): 이름='{original_tensor_to_modify.name}', 형태={original_tensor_to_modify.shape}, 타입={original_tensor_to_modify.dtype}")

# 2. Transpose의 입력으로 사용될 원본 텐서의 이름을 고유한 임시 이름으로 변경합니다.
#    이렇게 하면 최종 출력 텐서의 이름과 충돌하지 않습니다.
intermediate_name = TARGET_OUTPUT_NAME + "_intermediate_for_transpose"
original_tensor_to_modify.name = intermediate_name
print(f"원본 출력 텐서의 이름을 임시로 변경: '{intermediate_name}'")

current_shape = original_tensor_to_modify.shape # 이름 변경 후에도 shape 등 다른 속성은 유지됨
if len(current_shape) != 3:
    print(f"오류: 대상 출력 텐서 '{intermediate_name}'의 차원이 3이 아닙니다 (현재: {len(current_shape)}).")
    exit()

transposed_shape = [current_shape[0], current_shape[2], current_shape[1]]
print(f"Transpose 후 예상되는 형태: {transposed_shape}")

# 3. Transpose된 새로운 출력 Variable을 생성합니다.
#    이 Variable이 그래프의 새로운 최종 출력이 되며, 원래 목표했던 TARGET_OUTPUT_NAME을 가집니다.
final_transposed_output_var = gs.Variable(
    name=TARGET_OUTPUT_NAME, # 최종 그래프 출력 이름을 원래대로 (예: "output0")
    dtype=original_tensor_to_modify.dtype,
    shape=transposed_shape
)
print(f"새로운 최종 출력 Variable 생성: 이름='{final_transposed_output_var.name}', 형태={final_transposed_output_var.shape}")

# 4. Transpose 노드를 생성합니다.
#    입력: 이름이 변경된 original_tensor_to_modify
#    출력: final_transposed_output_var
transpose_attributes = {"perm": [0, 2, 1]}
transpose_node = gs.Node(
    op="Transpose",
    name="FinalOutputTransposeNode", # Transpose 노드의 고유한 이름
    attrs=transpose_attributes,
    inputs=[original_tensor_to_modify], # 이름이 변경된 원본 텐서를 입력으로 사용
    outputs=[final_transposed_output_var]    # 새 최종 출력 변수를 출력으로 사용
)
print(f"Transpose 노드 생성: 입력='{original_tensor_to_modify.name}', 출력='{final_transposed_output_var.name}', perm={transpose_attributes['perm']}")

# 5. 그래프에 새로운 Transpose 노드를 추가합니다.
graph.nodes.append(transpose_node)

# 6. 그래프의 최종 출력을 final_transposed_output_var로 명확히 설정합니다.
#    original_tensor_to_modify (이제 intermediate_name을 가짐)는 더 이상 그래프의 직접적인 출력이 아닙니다.
graph.outputs = [final_transposed_output_var]
print(f"그래프의 최종 출력을 '{final_transposed_output_var.name}' (으)로 설정 완료.")

# 7. 그래프를 정리하고 위상 정렬합니다.
#    cleanup()은 사용되지 않는 노드/텐서를 제거하고, toposort()는 노드를 실행 순서대로 정렬합니다.
print("그래프 정리 및 위상 정렬 시도 중...")
graph.cleanup().toposort()
print("그래프 정리 및 위상 정렬 수행 완료.")

# 8. 수정된 그래프를 새 ONNX 파일로 저장합니다.
try:
    onnx.save(gs.export_onnx(graph), OUTPUT_ONNX_FILE)
    print(f"성공: 수정된 ONNX 모델이 '{OUTPUT_ONNX_FILE}'로 저장되었습니다.")
except Exception as e:
    print(f"오류: 수정된 ONNX 모델 저장 실패 - {e}")
    exit()

# 최종 확인
print("\n--- 최종 확인 ---")
try:
    reloaded_model = onnx.load(OUTPUT_ONNX_FILE)
    onnx.checker.check_model(reloaded_model)
    print(f"'{OUTPUT_ONNX_FILE}' 모델 유효성 검사 통과.")
    final_output_info_reloaded = None
    for out_node in reloaded_model.graph.output:
        if out_node.name == TARGET_OUTPUT_NAME:
            final_output_info_reloaded = out_node
            break
    if final_output_info_reloaded:
        dims = [dim.dim_value for dim in final_output_info_reloaded.type.tensor_type.shape.dim]
        print(f"저장된 모델의 출력: 이름='{final_output_info_reloaded.name}', 형태={dims}")
    else:
        print(f"저장된 모델에서 '{TARGET_OUTPUT_NAME}' 출력을 찾지 못했습니다.")

except Exception as e:
    print(f"오류: 저장된 모델 확인 중 문제 발생 - {e}")

print("\n다음 단계를 진행하세요:")
print(f"1. DeepStream 설정 파일 (`config_infer_visdrone.txt`)의 `onnx-file` 경로를 새 파일인 '{OUTPUT_ONNX_FILE}'로 업데이트하세요.")
print(f"2. 기존 TensorRT 엔진 파일을 **삭제**하세요.")
print(f"3. DeepStream 애플리케이션 (`test_image_visdrone.py`)을 다시 실행하여 이 ONNX 모델이 로드되는지 먼저 확인하세요.")
print(f"4. 모델 로드가 성공하면, 그 때 `.so` 파일의 파서가 이 `{transposed_shape}` 형태와 14개 특징을 올바르게 처리하는지 확인합니다.")