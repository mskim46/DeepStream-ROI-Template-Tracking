from ultralytics import YOLO
import cv2
import os

# --- 설정 ---
model_path = "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/best.pt"
image_path = "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/test_image.jpg" # 테스트할 이미지 경로
output_dir = "pt_inference_output" # 결과 이미지를 저장할 디렉토리
confidence_threshold = 0.1 # 탐지 신뢰도 임계값 (낮게 설정하여 작은 객체도 확인)
# --- ---

# 결과 저장 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

try:
    print(f"Loading .pt model from: {model_path}")
    model = YOLO(model_path) # YOLOv8 (또는 호환되는 Ultralytics) 모델 로드
    print("Model loaded successfully.")

    print(f"Performing inference on image: {image_path}")
    # 모델을 사용하여 이미지 추론
    # save=True 옵션은 runs/detect/predict 폴더에 결과를 자동 저장합니다.
    # 여기서는 직접 결과를 처리하고 저장하는 예시를 보여드립니다.
    results = model.predict(source=image_path, conf=confidence_threshold)

    # 첫 번째 결과 (이미지가 하나이므로)
    result = results[0]
    
    if len(result.boxes) > 0:
        print(f"\n--- Detection Results from .pt model ({len(result.boxes)} objects found) ---")
        
        # 원본 이미지 로드 (결과 표기용)
        img_display = cv2.imread(image_path)
        if img_display is None:
            print(f"Error: Could not read image at {image_path}")
        else:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                label = model.names[class_id] # result.names[class_id] 도 가능
                confidence = float(box.conf)
                # bbox 좌표 (xmin, ymin, xmax, ymax)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                
                print(f"Object {i+1}: Label='{label}' (ID: {class_id}), Confidence={confidence:.4f}, BBox (xyxy)=[{x1}, {y1}, {x2}, {y2}]")
                
                # 이미지에 바운딩 박스와 레이블 그리기
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_display, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 결과 이미지 저장
            output_image_path = os.path.join(output_dir, "detected_by_pt_" + os.path.basename(image_path))
            cv2.imwrite(output_image_path, img_display)
            print(f"\nOutput image with detections saved to: {output_image_path}")
            print(f"Please open and check this image.")
    else:
        print("\n--- No objects detected by the .pt model with current settings. ---")
        print(f"Consider checking the image content or lowering the confidence_threshold (current: {confidence_threshold}).")
        # 원본 이미지를 결과 폴더에 복사 (비교용)
        if os.path.exists(image_path):
            cv2.imwrite(os.path.join(output_dir, "original_for_pt_test_" + os.path.basename(image_path)), cv2.imread(image_path))


except Exception as e:
    print(f"An error occurred during .pt model testing: {e}")
    import traceback
    traceback.print_exc()