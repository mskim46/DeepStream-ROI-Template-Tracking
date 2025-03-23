import os
import sys
import cv2
import numpy as np
import time

# 전역 변수
template_img = None
frame_count = 0

def main():
    global template_img, frame_count
    
    # 비디오 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    video_file = os.path.join(project_root, "test_video", "sample_720p.mp4")
    print("Using video file at:", video_file)
    
    # OpenCV VideoCapture 생성
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    # 비디오 정보 출력
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # 창 생성
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1280, 720)
    
    # 비디오 반복 재생 루프
    while True:
        ret, frame = cap.read()
        
        # 파일 끝에 도달하면 처음으로 돌아가기
        if not ret:
            print("End of video reached, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame #{frame_count}")
        
        # 첫 프레임에서 랜덤 ROI 선택
        if template_img is None:
            roi_w = width // 4
            roi_h = height // 4
            x = np.random.randint(0, width - roi_w)
            y = np.random.randint(0, height - roi_h)
            template_img = frame[y:y+roi_h, x:x+roi_w].copy()
            print(f"Random ROI selected at: ({x}, {y}, {roi_w}, {roi_h})")
            
            # 템플릿 이미지 저장 (디버깅용)
            cv2.imwrite("selected_template.jpg", template_img)
            print("Template image saved to 'selected_template.jpg'")
            
            # 빨간색으로 선택한 ROI 표시
            cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), (0, 0, 255), 3)
        else:
            # 템플릿 매칭 수행
            res = cv2.matchTemplate(frame, template_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if frame_count % 30 == 0:
                print(f"Match confidence: {max_val:.4f} at location {max_loc}")
            
            # 매칭된 위치에 사각형 및 텍스트 표시
            top_left = max_loc
            temp_h, temp_w = template_img.shape[:2]
            bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
            cv2.putText(frame, f"Match: {max_val:.2f}", 
                      (top_left[0], top_left[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_count}", 
                  (width - 150, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 화면에 표시
        cv2.imshow("Frame", frame)
        
        # 키 입력 대기 (약 30 FPS 속도로 제한)
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            print("User pressed 'q', quitting...")
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()