#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import cv2 # OpenCV 임포트
import numpy as np

# 전역 변수
loop = None
pipeline = None
appsink = None
tracker = None
tracking_box = None
is_tracking = False

def bus_call(bus, message, loop):
    """파이프라인 메시지 콜백 함수"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True

def on_new_sample(sink):
    """Appsink에서 새 샘플이 도착했을 때 호출되는 콜백"""
    global tracker, tracking_box, is_tracking

    sample = sink.pull_sample()
    if sample is None:
        return Gst.FlowReturn.OK

    # GstSample에서 데이터 추출
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    # GstBuffer를 NumPy 배열로 변환 (OpenCV에서 사용 가능하도록)
    # 참고: 실제 포맷(예: BGRx)에 따라 shape와 변환 방식이 달라질 수 있음
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("Failed to map GstBuffer")
        return Gst.FlowReturn.ERROR

    # BGR 형식으로 가정 (파이프라인에서 변환 필요)
    frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8)
    frame_copy = frame.copy() # 원본을 수정하지 않기 위해 복사

    buffer.unmap(map_info) # 사용 후 unmap

    if is_tracking:
        # 트래커 업데이트
        success, box = tracker.update(frame_copy)
        if success:
            tracking_box = tuple(map(int, box))
            # 추적 중인 박스 그리기
            p1 = (tracking_box[0], tracking_box[1])
            p2 = (tracking_box[0] + tracking_box[2], tracking_box[1] + tracking_box[3])
            cv2.rectangle(frame_copy, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame_copy, "Tracking", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        else:
            # 추적 실패
            cv2.putText(frame_copy, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            is_tracking = False # 추적 중지

    # 화면에 프레임 표시
    cv2.imshow("ROI Tracking - Press 's' to select ROI, 'q' to quit", frame_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not is_tracking: # 's' 키를 누르면 ROI 선택 시작
        # 현재 프레임에서 ROI 선택
        # selectROI는 창이 활성화된 상태에서 스페이스바나 엔터를 누를 때까지 대기
        roi = cv2.selectROI("ROI Tracking - Press 's' to select ROI, 'q' to quit", frame_copy, fromCenter=False, showCrosshair=True)
        if roi[2] > 0 and roi[3] > 0: # 유효한 ROI가 선택되었는지 확인
            # OpenCV 트래커 초기화 (CSRT 사용 예시)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame_copy, roi)
            tracking_box = roi
            is_tracking = True
            print(f"ROI selected: {roi}, Starting tracker...")
        else:
            print("ROI selection cancelled.")
        cv2.destroyWindow("ROI selector") # selectROI 창 닫기 (필요시)

    elif key == ord('q'): # 'q' 키를 누르면 종료
        print("Quitting...")
        loop.quit()

    return Gst.FlowReturn.OK

def main():
    global loop, pipeline, appsink

    # OpenCV 설치 확인
    try:
        print(f"Using OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"Error importing OpenCV: {e}")
        print("Please install OpenCV (e.g., pip install opencv-python)")
        sys.exit(1)

    # GStreamer 초기화
    GObject.threads_init()
    Gst.init(None)

    # 비디오 파일 경로
    video_file_path = "/home/DeepStream-ROI-Template-Tracking/test_video/sample_720p.mp4"

    # 파이프라인 정의 (디코딩 -> 포맷 변환 -> appsink)
    # nvvideoconvert는 GPU 가속 변환, videoconvert는 CPU 변환
    # OpenCV는 주로 BGR 포맷을 사용
    pipeline_str = (
        f"filesrc location={video_file_path} ! qtdemux ! h264parse ! nvv4l2decoder ! "
        f"nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! " # GPU 메모리 상의 NV12
        f"nvvideoconvert ! video/x-raw,format=BGRx ! " # GPU 메모리 상의 BGRx
        f"videoconvert ! video/x-raw,format=BGR ! " # CPU 메모리 상의 BGR
        f"appsink name=mysink emit-signals=true max-buffers=1 drop=true"
    )

    print("Using pipeline: \n", pipeline_str)
    pipeline = Gst.parse_launch(pipeline_str)
    if not pipeline:
        sys.stderr.write(" Pipeline creation failed \n")
        sys.exit(1)

    # Appsink 엘리먼트 가져오기
    appsink = pipeline.get_by_name("mysink")
    if not appsink:
        sys.stderr.write(" Failed to get appsink element 'mysink'\n")
        sys.exit(1)

    # Appsink 콜백 설정
    appsink.set_property("emit-signals", True)
    appsink.connect("new-sample", on_new_sample)

    # 메인 루프 생성 및 실행
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # 파이프라인 시작
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    # OpenCV 창 생성
    cv2.namedWindow("ROI Tracking - Press 's' to select ROI, 'q' to quit", cv2.WINDOW_NORMAL)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 정리
        print("Exiting app\n")
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # OpenCV 설치 확인 후 main 실행
    main()