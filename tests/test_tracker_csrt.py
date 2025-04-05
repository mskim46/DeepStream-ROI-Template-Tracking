#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import GObject, Gst, GLib, GstApp
import cv2
import numpy as np

# ... (전역 변수 선언 동일) ...
loop = None
pipeline = None
appsink = None
tracker = None
tracking_box = None
is_tracking = False
# 추가: qtdemux 연결을 위한 전역 변수
h264parser = None


def bus_call(bus, message, loop):
    # ... (bus_call 함수 내용은 동일) ...
    """파이프라인 메시지 콜백 함수"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        # 오류 발생 시 소스 엘리먼트 이름 확인 추가
        src_name = message.src.get_name() if message.src else "Unknown"
        print(f"Error details: Source={src_name}, Message={err.message}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    # elif t == Gst.MessageType.STATE_CHANGED: # 디버깅용 상태 변경 메시지
    #     old_state, new_state, pending_state = message.parse_state_changed()
    #     if message.src == pipeline:
    #          print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
    return True


# qtdemux의 pad-added 신호 처리 함수
def on_pad_added(element, pad, target_element):
    """qtdemux에서 새 패드가 추가될 때 호출되는 콜백"""
    pad_caps = pad.get_current_caps()
    if not pad_caps:
        print("Pad has no caps, ignoring")
        return

    structure_name = pad_caps.get_structure(0).get_name()
    print(f"Pad added: {structure_name}")

    # 비디오 패드만 연결
    if structure_name.startswith("video/x-h264"):
        sink_pad = target_element.get_static_pad("sink")
        if not sink_pad:
            print("Failed to get sink pad of h264parse")
            return

        if pad.link(sink_pad) == Gst.PadLinkReturn.OK:
            print("Successfully linked qtdemux video pad to h264parse sink pad")
        else:
            print("Failed to link qtdemux video pad to h264parse sink pad")
    # 필요하다면 오디오 등 다른 패드 처리 추가


def on_new_sample(sink: GstApp.AppSink):
    # ... (on_new_sample 함수 내용은 이전과 동일) ...
    """Appsink에서 새 샘플이 도착했을 때 호출되는 콜백"""
    global tracker, tracking_box, is_tracking, loop

    try:
        sample = sink.pull_sample()
    except AttributeError as e:
        print(f"ERROR: {e}")
        print("ERROR: pull_sample method still not found. Stopping pipeline.")
        if loop: loop.quit()
        return Gst.FlowReturn.ERROR
    except Exception as e:
        print(f"ERROR during pull_sample: {e}")
        if loop: loop.quit()
        return Gst.FlowReturn.ERROR

    if sample is None:
        print("Warning: pull_sample returned None.")
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    if not buffer:
        print("Error: Failed to get GstBuffer from sample.")
        return Gst.FlowReturn.ERROR

    caps = sample.get_caps()
    if not caps:
        print("Error: Failed to get GstCaps from sample.")
        return Gst.FlowReturn.ERROR

    structure = caps.get_structure(0)
    if not structure:
        print("Error: Failed to get GstStructure from caps.")
        return Gst.FlowReturn.ERROR

    height = structure.get_value("height")
    width = structure.get_value("width")

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("Error: Failed to map GstBuffer")
        return Gst.FlowReturn.ERROR

    expected_size = height * width * 3
    if map_info.size != expected_size:
         print(f"Warning: Buffer size ({map_info.size}) does not match expected size ({expected_size}) for BGR format.")

    try:
        frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8)
        frame_copy = frame.copy()
    except Exception as e:
        print(f"Error converting buffer to NumPy array: {e}")
        buffer.unmap(map_info)
        return Gst.FlowReturn.ERROR

    buffer.unmap(map_info)

    if is_tracking:
        try:
            success, box = tracker.update(frame_copy)
            if success:
                tracking_box = tuple(map(int, box))
                p1 = (tracking_box[0], tracking_box[1])
                p2 = (tracking_box[0] + tracking_box[2], tracking_box[1] + tracking_box[3])
                
                # 객체 크기에 따른 동적 설정
                box_area = tracking_box[2] * tracking_box[3]  # 박스 영역
                frame_area = frame_copy.shape[0] * frame_copy.shape[1]  # 전체 프레임 영역
                
                # 박스 크기에 따른 선 두께 계산
                thickness = max(1, int((box_area / frame_area) * 200))
                
                # 텍스트 크기 계산 (박스 크기에 비례)
                font_scale = max(0.5, min(2.0, (box_area / frame_area) * 500))
                
                # 박스와 텍스트 그리기
                cv2.rectangle(frame_copy, p1, p2, (0, 255, 0), thickness, 1)
                cv2.putText(frame_copy, "Tracking", 
                          (p1[0], p1[1]-max(10, int(font_scale * 15))), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, 
                          (0,255,0), 
                          max(1, int(thickness/2)))
            else:
                # 트래킹 실패 메시지도 동적 크기로 조정
                frame_height = frame_copy.shape[0]
                font_scale = frame_height / 1000.0  # 프레임 크기에 비례
                cv2.putText(frame_copy, "Tracking failure detected", 
                          (100, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, 
                          (0,0,255), 
                          max(1, int(font_scale * 2)))
                is_tracking = False
        except Exception as e:
            print(f"Error during tracker update: {e}")
            is_tracking = False

    try:
        cv2.imshow("ROI Tracking - Press 's' to select ROI, 'q' to quit", frame_copy)
        key = cv2.waitKey(1) & 0xFF
    except Exception as e:
        print(f"Error during cv2.imshow or waitKey: {e}")
        if loop: loop.quit()
        return Gst.FlowReturn.ERROR

    if key == ord('s') and not is_tracking:
        try:
            print("Select a ROI and then press SPACE or ENTER button!")
            print("Cancel the selection process by pressing c button!")
            roi = cv2.selectROI("ROI Tracking - Press 's' to select ROI, 'q' to quit", frame_copy, fromCenter=False, showCrosshair=True)
            if roi[2] > 0 and roi[3] > 0:
                # opencv-contrib-python 설치 후 이 부분 작동 기대
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame_copy, roi)
                tracking_box = roi
                is_tracking = True
                print(f"ROI selected: {roi}, Starting tracker...")
            else:
                print("ROI selection cancelled.")
        except AttributeError:
             print("ERROR: cv2.legacy.TrackerCSRT_create not found. Check OpenCV version and installation (opencv-contrib-python needed).")
             if loop: loop.quit()
        except Exception as e:
            print(f"Error during ROI selection or tracker init: {e}")

    elif key == ord('q'):
        print("Quitting...")
        if loop: loop.quit()

    return Gst.FlowReturn.OK


def main():
    global loop, pipeline, appsink, h264parser # h264parser 전역 변수 사용

    # ... (OpenCV 확인 및 Gst 초기화 동일) ...
    try:
        print(f"Using OpenCV version: {cv2.__version__}")
        # contrib 패키지 확인 (경고만 출력)
        if not hasattr(cv2, 'legacy') or not hasattr(cv2.legacy, 'TrackerCSRT_create'):
             print("Warning: cv2.legacy.TrackerCSRT_create not found. ")
             print("Ensure 'opencv-contrib-python' is installed for tracking.")
    except Exception as e:
        print(f"Error importing or checking OpenCV: {e}")
        sys.exit(1)

    Gst.init(None)

    video_file_path = "/home/DeepStream-ROI-Template-Tracking/test_video/sample_720p.mp4"

    # 파이프라인 생성 (Gst.parse_launch 대신 직접 생성)
    pipeline = Gst.Pipeline.new("pipeline0")
    if not pipeline:
        sys.stderr.write(" Pipeline creation failed\n")
        sys.exit(1)

    # 1. 엘리먼트 생성
    source = Gst.ElementFactory.make("filesrc", "file-source")
    qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux0")
    # h264parser는 전역 변수로 저장하여 콜백에서 사용
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvconv1")
    nvconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvconv2")
    videoconvert = Gst.ElementFactory.make("videoconvert", "convert")
    # videorate는 일단 제거하고 테스트 (필요시 나중에 추가)
    # videorate = Gst.ElementFactory.make("videorate", "rate-adjust")
    appsink = Gst.ElementFactory.make("appsink", "mysink")

    if not all([source, qtdemux, h264parser, decoder, nvconv1, nvconv2, videoconvert, appsink]):
        sys.stderr.write(" Failed to create some elements\n")
        sys.exit(1)

    # 2. 엘리먼트 속성 설정
    source.set_property("location", video_file_path)
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    appsink.set_property("sync", False) # 속도 조절 안 할 경우 false 유지

    # 3. 엘리먼트 파이프라인에 추가
    pipeline.add(source)
    pipeline.add(qtdemux)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(nvconv1)
    pipeline.add(nvconv2)
    pipeline.add(videoconvert)
    # pipeline.add(videorate) # 필요시 추가
    pipeline.add(appsink)

    # 4. 정적 엘리먼트 연결 (source -> qtdemux)
    if not source.link(qtdemux):
        sys.stderr.write(" Failed to link source to qtdemux\n")
        sys.exit(1)

    # 5. qtdemux의 pad-added 신호 연결 (동적 연결 처리)
    qtdemux.connect("pad-added", on_pad_added, h264parser)

    # 6. 나머지 엘리먼트 연결 (h264parser부터 appsink까지)
    # Caps 필터 추가 (nvvideoconvert 연결 시 필요할 수 있음)
    caps_nvmm = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
    filter_nvmm = Gst.ElementFactory.make("capsfilter", "filter-nvmm")
    filter_nvmm.set_property("caps", caps_nvmm)
    pipeline.add(filter_nvmm)

    caps_bgrx = Gst.Caps.from_string("video/x-raw,format=BGRx")
    filter_bgrx = Gst.ElementFactory.make("capsfilter", "filter-bgrx")
    filter_bgrx.set_property("caps", caps_bgrx)
    pipeline.add(filter_bgrx)

    caps_bgr = Gst.Caps.from_string("video/x-raw,format=BGR")
    filter_bgr = Gst.ElementFactory.make("capsfilter", "filter-bgr")
    filter_bgr.set_property("caps", caps_bgr)
    pipeline.add(filter_bgr)

    # caps_rate = Gst.Caps.from_string("video/x-raw,framerate=3/1") # videorate 사용 시
    # filter_rate = Gst.ElementFactory.make("capsfilter", "filter-rate")
    # filter_rate.set_property("caps", caps_rate)
    # pipeline.add(filter_rate)

    # 연결 순서: h264parser -> decoder -> nvconv1 -> filter_nvmm -> nvconv2 -> filter_bgrx -> videoconvert -> filter_bgr -> appsink
    if not h264parser.link(decoder): sys.exit("Failed link h264parser -> decoder")
    if not decoder.link(nvconv1): sys.exit("Failed link decoder -> nvconv1")
    if not nvconv1.link(filter_nvmm): sys.exit("Failed link nvconv1 -> filter_nvmm")
    if not filter_nvmm.link(nvconv2): sys.exit("Failed link filter_nvmm -> nvconv2")
    if not nvconv2.link(filter_bgrx): sys.exit("Failed link nvconv2 -> filter_bgrx")
    if not filter_bgrx.link(videoconvert): sys.exit("Failed link filter_bgrx -> videoconvert")
    if not videoconvert.link(filter_bgr): sys.exit("Failed link videoconvert -> filter_bgr")
    # if not filter_bgr.link(videorate): sys.exit("Failed link filter_bgr -> videorate") # videorate 사용 시
    # if not videorate.link(filter_rate): sys.exit("Failed link videorate -> filter_rate") # videorate 사용 시
    # if not filter_rate.link(appsink): sys.exit("Failed link filter_rate -> appsink") # videorate 사용 시
    if not filter_bgr.link(appsink): sys.exit("Failed link filter_bgr -> appsink") # videorate 미사용 시

    # Appsink 콜백 설정
    appsink.connect("new-sample", on_new_sample)

    # 메인 루프 생성 및 실행
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    cv2.namedWindow("ROI Tracking - Press 's' to select ROI, 'q' to quit", cv2.WINDOW_NORMAL)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting app\n")
        if pipeline:
            pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        if loop and loop.is_running():
            loop.quit()


if __name__ == '__main__':
    main()