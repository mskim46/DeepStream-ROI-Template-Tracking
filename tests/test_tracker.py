#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import GObject, Gst, GLib, GstApp
import cv2
import numpy as np
import pyds

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


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                # 여기서 탐지된 객체 정보를 처리할 수 있습니다
                print(f"Detected object: {obj_meta.class_id}, confidence: {obj_meta.confidence}")
            except StopIteration:
                break
            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


def main():
    global loop, pipeline, appsink, h264parser

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

    # 1. 엘리먼트 생성 (추론 관련 엘리먼트 추가)
    source = Gst.ElementFactory.make("filesrc", "file-source")
    qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux0")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    
    # 추론 엘리먼트 추가
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvconv1")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    nvconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvconv2")
    videoconvert = Gst.ElementFactory.make("videoconvert", "convert")
    appsink = Gst.ElementFactory.make("appsink", "mysink")

    if not all([source, qtdemux, h264parser, decoder, streammux, pgie, 
                nvconv1, nvosd, nvconv2, videoconvert, appsink]):
        sys.stderr.write(" Failed to create some elements\n")
        sys.exit(1)

    # 2. 엘리먼트 속성 설정
    source.set_property("location", video_file_path)
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    appsink.set_property("sync", False) # 속도 조절 안 할 경우 false 유지

    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    # nvinfer 설정
    pgie.set_property('config-file-path', 'configs/config_infer_primary_project.txt')

    # 3. 엘리먼트 파이프라인에 추가
    pipeline.add(source)
    pipeline.add(qtdemux)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvconv1)
    pipeline.add(nvosd)
    pipeline.add(nvconv2)
    pipeline.add(videoconvert)
    pipeline.add(appsink)

    # 4. 엘리먼트 연결
    # source -> qtdemux는 동일
    if not source.link(qtdemux):
        sys.stderr.write(" Failed to link source to qtdemux\n")
        sys.exit(1)

    # qtdemux -> h264parser (동적 패드 연결)
    qtdemux.connect("pad-added", on_pad_added, h264parser)

    # decoder의 src 패드를 streammux의 sink_0 패드에 연결
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
        sys.exit(1)

    # 나머지 엘리먼트 연결
    if not h264parser.link(decoder): sys.exit("Failed to link h264parser -> decoder")
    srcpad = decoder.get_static_pad("src")
    if not srcpad.link(sinkpad): sys.exit("Failed to link decoder -> streammux")
    if not streammux.link(pgie): sys.exit("Failed to link streammux -> pgie")
    if not pgie.link(nvconv1): sys.exit("Failed to link pgie -> nvconv1")
    if not nvconv1.link(nvosd): sys.exit("Failed to link nvconv1 -> nvosd")
    if not nvosd.link(nvconv2): sys.exit("Failed to link nvosd -> nvconv2")
    if not nvconv2.link(videoconvert): sys.exit("Failed to link nvconv2 -> videoconvert")
    if not videoconvert.link(appsink): sys.exit("Failed to link videoconvert -> appsink")

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

    # main 함수에서 probe 설정
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

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