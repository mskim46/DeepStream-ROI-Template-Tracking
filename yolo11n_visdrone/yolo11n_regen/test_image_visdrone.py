#!/usr/bin/env python3

import sys
print("DEBUG: Python script execution started.") # DEBUG PRINT

try:
    import gi
    print("DEBUG: gi module imported.") # DEBUG PRINT
    gi.require_version('Gst', '1.0')
    # gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GLib, GObject
    print("DEBUG: Gst, GLib, GObject imported.") # DEBUG PRINT
    
    import cv2 # OpenCV 사용
    print("DEBUG: cv2 module imported.") # DEBUG PRINT
    import numpy as np
    print("DEBUG: numpy module imported.") # DEBUG PRINT

except ImportError as ie:
    print(f"CRITICAL ERROR: Failed to import basic modules: {ie}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error during basic imports: {e}")
    sys.exit(1)

# DeepStream 메타데이터 구조체 임포트 (경로 확인 필요)
# 일반적으로 site-packages 아래에 있음
# 예: /opt/nvidia/deepstream/deepstream/lib/pyds.so
# 이 경로가 sys.path에 있거나, pyds.so가 있는 디렉토리에서 실행해야 함
# 또는 PYTHONPATH 환경 변수에 해당 경로 추가
try:
    import pyds
    print("DEBUG: pyds module imported successfully.") # DEBUG PRINT
except ImportError:
    print("ERROR: pyds module not found. Make sure DeepStream Python bindings are installed and in PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error during pyds import: {e}")
    sys.exit(1)

# 전역 변수로 탐지 결과 저장 (간단한 예시용)
detection_results = []

# PGIE의 src 패드에 연결될 프로브 함수
def pgie_src_pad_buffer_probe(pad, info, u_data):
    global detection_results
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        # print("Unable to get GstBuffer ") # 너무 자주 출력될 수 있어 주석 처리
        return Gst.PadProbeReturn.OK

    # 배치 메타데이터 가져오기
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
            except StopIteration:
                break
            
            # 탐지 결과 저장
            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)
            
            result = {
                "class_id": obj_meta.class_id,
                "label": obj_meta.obj_label,
                "confidence": obj_meta.confidence,
                "bbox": [left, top, width, height]
            }
            detection_results.append(result)
            
            # 콘솔에 간단히 출력 (디버깅용)
            # print(f"  Class: {obj_meta.obj_label}, Conf: {obj_meta.confidence:.2f}, "
            #       f"BBox: [{left},{top},{width},{height}]")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

def main(args):
    print("DEBUG: Entered main function.") # DEBUG PRINT
    global detection_results
    if len(args) != 2:
        sys.stderr.write("Usage: %s <image_file_path>\n" % args[0])
        sys.exit(1)

    image_path = args[1]
    print(f"DEBUG: Image path: {image_path}") # DEBUG PRINT

    # OpenCV로 이미지 로드
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not read image {image_path}")
            sys.exit(1)
        image_height, image_width, _ = img.shape
        print(f"DEBUG: Input image: {image_path}, Dimensions: {image_width}x{image_height}") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR: Failed to load image with OpenCV: {e}")
        sys.exit(1)
    
    # GStreamer 초기화
    try:
        GObject.threads_init()
        print("DEBUG: GObject.threads_init() called.") # DEBUG PRINT
        Gst.init(None)
        print("DEBUG: Gst.init(None) called.") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR: GStreamer initialization failed: {e}")
        sys.exit(1)

    # 파이프라인 생성
    print("DEBUG: Creating Pipeline.") # DEBUG PRINT (이전: Creating Pipeline \n)
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("ERROR: Unable to create Pipeline \n")
        sys.exit(1) # 파이프라인 생성 실패 시 종료

    # 엘리먼트 생성
    print("DEBUG: Creating Source (appsrc).") # DEBUG PRINT
    appsrc = Gst.ElementFactory.make("appsrc", "app-source")
    if not appsrc:
        sys.stderr.write("ERROR: Unable to create appsrc \n")
        sys.exit(1)

    # (선택적) 이미지 포맷 변환 및 크기 조정을 위한 videoconvert, capsfilter
    videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
    if not videoconvert: sys.stderr.write("ERROR: Unable to create videoconvert\n"); sys.exit(1)
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert") # GPU 가속 변환
    if not nvvidconv: sys.stderr.write("ERROR: Unable to create nvvideoconvert\n"); sys.exit(1)
    
    # 스트림 먹서 생성 (단일 소스)
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("ERROR: Unable to create NvStreamMux \n")
        sys.exit(1)

    # PGIE (Primary GStreamer Inference Engine) 생성
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("ERROR: Unable to create pgie \n")
        sys.exit(1)
    
    # (선택적) OSD 생성 - 여기서는 결과만 추출하므로 필수는 아님
    # nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    # if not nvdsosd:
    #     sys.stderr.write(" Unable to create nvdsosd \n")
    #     return

    # 싱크 생성 (fakesink 사용, 화면 출력 없음)
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    if not sink:
        sys.stderr.write("ERROR: Unable to create fakesink \n")
        sys.exit(1)

    print("DEBUG: All GStreamer elements created.") # DEBUG PRINT

    # appsrc 설정
    # OpenCV는 BGR 포맷으로 이미지를 읽음. DeepStream은 RGBA 등을 선호할 수 있음.
    # 여기서는 BGR로 시도하고, 필요시 videoconvert/nvvideoconvert로 변환.
    # 모델 입력 크기(예: 640x640)와 이미지 크기가 다를 수 있으므로,
    # nvinfer의 maintain-aspect-ratio=1 설정이 중요.
    # 또는 appsrc 전에 OpenCV로 리사이즈 할 수도 있음.
    caps_str = f"video/x-raw,format=BGR,width={image_width},height={image_height},framerate=1/1"
    caps = Gst.Caps.from_string(caps_str)
    appsrc.set_property("caps", caps)
    appsrc.set_property("format", Gst.Format.TIME) # 또는 GST_FORMAT_BYTES
    appsrc.set_property('block', True) # 블로킹 모드
    print(f"DEBUG: appsrc caps set to: {caps_str}") # DEBUG PRINT

    # 스트림 먹서 설정
    streammux.set_property('width', image_width) # 또는 모델 입력 너비
    streammux.set_property('height', image_height) # 또는 모델 입력 높이
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000) # microseconds
    print("DEBUG: streammux properties set.") # DEBUG PRINT

    # PGIE 설정
    pgie_config_path = "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/yolo11n_regen/config_infer_deep_yolo11.txt"
    pgie.set_property('config-file-path', pgie_config_path)
    print(f"DEBUG: pgie config-file-path set to: {pgie_config_path}") # DEBUG PRINT

    # 파이프라인에 엘리먼트 추가
    print("DEBUG: Adding elements to Pipeline.") # DEBUG PRINT
    try:
        pipeline.add(appsrc)
        pipeline.add(videoconvert) # BGR -> 다른 포맷 (예: I420)
        pipeline.add(nvvidconv)    # I420 -> NV12 (nvinfer가 선호하는 포맷 중 하나)
        pipeline.add(streammux)
        pipeline.add(pgie)
        # pipeline.add(nvdsosd)
        pipeline.add(sink)
    except Exception as e:
        print(f"ERROR: Failed to add elements to pipeline: {e}")
        sys.exit(1)
    print("DEBUG: All elements added to pipeline.") # DEBUG PRINT

    # 엘리먼트 연결
    print("DEBUG: Linking elements in the Pipeline.") # DEBUG PRINT
    # appsrc -> videoconvert -> nvvidconv -> streammux (sink_0)
    # streammux -> pgie -> (nvdsosd) -> sink

    # appsrc -> videoconvert
    if not appsrc.link(videoconvert):
        sys.stderr.write("ERROR: Elements could not be linked: appsrc -> videoconvert\n")
        sys.exit(1)
    
    # videoconvert -> nvvidconv
    # nvvidconv에 대한 입력 caps 설정 (videoconvert 출력과 맞춤)
    caps_i420 = Gst.Caps.from_string("video/x-raw, format=I420")
    filter_i420 = Gst.ElementFactory.make("capsfilter", "filter_i420")
    if not filter_i420: sys.stderr.write("ERROR: Unable to create filter_i420\n"); sys.exit(1)
    filter_i420.set_property("caps", caps_i420)
    pipeline.add(filter_i420) # 파이프라인에 추가하는 것을 잊지 말 것
    
    if not videoconvert.link(filter_i420):
        sys.stderr.write("ERROR: Elements could not be linked: videoconvert -> filter_i420\n")
        sys.exit(1)
    if not filter_i420.link(nvvidconv):
        sys.stderr.write("ERROR: Elements could not be linked: filter_i420 -> nvvidconv\n")
        sys.exit(1)

    # nvvidconv -> streammux
    # streammux의 sink 패드 가져오기
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("ERROR: Unable to get the sink pad of streammux \n")
        sys.exit(1)
    srcpad = nvvidconv.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("ERROR: Unable to get source pad of nvvidconv \n")
        sys.exit(1)
    if not srcpad.link(sinkpad) == Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: Unable to link nvvidconv to streammux\n")
        sys.exit(1)

    # streammux -> pgie
    if not streammux.link(pgie):
        sys.stderr.write("ERROR: Elements could not be linked: streammux -> pgie\n")
        sys.exit(1)
    
    # pgie -> sink (nvdsosd를 사용한다면 pgie -> nvdsosd -> sink)
    if not pgie.link(sink):
        sys.stderr.write("ERROR: Elements could not be linked: pgie -> sink\n")
        sys.exit(1)
    print("DEBUG: All elements linked successfully.") # DEBUG PRINT

    # 메인 루프 및 버스 메시지 핸들러
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("DEBUG: Bus call: End-of-stream received.") # DEBUG PRINT
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error from GStreamer bus: %s: %s\n" % (err, debug)) # DEBUG PRINT
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)

    # PGIE의 src 패드에 프로브 추가
    pgiesrcpad = pgie.get_static_pad("src")
    if not pgiesrcpad:
        sys.stderr.write("ERROR: Unable to get src pad of pgie \n")
        sys.exit(1)
    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
    print("DEBUG: Probe added to pgie src pad.") # DEBUG PRINT

    # 파이프라인 시작
    print("DEBUG: Attempting to set pipeline to PLAYING state.") # DEBUG PRINT
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Failed to set pipeline to PLAYING state.")
        # 에러 메시지를 보기 위해 버스에서 메시지를 확인하려고 시도할 수 있지만,
        # PLAYING으로 가지 못하면 버스 메시지가 안 올 수도 있음.
        # 이 경우, GStreamer 디버그 레벨을 높여서 실행하는 것이 도움이 됨.
        # 예: GST_DEBUG=3 python3 test_image_visdrone.py ...
        pipeline.set_state(Gst.State.NULL) # 정리 시도
        sys.exit(1)
    print("DEBUG: Pipeline state set to PLAYING. Entering GLib main loop.") # DEBUG PRINT

    try:
        print("DEBUG: Preparing image buffer for appsrc.") # DEBUG PRINT
        data = img.tobytes()
        gst_buffer = Gst.Buffer.new_allocate(None, len(data), None)
        gst_buffer.fill(0, data)
        
        print("DEBUG: Pushing buffer to appsrc.") # DEBUG PRINT
        retval = appsrc.emit("push-buffer", gst_buffer)
        if retval != Gst.FlowReturn.OK:
            print(f"ERROR: Failed to push buffer to appsrc: {retval}")
        else:
            print("DEBUG: Buffer pushed successfully.") # DEBUG PRINT
        
        print("DEBUG: Emitting end-of-stream to appsrc.") # DEBUG PRINT
        appsrc.emit("end-of-stream")

        print("DEBUG: Entering loop.run()") # DEBUG PRINT
        loop.run() 
        print("DEBUG: Exited loop.run()") # DEBUG PRINT
    except Exception as e:
        print(f"ERROR: Exception during pipeline execution or buffer push: {e}")
    finally:
        print("DEBUG: In finally block. Setting pipeline to NULL.") # DEBUG PRINT
        pipeline.set_state(Gst.State.NULL)
        print("DEBUG: Pipeline state set to NULL.") # DEBUG PRINT

        # 결과 출력
        print("\n--- Detection Results ---")
        if detection_results:
            for i, res in enumerate(detection_results):
                print(f"Object {i+1}:")
                print(f"  Class ID: {res['class_id']}")
                print(f"  Label: {res['label']}")
                print(f"  Confidence: {res['confidence']:.4f}")
                print(f"  BBox (x,y,w,h): {res['bbox']}")
        else:
            print("No objects detected.")
        
        # (선택 사항) 결과 이미지 저장
        import os 
        output_image_path = "detected_" + os.path.basename(image_path)
        img_display = cv2.imread(image_path) 
        if img_display is not None:
            for res in detection_results:
                x, y, w, h = res['bbox']
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_display, f"{res['label']} {res['confidence']:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(output_image_path, img_display)
            print(f"Output image saved to: {output_image_path}")
        else:
            print(f"Could not reload image {image_path} for saving with detections.")
        print("DEBUG: End of main function.") # DEBUG PRINT

if __name__ == '__main__':
    print("DEBUG: Script __main__ block started.") # DEBUG PRINT
    try:
        sys.exit(main(sys.argv))
    except SystemExit:
        print("DEBUG: SystemExit caught, script will now exit.") # DEBUG PRINT
    except Exception as e:
        print(f"CRITICAL ERROR in __main__: {e}") # DEBUG PRINT