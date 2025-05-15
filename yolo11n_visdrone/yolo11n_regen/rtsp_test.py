import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GLib, GObject
from gi.repository import GstRtsp

# Initialize GStreamer
Gst.init(None)

# Programmatically set GStreamer debug levels
# This is an alternative/supplement to setting GST_DEBUG environment variable
# Gst.DebugLevel.NONE (0) to Gst.DebugLevel.MEMDUMP (9)
# Level 6 is LOGS (very verbose), Level 5 is DEBUG (verbose)
# Level 4 is INFO, Level 3 is WARNING
#Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING) # Set a general default
#Gst.debug_set_threshold_for_name("rtph264depay", Gst.DebugLevel.LOG) # LOGS -> LOG
#Gst.debug_set_threshold_for_name("h264parse", Gst.DebugLevel.LOG)   # LOGS -> LOG
#Gst.debug_set_threshold_for_name("GST_STATES", Gst.DebugLevel.DEBUG) # For GStreamer state change messages
#Gst.debug_set_threshold_for_name("GST_PADS", Gst.DebugLevel.DEBUG)   # For GStreamer pad linking messages
#Gst.debug_set_threshold_for_name("GST_BIN", Gst.DebugLevel.DEBUG)    # For GStreamer bin (pipeline) operations like add/remove

# Standard GStreamer pipeline elements
PGIE_CLASS_ID_VEHICLE = 0

# PGIE 설정
pgie_config_file = "config_infer_deep_yolo11.txt" # PGIE 설정 파일 경로

# 콜백 함수: rtspsrc에서 패드가 동적으로 생성될 때 호출됨
def on_pad_added(src_element, new_pad, depay_element): # user_data is depay_element
    print(f"!!! on_pad_added CALLED for {src_element.get_name()} with pad {new_pad.get_name()} !!!")

    new_pad_caps = new_pad.get_current_caps()
    if not new_pad_caps:
        print("  New pad has no caps, ignoring.")
        return

    new_pad_struct = new_pad_caps.get_structure(0)
    new_pad_type = new_pad_struct.get_name()

    if not new_pad_type.startswith("application/x-rtp"):
        print(f"  New pad type '{new_pad_type}' is not 'application/x-rtp'. Ignoring.")
        return

    encoding_name = new_pad_struct.get_string("encoding-name")
    if encoding_name and encoding_name.upper() == "H264":
        depay_sink_pad = depay_element.get_static_pad("sink")
        if not depay_sink_pad:
            sys.stderr.write(" Unable to get sink pad of depayloader \n")
            return

        if depay_sink_pad.is_linked():
            print(f"  Sink pad of {depay_element.get_name()} is already linked. Ignoring.")
            depay_sink_pad.unref()
            return

        # 디버깅: depayloader의 sink 패드 capabilities 확인
        depay_sink_pad_caps = depay_sink_pad.get_current_caps()
        if not depay_sink_pad_caps:
            depay_sink_pad_template = depay_sink_pad.get_pad_template()
            if depay_sink_pad_template:
                depay_sink_pad_template_caps = depay_sink_pad_template.get_caps()
                print(f"  Depay sink pad template caps: {depay_sink_pad_template_caps.to_string() if depay_sink_pad_template_caps else 'None'}")
                depay_sink_pad_template.unref()
            else:
                print("  Depay sink pad has no template.")
        else:
            print(f"  Depay sink pad current caps: {depay_sink_pad_caps.to_string()}")

        link_result = new_pad.link(depay_sink_pad)
        if link_result != Gst.PadLinkReturn.OK:
            print(f"  Failed to link '{new_pad.get_name()}' to '{depay_sink_pad.get_name()}'. Result: {link_result}")
        else:
            print(f"  Successfully linked '{new_pad.get_name()}' to '{depay_sink_pad.get_name()}'.")
        
        depay_sink_pad.unref()
    else:
        print(f"  New pad encoding '{encoding_name}' is not H264. Ignoring.")

    return

# === 추가 시작: bus_call 함수 정의 ===
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error received from element {message.src.get_name()}: {err.message}\n")
        if debug:
            sys.stderr.write(f"Debugging information: {debug}\n")
        loop.quit()
    return True
# === 추가 끝 ===

def main(args):
    '''
    # Initialize GStreamer
    Gst.init(None)
    # Programmatically set GStreamer debug levels
    Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING) # Set a general default
    Gst.debug_set_threshold_for_name("rtph264depay", Gst.DebugLevel.LOG) # LOGS -> LOG
    Gst.debug_set_threshold_for_name("h264parse", Gst.DebugLevel.LOG)   # LOGS -> LOG
    Gst.debug_set_threshold_for_name("nvv4l2decoder", Gst.DebugLevel.LOG) # LOGS -> LOG
    Gst.debug_set_threshold_for_name("rtspsrc", Gst.DebugLevel.DEBUG)     # rtspsrc 로그 추가
    Gst.debug_set_threshold_for_name("GST_STATES", Gst.DebugLevel.DEBUG)
    Gst.debug_set_threshold_for_name("GST_PADS", Gst.DebugLevel.DEBUG)
    Gst.debug_set_threshold_for_name("GST_BIN", Gst.DebugLevel.DEBUG)
    '''
    
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <rtsp_uri>\n" % args[0])
        sys.exit(1)
    rtsp_uri = args[1]

    print("Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
        return

    print("Creating Source (rtspsrc)\n")
    source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
    if not source:
        sys.stderr.write(" Unable to create Source (rtspsrc)\n")
        return
    source.set_property("location", rtsp_uri)
    # rtspsrc 속성 설정
    # source.set_property("latency", 200) # 기본값 사용을 위해 주석 처리
    source.set_property("protocols", GstRtsp.RTSPLowerTrans.TCP) # TCP 강제
    # TCP 전용으로 하려면: source.set_property("protocols", GstRtsp.RTSPLowerTrans.TCP)
    # UDP 전용으로 하려면: source.set_property("protocols", GstRtsp.RTSPLowerTrans.UDP)
    # 기본값 (UDP 시도 후 TCP): GstRtsp.RTSPLowerTrans.UDP | GstRtsp.RTSPLowerTrans.TCP (또는 설정 안함)

    print("Creating H264 Depayloader (rtph264depay)\n")
    depay = Gst.ElementFactory.make("rtph264depay", "rtp-h264-depay")
    if not depay:
        sys.stderr.write(" Unable to create H264 depayloader\n")
        return

    print("Creating H264 Parser (h264parse)\n")
    parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not parser:
        sys.stderr.write(" Unable to create H264 parser\n")
        return

    print("Creating Decoder (nvv4l2decoder)\n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder\n")
        return

    print("Creating Stream Muxer (nvstreammux)\n")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
        return
    
    # 개별 속성 설정 대신 설정 파일 사용 - 올바른 속성 이름으로 수정
    #streammux.set_property('config-file-path', "mux_config.txt") # 'config-file' -> 'config-file-path'
    streammux.set_property('width', 1280) # 설정 파일에서 처리
    streammux.set_property('height', 720) # 설정 파일에서 처리
    streammux.set_property('batch-size', 1) # 설정 파일에서 처리
    streammux.set_property('batched-push-timeout', 40000) # 설정 파일에서 처리
    # streammux.set_property('live-source', True) # nvstreammux 7.1.0에는 이 속성이 없으므로 제거 또는 주석 처리

    print("Creating PGIE (nvinfer)\n")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
        return
    pgie.set_property('config-file-path', pgie_config_file)

    print("Creating OSD (nvdsosd)\n")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
        return

    # === Sink를 nveglglessink로 변경 ===
    print("Creating Sink (nveglglessink)\n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")
        return
    # sink.set_property("sync", False) # nveglglessink는 일반적으로 sync=True 또는 기본값 사용

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(depay)
    pipeline.add(parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)    # pgie 추가
    pipeline.add(nvosd)   # nvosd 추가
    pipeline.add(sink)

    print("Linking static elements: depay -> parser -> decoder \n")
    if not depay.link(parser):
        sys.stderr.write("ERROR: Could not link depay -> parser\n")
        return
    if not parser.link(decoder):
        sys.stderr.write("ERROR: Could not link parser -> decoder\n")
        return

    print("Linking decoder -> streammux \n")
    # streammux는 여러 입력을 받을 수 있으므로, sink 패드를 요청해야 함
    # sinkpad = streammux.get_request_pad("sink_0") # Deprecated
    sinkpad = streammux.request_pad_simple("sink_0") 
    
    if not sinkpad:
        sys.stderr.write(" Unable to request the sink pad 'sink_0' from streammux \n") 
        return
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
        if sinkpad: sinkpad.unref()
        return

    link_result = srcpad.link(sinkpad)
    if link_result != Gst.PadLinkReturn.OK:
        sys.stderr.write(f"ERROR: Failed to link decoder to stream muxer. Link result: {link_result}\n")
        # ... (이전 디버깅 코드 유지) ...
        src_caps = srcpad.get_current_caps()
        if src_caps:
            sys.stderr.write(f"  Decoder src pad current caps: {src_caps.to_string()}\n")
            src_caps.unref()
        else:
            src_pad_template = srcpad.get_pad_template()
            if src_pad_template:
                src_template_caps = src_pad_template.get_caps()
                if src_template_caps:
                    sys.stderr.write(f"  Decoder src pad template caps: {src_template_caps.to_string()}\n")
                    src_template_caps.unref()
                else: sys.stderr.write("  Decoder src pad template has no caps.\n")
                src_pad_template.unref()
            else: sys.stderr.write("  Decoder src pad has no template and no current caps.\n")

        sink_pad_template = sinkpad_template
        if sink_pad_template:
            sink_template_caps = sink_pad_template.get_caps()
            if sink_template_caps:
                sys.stderr.write(f"  Streammux sink pad template caps: {sink_template_caps.to_string()}\n")
                sink_template_caps.unref()
            else: sys.stderr.write("  Streammux sink pad template has no caps.\n")
            sink_pad_template.unref()
        else: sys.stderr.write("  Streammux sink pad has no template.\n")
        
        srcpad.unref()
        sinkpad.unref()
        return
    srcpad.unref()
    sinkpad.unref()

    # === streammux -> pgie -> nvosd -> sink 연결 ===
    print("Linking streammux -> pgie -> nvosd -> sink \n")
    if not streammux.link(pgie):
        sys.stderr.write("ERROR: Could not link streammux -> pgie\n")
        return
    if not pgie.link(nvosd):
        sys.stderr.write("ERROR: Could not link pgie -> nvosd\n")
        return
    if not nvosd.link(sink):
        sys.stderr.write("ERROR: Could not link nvosd -> sink\n")
        return

    # rtspsrc의 "pad-added" 시그널 연결
    source.connect("pad-added", on_pad_added, depay)

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))