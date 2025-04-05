#!/usr/bin/env python3

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    return True

def pad_added_handler(src, new_pad, data):
    print(f"\nReceived new pad '{new_pad.get_name()}' from '{src.get_name()}':")
    
    # 새 패드의 캡스 확인
    new_pad_caps = new_pad.get_current_caps()
    new_pad_struct = new_pad_caps.get_structure(0)
    new_pad_type = new_pad_struct.get_name()
    
    print(f"New pad type: {new_pad_type}")

    # h264parse의 sink 패드 가져오기
    h264parse = data
    sink_pad = h264parse.get_static_pad("sink")
    
    # 패드 연결
    if not sink_pad.is_linked():
        ret = new_pad.link(sink_pad)
        if ret == Gst.PadLinkReturn.OK:
            print("Successfully linked new pad")
        else:
            print("Failed to link pads")

def main():
    # GStreamer 초기화
    Gst.init(None)

    # 파이프라인 생성
    pipeline = Gst.Pipeline()

    # 엘리먼트 생성
    source = Gst.ElementFactory.make("filesrc", "file-source")
    demux = Gst.ElementFactory.make("qtdemux", "demuxer")
    h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
    osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    if not all([pipeline, source, demux, h264parse, decoder, streammux, 
                pgie, converter, osd, sink]):
        print("One or more elements could not be created. Exiting.")
        sys.exit(1)

    # 속성 설정
    source.set_property("location", "/home/DeepStream-ROI-Template-Tracking/test_video/sample_720p.mp4")
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    pgie.set_property("config-file-path", "/home/DeepStream-ROI-Template-Tracking/configs/config_infer_primary_project.txt")

    # 파이프라인에 엘리먼트 추가
    pipeline.add(source)
    pipeline.add(demux)
    pipeline.add(h264parse)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(converter)
    pipeline.add(osd)
    pipeline.add(sink)

    # 엘리먼트 연결
    source.link(demux)
    demux.connect("pad-added", pad_added_handler, h264parse)
    h264parse.link(decoder)
    
    # decoder -> streammux 연결
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = decoder.get_static_pad("src")
    if not srcpad.link(sinkpad) == Gst.PadLinkReturn.OK:
        print("Failed to link decoder to streammux")
        sys.exit(1)

    # 나머지 엘리먼트 연결
    streammux.link(pgie)
    pgie.link(converter)
    converter.link(osd)
    osd.link(sink)

    # 메시지 핸들러 설정
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    bus.connect("message", bus_call, loop)

    # 파이프라인 시작
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()