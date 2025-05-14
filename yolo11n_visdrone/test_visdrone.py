#!/usr/bin/env python3

import sys
import os
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

def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    # Get Gst.Buffer from pad
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
    
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
        
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Print detection information
            print(f"Frame {frame_number}: Detected {obj_meta.obj_label} with confidence {obj_meta.confidence:.2f} at "
                  f"({obj_meta.rect_params.left:.1f}, {obj_meta.rect_params.top:.1f}, "
                  f"{obj_meta.rect_params.width:.1f}, {obj_meta.rect_params.height:.1f})")
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def main():
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: python3 test_visdrone.py <video_file>")
        print("Example: python3 test_visdrone.py /home/godsublab/DeepStream-ROI-Template-Tracking/test_video/sample_720p.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    if not os.path.exists(video_file):
        print(f"Error: Video file {video_file} does not exist")
        sys.exit(1)
    
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    
    # Create Pipeline
    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        print("Failed to create pipeline")
        sys.exit(1)
    
    # Create elements
    print("Creating elements")
    # Source element for reading from file
    source = Gst.ElementFactory.make("filesrc", "file-source")
    # Parsing element for parsing mp4 file
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    # Decoder for decoding the video
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    # Stream muxer for batching frames
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    # Primary inference engine
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    # Converter for color space conversion
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    # OSD for drawing bounding boxes
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    # Sink for display
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    
    # For MP4 container
    qtdemux = Gst.ElementFactory.make("qtdemux", "qtdemux")
    
    if not all([source, h264parser, decoder, streammux, pgie, nvvidconv, nvosd, sink, qtdemux]):
        print("One or more elements could not be created. Exiting.")
        sys.exit(1)
    
    # Set properties
    source.set_property("location", video_file)
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    pgie.set_property("config-file-path", "/home/nvidia/DeepStream-ROI-Template-Tracking/yolo11n_visdrone/config_infer_visdrone.txt")
    
    # Add elements to pipeline
    print("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(qtdemux)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    
    # Link elements
    print("Linking elements in the Pipeline")
    source.link(qtdemux)
    
    # Connect demuxer to parser
    qtdemux.connect("pad-added", lambda src, pad: pad.link(h264parser.get_static_pad("sink")))
    
    # Link parser -> decoder
    h264parser.link(decoder)
    
    # Get sink pad of streammux
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        print("Failed to get sink pad of streammux")
        sys.exit(1)
    
    # Get source pad of decoder
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        print("Failed to get source pad of decoder")
        sys.exit(1)
    
    # Link decoder -> streammux
    srcpad.link(sinkpad)
    
    # Link the remaining elements
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)
    
    # Add probe to get detection results
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        print("Unable to get sink pad of nvosd")
        sys.exit(1)
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start play back and listen to events
    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")

if __name__ == "__main__":
    main() 