# ... existing code ...
import sys
sys.path.append("../")
from common.bus_call import bus_call
from common.platform_info import PlatformInfo # Keep PlatformInfo if used elsewhere, or remove if only for encoder properties
import pyds
import platform
import math
import time
from ctypes import *
import gi
gi.require_version("Gst", "1.0")
# gi.require_version("GstRtspServer", "1.0") # REMOVE: RTSP Server not needed
from gi.repository import Gst, GLib # REMOVE: GstRtspServer from here
import configparser
import datetime

import argparse

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
# pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"] # OLD VALUE
# Read labels from labels.txt
try:
    with open("labels.txt", "r") as f:
        pgie_classes_str = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    sys.stderr.write("Error: labels.txt not found. Using default classes.\n")
    pgie_classes_str = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10"] # Fallback

# pgie_src_pad_buffer_probe function remains unchanged
# ... existing pgie_src_pad_buffer_probe code ...

# cb_newpad function remains unchanged
# ... existing cb_newpad code ...

# decodebin_child_added function remains unchanged
# ... existing decodebin_child_added code ...

# create_source_bin function remains unchanged
# ... existing create_source_bin code ...

def pgie_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        print(
            "Frame Number=",
            frame_number
        )
        if ts_from_rtsp:
            ts = frame_meta.ntp_timestamp/1000000000 # Retrieve timestamp, put decimal in proper position for Unix format
            print("RTSP Timestamp:",datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) # Convert timestamp to UTC

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if ts_from_rtsp:
        if name.find("source") != -1:
            pyds.configure_source_for_ntp_sync(hash(Object))


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    number_sources = len(args)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    print("Creating Pgie \n ")
    if gie=="nvinfer":
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if platform_info.is_integrated_gpu():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        #encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")

    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    
    if ts_from_rtsp:
        streammux.set_property("attach-sys-ts", 0)

    if gie=="nvinfer":
        pgie.set_property("config-file-path", "config_infer_deep_yolo11.txt")
    else:
        pgie.set_property("config-file-path", "config_infer_deep_yolo11_inferserver.txt")


    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)

    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad=pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print(
        "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
        % rtsp_port_num
    )

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-g", "--gie", default="nvinfer",
                  help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer','nvinferserver'])
    parser.add_argument("-c", "--codec", default="H264",
                  help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=False, dest='rtsp_ts', help="Attach NTP timestamp from RTSP source",
    )
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global gie
    global ts_from_rtsp
    gie = args.gie
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    return stream_path

def main(args):
    # Check input arguments
    number_sources = len(args)

    platform_info = PlatformInfo() # Keep if needed for other parts, or if you might add encoder back later
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False # This is determined by input URI, keep it

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
            # Clean up and exit
            return
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
            # Clean up and exit
            return
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            sys.stderr.write(f"Failed to link {srcpad.get_name()} to {sinkpad.get_name()}\n")
            # Clean up and exit
            return


    print("Creating Pgie \n ")
    if gie=="nvinfer": # gie is a global variable from parse_args
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # REMOVE: nvvidconv_postosd and caps for encoder
    # nvvidconv_postosd = Gst.ElementFactory.make(
    #     "nvvideoconvert", "convertor_postosd")
    # if not nvvidconv_postosd:
    #     sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    #
    # # Create a caps filter
    # caps = Gst.ElementFactory.make("capsfilter", "filter")
    # caps.set_property(
    #     "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    # )

    # REMOVE: Encoder, rtppay, and udpsink elements
    # # Make the encoder
    # if codec == "H264": # codec is a global variable from parse_args
    #     encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    #     print("Creating H264 Encoder")
    # elif codec == "H265":
    #     encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    #     print("Creating H265 Encoder")
    # if not encoder:
    #     sys.stderr.write(" Unable to create encoder")
    # encoder.set_property("bitrate", bitrate) # bitrate is a global variable
    # if platform_info.is_integrated_gpu():
    #     encoder.set_property("preset-level", 1)
    #     encoder.set_property("insert-sps-pps", 1)
    #     #encoder.set_property("bufapi-version", 1)
    #
    # # Make the payload-encode video into RTP packets
    # if codec == "H264":
    #     rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    #     print("Creating H264 rtppay")
    # elif codec == "H265":
    #     rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
    #     print("Creating H265 rtppay")
    # if not rtppay:
    #     sys.stderr.write(" Unable to create rtppay")
    #
    # # Make the UDP sink
    # updsink_port_num = 5400 # This will be removed
    # sink = Gst.ElementFactory.make("udpsink", "udpsink")
    # if not sink:
    #     sys.stderr.write(" Unable to create udpsink")
    #
    # sink.set_property("host", "224.224.255.255")
    # sink.set_property("port", updsink_port_num)
    # sink.set_property("async", False)
    # sink.set_property("sync", 1)

    # ADD: Create a local display sink (e.g., nveglglessink)
    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")
    # Optionally, set properties for the sink, e.g., window size, sync
    # sink.set_property("window-x", 0)
    # sink.set_property("window-y", 0)
    # sink.set_property("window-width", TILED_OUTPUT_WIDTH) # Or MUXER_OUTPUT_WIDTH if tiler is not used before sink
    # sink.set_property("window-height", TILED_OUTPUT_HEIGHT) # Or MUXER_OUTPUT_HEIGHT
    sink.set_property("sync", 1) # Or 0 for lower latency with live sources
    sink.set_property("async", False)


    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    
    if ts_from_rtsp: # ts_from_rtsp is a global variable
        streammux.set_property("attach-sys-ts", False) # Original was 0, which is False. Set to False for clarity.

    if gie=="nvinfer":
        pgie.set_property("config-file-path", "config_infer_deep_yolo11.txt")
    else:
        pgie.set_property("config-file-path", "config_infer_deep_yolo11_inferserver.txt")


    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    # sink.set_property("qos", 0) # This was for udpsink, nveglglessink might not have/need it

    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    # REMOVE: pipeline.add for nvvidconv_postosd, caps, encoder, rtppay
    # pipeline.add(nvvidconv_postosd)
    # pipeline.add(caps)
    # pipeline.add(encoder)
    # pipeline.add(rtppay)
    pipeline.add(sink) # Add the new sink

    # Link elements
    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvosd)
    # MODIFY: Link nvosd to the new sink
    nvosd.link(sink)
    # REMOVE: Old linking for encoder chain
    # nvosd.link(nvvidconv_postosd)
    # nvvidconv_postosd.link(caps)
    # caps.link(encoder)
    # encoder.link(rtppay)
    # rtppay.link(sink) # This sink was udpsink

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop) # bus_call from common.bus_call

    pgie_src_pad=pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
        # pyds.nvds_add_pad_probe(pgie_src_pad, Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0) # Alternative way

    # REMOVE: RTSP Server setup
    # # Start streaming
    # rtsp_port_num = 8554
    #
    # server = GstRtspServer.RTSPServer.new()
    # server.props.service = "%d" % rtsp_port_num
    # server.attach(None)
    #
    # factory = GstRtspServer.RTSPMediaFactory.new()
    # factory.set_launch(
    #     '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
    #     % (updsink_port_num, codec)
    # )
    # factory.set_shared(True)
    # server.get_mount_points().add_factory("/ds-test", factory)
    #
    # print(
    #     "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
    #     % rtsp_port_num
    # )

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ') # Can update description
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream or RTSP URI", nargs="+", default=["a"], required=True) # Keep input
    parser.add_argument("-g", "--gie", default="nvinfer",
                  help="choose GPU inference engine type nvinfer or nvinferserver , default=nvinfer", choices=['nvinfer','nvinferserver']) # Keep GIE selection
    # REMOVE: Arguments for RTSP output codec and bitrate
    # parser.add_argument("-c", "--codec", default="H264",
    #               help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    # parser.add_argument("-b", "--bitrate", default=4000000,
    #               help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=False, dest='rtsp_ts', help="Attach NTP timestamp from RTSP source",
    ) # Keep RTSP timestamp option
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # REMOVE: Global variables for codec and bitrate
    # global codec
    # global bitrate
    global stream_path
    global gie
    global ts_from_rtsp # Keep this

    gie = args.gie
    # codec = args.codec # REMOVE
    # bitrate = args.bitrate # REMOVE
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    return stream_path

if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
# ... existing code ...
