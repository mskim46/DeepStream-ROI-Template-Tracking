#!/usr/bin/env python3

import sys
# sys.path.append('../') # pyds가 표준 경로에 설치되므로 이 줄은 필요 없을 수 있습니다.
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
import time

# 대상 객체 ID
TARGET_OBJECT_ID = 0

# 전역 변수 (필요시 추가)
loop = None

def bus_call(bus, message, loop):
    """파이프라인 메시지 콜백 함수"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    # ... (기존 osd_sink_pad_buffer_probe 함수 내용은 동일) ...
    """OSD 싱크 패드 프로브 함수"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # 배치 메타데이터 가져오기
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
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
                # 객체 메타데이터 가져오기
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # object_id 확인
            current_object_id = obj_meta.object_id
            # print(f"Frame {frame_number}: Detected object ID {current_object_id}") # 디버깅 필요시 주석 해제

            if current_object_id == TARGET_OBJECT_ID:
                print(f"--- Found Target Object (ID: {TARGET_OBJECT_ID}) in Frame {frame_number} ---")

                # 대상 객체에 대한 처리 수행
                # 예: 바운딩 박스 색상 변경 또는 특정 정보 표시
                rect_params = obj_meta.rect_params
                rect_params.border_color.set(0.0, 1.0, 1.0, 1.0) # 청록색 (Cyan)으로 변경
                rect_params.has_bg_color = 1
                rect_params.bg_color.set(0.0, 1.0, 1.0, 0.4) # 반투명 청록색 배경

                # OSD 텍스트 수정 (선택 사항)
                txt_params = obj_meta.text_params
                if txt_params.display_text: # 기존 텍스트가 있으면 추가
                     txt_params.display_text = f"TARGET {TARGET_OBJECT_ID}: {txt_params.display_text}"
                else:
                     txt_params.display_text = f"TARGET {TARGET_OBJECT_ID}"

                # 추가적인 로직 (예: 좌표 로깅, 이벤트 발생 등)
                top = int(rect_params.top)
                left = int(rect_params.left)
                width = int(rect_params.width)
                height = int(rect_params.height)
                print(f"Target Object Location: ({left}, {top}, {width}, {height})")

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
    global loop
    # GStreamer 초기화
    GObject.threads_init()
    Gst.init(None)

    # DeepSORT 설정 파일 경로
    tracker_config_path = "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml"
    # 추론 설정 파일 경로
    infer_config_path = "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_infer_primary.txt"
    # 비디오 파일 경로
    video_file_path = "/home/DeepStream-ROI-Template-Tracking/test_video/sample_720p.mp4"
    # 트래커 라이브러리 경로
    tracker_lib_path = "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so"

    # 파이프라인 생성 (명시적 링크 방식)
    pipeline_str = (
        # 파일 소스 및 디먹서 (이름 지정)
        f"filesrc location={video_file_path} ! qtdemux name=demux "
        # 스트림 먹서 정의 (이름 지정)
        f"nvstreammux name=m batch-size=1 width=1280 height=720 "
        # 비디오 처리 브랜치: demux의 video_0 패드 -> 큐 -> 파서 -> 디코더 -> 큐 -> 먹서의 sink_0 패드
        f"demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! queue ! m.sink_0 "
        # 메인 처리 브랜치: 먹서의 src 패드 -> 큐 -> 추론 -> 트래커 -> 변환 -> OSD -> 싱크
        f"m.src ! queue ! "
        f"nvinfer config-file-path={infer_config_path} ! "
        f"nvtracker ll-lib-file={tracker_lib_path} ll-config-file={tracker_config_path} tracker-width=640 tracker-height=384 ! "
        # 마지막 부분을 fakesink로 변경
        f"nvvideoconvert ! nvdsosd name=nvosd ! fakesink"
    )

    print("Using pipeline: \n", pipeline_str)
    pipeline = Gst.parse_launch(pipeline_str)
    if not pipeline:
        sys.stderr.write(" Pipeline creation failed \n")
        sys.exit(1)

    # OSD 엘리먼트 가져오기 (이름 사용)
    osd = pipeline.get_by_name("nvosd")
    if not osd:
        sys.stderr.write(" Unable to get OSD 'nvosd' \n")
        sys.exit(1)

    # OSD 싱크 패드에 프로브 추가
    osdsinkpad = osd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of OSD \n")
        sys.exit(1)
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # 메인 루프 생성 및 실행
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop) # 메시지 핸들러 연결

    print("Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt: # Ctrl+C 처리
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 정리
        print("Exiting app\n")
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main())