import sys
import gi
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# 전역 변수: 템플릿 및 ROI 선택과 관련된 변수들
template_img = None     # 선택된 템플릿 이미지
roi = None              # 선택된 ROI 좌표 (x, y, w, h)
roi_selected = False    # ROI가 선택되었는지 여부
drawing = False         # 드래그 중인지 여부
ix, iy = -1, -1         # 시작점 좌표
frame_for_selection = None  # 현재 프레임(ROI 선택용)

def select_roi(event, x, y, flags, param):
    """
    OpenCV 마우스 콜백 함수: ROI(관심 영역)를 선택하기 위한 드래그 이벤트 처리.
    """
    global ix, iy, drawing, roi, roi_selected, template_img, frame_for_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and frame_for_selection is not None:
            frame_copy = frame_for_selection.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi = (x1, y1, x2 - x1, y2 - y1)
        roi_selected = True
        if frame_for_selection is not None:
            template_img = frame_for_selection[y1:y2, x1:x2]
        print("ROI selected:", roi)

def on_new_sample(sink):
    """
    GStreamer 앱싱크의 새로운 샘플 이벤트 콜백 함수.
    프레임을 numpy 배열로 변환한 후, 템플릿이 선택되었다면 템플릿 매칭을 실행합니다.
    """
    global frame_for_selection, template_img

    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # 프레임의 가로, 세로 크기를 추출
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        # 버퍼 데이터를 numpy 배열로 변환
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, np.uint8)
        frame = frame.reshape((height, width, 3))
        buf.unmap(map_info)

        # ROI 선택을 위해 현재 프레임을 저장(마우스 콜백에서 사용)
        frame_for_selection = frame.copy()

        # 템플릿 매칭: 템플릿이 설정되어 있다면 매 프레임에 대해 매칭 수행
        if template_img is not None:
            res = cv2.matchTemplate(frame, template_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            h, w = template_img.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 'q' 키를 누르면 GLib 메인루프 종료
            GLib.MainLoop().quit()

    return Gst.FlowReturn.OK

def main():
    # GStreamer 초기화
    Gst.init(None)

    # OpenCV 창 생성 및 마우스 콜백 설정 (ROI 선택용)
    cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Frame", select_roi)

    # DeepStream/ GStreamer 파이프라인 구성
    # 여기서 filesrc를 사용해 로컬 비디오 파일을 읽어오며, deepstream 플러그인(예: decodebin) 등을 활용합니다.
    pipeline_str = (
        "filesrc location=your_video.mp4 ! decodebin ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink name=sink"
    )
    pipeline = Gst.parse_launch(pipeline_str)

    # appsink 설정: 새로운 샘플이 도착할 때마다 on_new_sample 콜백 호출
    appsink = pipeline.get_by_name("sink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    appsink.connect("new-sample", on_new_sample)

    # 파이프라인 실행
    pipeline.set_state(Gst.State.PLAYING)

    # GLib 메인루프를 실행해서 파이프라인 이벤트 및 콜백을 처리
    loop = GLib.MainLoop()
    try:
        loop.run()
    except Exception as e:
        print("Error:", e)

    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()