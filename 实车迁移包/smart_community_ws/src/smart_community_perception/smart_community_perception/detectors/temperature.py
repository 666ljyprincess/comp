import cv2

from ..detector_base import Detector, Detection


class TemperatureDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 180, 180), (8, 255, 255)) | \
              cv2.inRange(hsv, (175, 180, 180), (180, 255, 255))
        state = "ABNORMAL" if cv2.countNonZero(red) > 180 else "NORMAL"
        return [Detection(0, 0, image.shape[1], image.shape[0], state, 1.0)]


def summarize(dets):
    state = dets[0].label if dets else "NORMAL"
    zh = {"ABNORMAL": "检测到异常高温", "NORMAL": "温度正常"}
    return zh[state], {"temperature": state}
