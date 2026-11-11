import cv2

from ..detector_base import Detector, Detection


class SignDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 120, 100), (10, 255, 255)) | \
              cv2.inRange(hsv, (170, 120, 100), (180, 255, 255))
        if cv2.countNonZero(red) < 200:
            return []
        x, y, w, h = cv2.boundingRect(red)
        return [Detection(x, y, x + w, y + h, "禁止直行", 1.0)]


def summarize(dets):
    if not dets:
        return "未识别到指示牌", {"sign": "none"}
    return "前方是禁止直行指示牌", {"sign": "禁止直行"}
