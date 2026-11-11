import cv2
import numpy as np

from ..detector_base import Detector, Detection


class MeterDetector(Detector):
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(cv2.Canny(gray, 70, 160), 1, np.pi / 180, 55,
                                minLineLength=30, maxLineGap=8)
        best = None
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                length = float(np.hypot(x2 - x1, y2 - y1))
                if best is None or length > best[0]:
                    best = (length, x1, y1, x2, y2)
        if best is None:
            value = -1.0
        else:
            _, x1, y1, x2, y2 = best
            angle = float(np.degrees(np.arctan2(-(y2 - y1), x2 - x1)))
            if angle < 0:
                angle += 180.0
            value = max(0.0, min(1.0, angle / 180.0))
        return [Detection(0, 0, image.shape[1], image.shape[0], "meter", 1.0,
                          {"value": value})]


def summarize(dets):
    if not dets:
        return "未识别到仪表", {"meter": -1.0}
    v = dets[0].extra["value"]
    if v < 0:
        return "未识别到仪表", {"meter": v}
    return f"仪表读数{v:.2f}", {"meter": v}
