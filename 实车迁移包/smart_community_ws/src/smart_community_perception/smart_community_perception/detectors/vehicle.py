import cv2

from ..detector_base import Detector, Detection


class VehicleDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 80, 60), (125, 255, 255))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 350 and w > h * 1.1:
                dets.append(Detection(x, y, x + w, y + h, "车辆", 1.0))
        return dets


def summarize(dets):
    n = len(dets)
    return f"检测到{n}辆车", {"vehicle": n}
