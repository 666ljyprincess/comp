import cv2

from ..detector_base import Detector, Detection


class FireDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 170, 160), (18, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 120:
                dets.append(Detection(x, y, x + w, y + h, "火灾", 1.0))
        return dets


def summarize(dets):
    if not dets:
        return "未发现火灾隐患", {"fire": 0}
    return f"艺术大厦发现火灾隐患{len(dets)}个", {"fire": len(dets)}
