import cv2

from ..detector_base import Detector, Detection


class ElectricBikeDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (78, 80, 70), (102, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 150:
                label = "倒伏" if w > h * 1.5 else "正常"
                dets.append(Detection(x, y, x + w, y + h, label, 1.0))
        return dets


def summarize(dets):
    up = sum(1 for d in dets if d.label == "正常")
    down = sum(1 for d in dets if d.label == "倒伏")
    return f"电动车正常{up}辆，倒伏{down}辆", {"upright": up, "fallen": down}
