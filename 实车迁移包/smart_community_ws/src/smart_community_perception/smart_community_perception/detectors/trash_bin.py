import cv2

from ..detector_base import Detector, Detection

CATS = {"blue": "可回收", "green": "厨余", "red": "有害"}


class TrashBinDetector(Detector):
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        blue = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
        green = cv2.inRange(hsv, (40, 90, 50), (90, 255, 255))
        red = cv2.inRange(hsv, (0, 140, 100), (10, 255, 255)) | \
              cv2.inRange(hsv, (170, 140, 100), (180, 255, 255))
        counts = {"可回收": cv2.countNonZero(blue), "厨余": cv2.countNonZero(green),
                  "有害": cv2.countNonZero(red)}
        typ = max(counts, key=counts.get)
        if counts[typ] < 150:
            return []
        # 开盖判断：顶部区域是否有红色盖
        lid = red[:max(1, h // 5), :]
        state = "打开" if cv2.countNonZero(lid) > 80 else "关闭"
        # 投放判断：桶身区域内，除桶色外的异色块即垃圾；垃圾色 != 桶色 → 投放错误
        body = slice(max(1, h // 5), h)
        body_masks = {"可回收": blue[body], "厨余": green[body], "有害": red[body]}
        garbage = None
        for cat, m in body_masks.items():
            if cat == typ:
                continue
            if cv2.countNonZero(m) > 60:
                garbage = cat
                break
        disposal = "错误" if garbage else "正确"
        return [Detection(0, 0, w, h, typ, 1.0,
                          {"state": state, "garbage": garbage or "无", "disposal": disposal})]


def summarize(dets):
    if not dets:
        return "未识别到垃圾桶", {"bin": "none"}
    d = dets[0]
    e = d.extra
    s = f"{d.label}垃圾桶状态为{e['state']}，垃圾为{e['garbage']}，投放{e['disposal']}"
    return s, {"bin": d.label, "state": e["state"], "garbage": e["garbage"], "disposal": e["disposal"]}
