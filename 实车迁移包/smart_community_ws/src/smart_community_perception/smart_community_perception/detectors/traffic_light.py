"""Traffic light detector for the smart-community Gazebo task.

Key points:
* NEVER decide from the number of red/green pixels in the whole picture.
* Preferred mode: set a normalized ROI covering the THREE lamp lenses. Its
  thirds are scored by the amount of BRIGHT, correctly colored pixels.
* Without a configured ROI a conservative color-blob fallback is used.
* Ambiguous or absent signals are UNKNOWN (fail-closed in the mission manager).

The on/off texture appearance depends on the actual Gazebo materials. Calibrate
ROI and thresholds against real /camera/image_raw frames before competition.
"""

import cv2
import numpy as np

from ..detector_base import Detector, Detection


class TrafficLightDetector(Detector):
    COLORS = ('RED', 'YELLOW', 'GREEN')

    def __init__(self, roi=None, order='AUTO'):
        self.roi = None
        self.order = 'AUTO'
        self.set_roi(roi)
        self.set_order(order)

    def set_roi(self, roi):
        """Pass None/'' for automatic mode, otherwise (x1,y1,x2,y2) in [0,1]."""
        if roi is None or roi == '':
            self.roi = None
            return
        if isinstance(roi, str):
            values = [float(t.strip()) for t in roi.split(',')]
        else:
            values = [float(t) for t in roi]
        if len(values) != 4:
            raise ValueError('traffic_light_roi requires x1,y1,x2,y2')
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('traffic_light_roi must be normalized (0..1), x1<x2 and y1<y2')
        self.roi = tuple(values)

    def set_order(self, order):
        order = str(order).upper().strip()
        if order not in ('AUTO', 'RYG', 'GYR'):
            raise ValueError('traffic_light_order must be AUTO, RYG or GYR')
        self.order = order

    @staticmethod
    def _color_mask(hsv, label, min_v=155):
        if label == 'RED':
            return cv2.inRange(hsv, (0, 105, min_v), (12, 255, 255)) | \
                   cv2.inRange(hsv, (170, 105, min_v), (179, 255, 255))
        if label == 'YELLOW':
            return cv2.inRange(hsv, (17, 100, min_v), (38, 255, 255))
        return cv2.inRange(hsv, (40, 85, min_v), (95, 255, 255))

    @staticmethod
    def _unknown(rect, scores=None):
        x1, y1, x2, y2 = rect
        return [Detection(x1, y1, x2, y2, 'UNKNOWN', 0.0,
                          extra={'scores': scores or {}, 'mode': 'unknown'})]

    def detect(self, image):
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        if self.roi is None:
            return self._detect_auto(image)
        x1, y1, x2, y2 = self.roi
        rect = (int(x1 * width), int(y1 * height),
                max(int(x1 * width) + 1, int(x2 * width)),
                max(int(y1 * height) + 1, int(y2 * height)))
        return self._detect_lamp_row(image, rect)

    def _detect_lamp_row(self, image, rect):
        """Recommended: the ROI encloses all three lamp lenses, not the whole road."""
        x1, y1, x2, y2 = rect
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if h < 12 or w < 36:
            return self._unknown(rect)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        # Search both mirror orders when AUTO is used (the world has two lights).
        order_map = {'R': 'RED', 'Y': 'YELLOW', 'G': 'GREEN'}
        trials = []
        for order_str in (('RYG', 'GYR') if self.order == 'AUTO' else (self.order,)):
            scores, boxes = {}, {}
            for i, name in enumerate(order_str):
                label = order_map[name]
                left = int(i * w / 3)
                right = int((i + 1) * w / 3)
                sub = hsv[:, left:right]
                if sub.size == 0:
                    return self._unknown(rect)
                sh, sw = sub.shape[:2]
                xx, yy = np.ogrid[:sh, :sw]
                ry = max(sh * 0.43, 1.0)
                rx = max(sw * 0.43, 1.0)
                lens = ((xx - (sw - 1) / 2.0) / rx) ** 2 + \
                       ((yy - (sh - 1) / 2.0) / ry) ** 2 <= 1.0
                colored = self._color_mask(sub, label, min_v=155) > 0
                selected = lens & colored
                vv = v[:, left:right]
                strength = np.maximum(vv[selected] - 155.0, 0).sum() / (100.0 * max(1, int(lens.sum())))
                scores[label] = float(strength)
                boxes[label] = (x1 + left, y1, x1 + right, y2)
            ranking = sorted(scores, key=scores.get, reverse=True)
            best, second = ranking[0], ranking[1]
            trials.append((scores[best], scores[second], best, boxes[best], scores))
        bs, ss, best, bx, scores = max(trials, key=lambda t: (t[0], t[0] - t[1]))
        if bs < 0.038 or bs < ss * 1.30 or bs - ss < 0.018:
            return self._unknown(rect, scores)
        confidence = min(1.0, 0.55 + 0.45 * (bs - ss) / max(bs, 1e-6))
        return [Detection(*bx, best, float(confidence), extra={'scores': scores, 'mode': 'roi'})]

    def _detect_auto(self, image):
        """Conservative fallback if ROI is not yet calibrated.

        Picks a luminous near-round color blob, rejecting big background regions.
        A fixed ROI over the actual signal is substantially more reliable.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        frame_area = w * h
        min_area = max(7.0, frame_area * 0.000010)
        max_area = frame_area * 0.018
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        found = []
        for label in self.COLORS:
            binary = self._color_mask(hsv, label, min_v=180)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if not (min_area <= area <= max_area):
                    continue
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < 4 or ch < 4 or not (0.55 <= cw / ch <= 1.8):
                    continue
                if area / float(cw * ch) < 0.35:
                    continue
                mask = np.zeros((ch, cw), np.uint8)
                cv2.drawContours(mask, [cnt - [x, y]], -1, 255, -1)
                mean_v = cv2.mean(value[y:y + ch, x:x + cw], mask=mask)[0]
                margin = max(cw, ch)
                sx1, sy1 = max(0, x - margin), max(0, y - margin)
                sx2, sy2 = min(w, x + cw + margin), min(h, y + ch + margin)
                surround = value[sy1:sy2, sx1:sx2]
                if surround.size == 0:
                    continue
                # Surrounding background should be darker than the illuminated bulb.
                context_v = float(np.median(surround))
                contrast = mean_v - context_v
                if contrast < 35:
                    continue
                score = (mean_v - 150.0) * np.sqrt(area) * max(0.0, contrast) / 100.0
                found.append((float(score), label, (x, y, x + cw, y + ch)))

        found.sort(reverse=True)
        if not found:
            return self._unknown((0, 0, w, h))
        winner = found[0]
        # Duplicate blobs of the same color are okay. Conflicting colors are not.
        competitor = next((item for item in found[1:] if item[1] != winner[1]), None)
        score2 = competitor[0] if competitor else 0.0
        if competitor and winner[0] < 1.30 * score2:
            return self._unknown(winner[2])
        confidence = min(1.0, 0.55 + 0.45 * (winner[0] - score2) / max(1e-6, winner[0]))
        return [Detection(*winner[2], winner[1], float(confidence),
                          extra={'mode': 'auto', 'strength': round(winner[0], 3)})]


def summarize(detections):
    d = detections[0] if detections else None
    state = d.label if d else 'UNKNOWN'
    if state not in ('RED', 'YELLOW', 'GREEN'):
        state = 'UNKNOWN'
    zh = {
        'RED': '前方红灯，请等待',
        'YELLOW': '黄灯，请等待',
        'GREEN': '检测到绿灯，请确认后通行',
        'UNKNOWN': '信号灯状态不明确，请等待',
    }
    payload = {
        'state': state,
        'confidence': round(float(d.score), 3) if d else 0.0,
    }
    if d and d.extra:
        payload['mode'] = d.extra.get('mode', 'unknown')
        if d.extra.get('scores'):
            payload['scores'] = {k: round(float(v), 3) for k, v in d.extra['scores'].items()}
    return zh[state], payload
