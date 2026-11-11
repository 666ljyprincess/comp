"""基于自定义两类别 YOLO 的社区人偶识别。

通过高宽比、最小像素尺寸、画面边缘和多帧确认抑制车牌/车辆等误检。
两类别：resident / outsider。无法严格证明全身完整时，不应声称绝对准确。
"""
from ..detector_base import Detector, Detection
from ..people_filter import PeopleFilterConfig, FrameConfirm, filter_people_boxes


class PeopleDetector(Detector):
    def __init__(self):
        self.model = None
        self.weights = ''
        self.conf = 0.60
        self.imgsz = 640
        self.filter_config = PeopleFilterConfig()
        self.confirm = FrameConfirm(2)
        self.last_rejected = {}
        self.last_raw_count = 0
        self.last_geometric_count = 0
        self.error = 'MODEL_MISSING'

    def reset_temporal(self):
        self.confirm.reset()

    def configure(self, weights, conf=0.60, imgsz=640, *, min_height=45,
                  min_aspect=1.90, max_aspect=6.00, edge_margin=3,
                  confirm_frames=2):
        import os
        weights = os.path.abspath(os.path.expanduser(str(weights or '').strip())) if weights else ''
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        if not (0 < self.conf <= 1):
            raise ValueError('people_confidence 必须在 (0, 1]')
        if not (0 < float(min_aspect) < float(max_aspect)):
            raise ValueError('people_min_aspect 必须小于 people_max_aspect')
        cfg = PeopleFilterConfig(confidence=self.conf,
                                 min_height=max(1,int(min_height)),
                                 min_aspect=float(min_aspect), max_aspect=float(max_aspect),
                                 edge_margin=max(0,int(edge_margin)),
                                 confirm_frames=max(1,int(confirm_frames)))
        if self.confirm.min_frames != cfg.confirm_frames:
            self.confirm = FrameConfirm(cfg.confirm_frames)
        self.filter_config = cfg
        if weights == self.weights and self.model is not None:
            return
        self.weights = weights
        self.model = None
        self.confirm.reset()
        if not weights or not os.path.isfile(weights):
            self.error = 'MODEL_MISSING: ' + (weights or '(empty people_weights)')
            return
        try:
            from ultralytics import YOLO
            model = YOLO(weights)
            names = model.names
            values = [str(names[i]).lower().strip() for i in sorted(names)] if isinstance(names,dict) else [str(v).lower().strip() for v in names]
            allowed = ({'resident', 'outsider'}, {'居民', '外来人员'})
            if len(values) != 2 or set(values) not in allowed:
                self.error = f'WRONG_CLASSES: expected resident/outsider, got {values}'
                return
            self.model = model
            self.error = ''
        except Exception as exc:
            self.error = f'LOAD_ERROR: {exc}'

    def detect(self, image):
        self.last_rejected = {}
        self.last_raw_count = 0
        self.last_geometric_count = 0
        if self.model is None:
            return []
        try:
            outputs = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                         verbose=False, device='cpu')
            raw = []
            name_map = {'resident': '居民', 'outsider': '外来人员',
                        '居民': '居民', '外来人员': '外来人员'}
            for result in outputs:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = str(self.model.names[cls]).lower().strip()
                    if name not in name_map:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw.append(Detection(x1, y1, x2, y2, name_map[name], float(box.conf[0])))
            self.last_raw_count = len(raw)
            cleaned, self.last_rejected = filter_people_boxes(raw, image.shape, self.filter_config)
            self.last_geometric_count = len(cleaned)
            accepted = self.confirm.update(cleaned)
            self.error = ''
            return accepted
        except Exception as exc:
            self.confirm.reset()
            self.error = f'INFERENCE_ERROR: {exc}'
            return []


def summarize(detections):
    resident = sum(d.label == '居民' for d in detections)
    outsider = sum(d.label == '外来人员' for d in detections)
    return (f'当前画面确认{resident + outsider}个完整候选人偶（非区域累计）',
            {'resident': resident, 'outsider': outsider, 'total': resident + outsider})
