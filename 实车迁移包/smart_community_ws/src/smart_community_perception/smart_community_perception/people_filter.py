"""YOLO 人偶检测后处理。纯 Python，方便独立单元测试。

限制：几何、画面边界和连续帧过滤是工程启发式规则，不能从单一
目标框证明人偶的头脚确实完整，也不能根治模型本身的分类错误。
"""
from dataclasses import dataclass
import math


@dataclass
class PeopleFilterConfig:
    confidence: float = 0.60
    min_height: int = 45
    min_width: int = 10
    min_aspect: float = 1.90   # 人物立牌 15 cm 高 / 5 cm 宽；检测框不是严格的物理尺寸
    max_aspect: float = 6.00
    edge_margin: int = 3
    confirm_frames: int = 2


def filter_people_boxes(detections, image_shape, cfg=None):
    """返回 (保留的 Detection, 过滤原因字典)。仅考虑已由 YOLO 判为人的框。"""
    cfg = cfg or PeopleFilterConfig()
    h, w = int(image_shape[0]), int(image_shape[1])
    if h < 1 or w < 1:
        raise ValueError('图像尺寸无效')
    selected = []
    rejected = {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for d in detections:
        if d.label not in ('居民', '外来人员'):
            reject('not_person_label')
            continue
        x1, y1, x2, y2 = map(float, (d.x1, d.y1, d.x2, d.y2))
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            reject('invalid_box')
            continue
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            reject('invalid_box')
        elif float(d.score) < cfg.confidence:
            reject('low_confidence')
        elif bh < cfg.min_height or bw < cfg.min_width:
            reject('too_small')
        elif not cfg.min_aspect <= bh / bw <= cfg.max_aspect:
            reject('not_tall_standee')
        elif (x1 <= cfg.edge_margin or y1 <= cfg.edge_margin or
              x2 >= w - cfg.edge_margin or y2 >= h - cfg.edge_margin):
            reject('touches_image_edge')
        else:
            selected.append(d)

    # 少数情况下 YOLO 不同类别在相同人偶上给出重叠框，只保留更可信的。
    selected.sort(key=lambda b: float(b.score), reverse=True)
    unique = []
    for d in selected:
        if any(_iou(d, kept) >= .65 for kept in unique):
            reject('overlapping_duplicate')
            continue
        unique.append(d)
    return unique, rejected


def _iou(a, b):
    ix1, iy1 = max(a.x1,b.x1), max(a.y1,b.y1)
    ix2, iy2 = min(a.x2,b.x2), min(a.y2,b.y2)
    inter = max(0.,ix2-ix1) * max(0.,iy2-iy1)
    area_a = max(0.,a.x2-a.x1) * max(0.,a.y2-a.y1)
    area_b = max(0.,b.x2-b.x1) * max(0.,b.y2-b.y1)
    return inter / max(1e-6, area_a + area_b - inter)


class FrameConfirm:
    """连续多帧观察到相近位置、相同类别的检测框，才公布为有效人偶。

    这不是跨站点去重。跨位置的唯一人物累计仍由原有 people_fusion.py 完成。
    """
    def __init__(self, min_frames=2, max_missed=1):
        self.min_frames = max(1, int(min_frames))
        self.max_missed = int(max_missed)
        self.reset()

    def reset(self):
        self.tracks = []
        self.frame = 0

    @staticmethod
    def _match_distance(a, b):
        if a.label != b.label:
            return None
        aw, ah = a.x2-a.x1, a.y2-a.y1
        bw, bh = b.x2-b.x1, b.y2-b.y1
        if min(aw, ah, bw, bh) <= 0:
            return None
        if not (.55 <= bw/aw <= 1.85 and .65 <= bh/ah <= 1.55):
            return None
        dx = abs((a.x1+a.x2)-(b.x1+b.x2)) / (aw + bw)
        dy = abs((a.y1+a.y2)-(b.y1+b.y2)) / (ah + bh)
        if dx > .75 or dy > .35:
            return None
        return dx + dy

    def update(self, detections):
        self.frame += 1
        # 先挑选候选与已有轨迹的最优匹配，防止同帧多个对象配到同一人。
        pairs = []
        for ti,t in enumerate(self.tracks):
            for di,d in enumerate(detections):
                dist = self._match_distance(t['box'], d)
                if dist is not None:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_tracks, used_detections = set(), set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_detections:
                continue
            t = self.tracks[ti]
            if t['last'] == self.frame - 1:
                t['hits'] += 1
            else:
                t['hits'] = 1  # 必须连续，不把零散误识别凑成稳定目标
            t['box'] = detections[di]
            t['last'] = self.frame
            used_tracks.add(ti)
            used_detections.add(di)
        for di, d in enumerate(detections):
            if di not in used_detections:
                self.tracks.append({'box':d, 'hits':1, 'last':self.frame})
        self.tracks = [t for t in self.tracks if self.frame - t['last'] <= self.max_missed]
        return [t['box'] for t in self.tracks
                if t['last'] == self.frame and t['hits'] >= self.min_frames]
