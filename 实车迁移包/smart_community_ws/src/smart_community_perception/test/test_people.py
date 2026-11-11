"""无需 ROS2/Ultralytics/GPU 即可验证人物后处理逻辑。"""
from types import SimpleNamespace
import numpy as np

from smart_community_perception.detector_base import Detection
from smart_community_perception.detectors.people import PeopleDetector, summarize
from smart_community_perception.people_filter import (
    PeopleFilterConfig, FrameConfirm, filter_people_boxes,
)


def det(x1=100,y1=100,x2=135,y2=235,label='居民',score=.92):
    return Detection(x1,y1,x2,y2,label,score)


def test_only_full_tall_cards_survive():
    boxes = [
        det(),                                     # 完整居民卡
        det(190,100,225,238,'外来人员',.90),         # 完整非社区人员卡
        det(420,220,454,250,score=.98),           # 车，虽然高置信度也拒绝
        det(350,240,363,270,score=.97),           # 过小的背景物件
        det(610,80,640,222,score=.98),            # 靠边被裁的人
        det(80,300,180,452,score=.99),            # 宽大误检
        det(310,100,343,245,score=.20),           # 置信度过低
    ]
    passed, reasons = filter_people_boxes(boxes, (480,640,3))
    assert len(passed) == 2
    assert {x.label for x in passed} == {'居民','外来人员'}
    assert reasons['too_small'] == 2
    assert reasons['touches_image_edge'] == 1
    assert reasons['not_tall_standee'] == 1
    assert reasons['low_confidence'] == 1


def test_temporal_requires_consecutive_frames_and_reset():
    filt = FrameConfirm(min_frames=2)
    assert filt.update([det()]) == []
    confirmed = filt.update([det(105,103,141,239)])
    assert len(confirmed) == 1
    assert len(filt.update([])) == 0
    assert len(filt.update([det(107,103,143,241)])) == 0  # 中间漏帧，计数从 1 重新开始
    assert len(filt.update([det(108,102,144,239)])) == 1
    filt.reset()
    assert filt.update([det()]) == []


def test_wrong_class_does_not_count_as_person():
    passed, reasons = filter_people_boxes([det(label='车辆')], (480,640,3))
    assert not passed and reasons['not_person_label'] == 1


def test_overlap_deduplicate():
    passed, reasons = filter_people_boxes([det(), det(101,101,136,236,score=.81)], (480,640,3))
    assert len(passed) == 1 and reasons['overlapping_duplicate'] == 1


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = np.array([xyxy],dtype=float)
        self.cls = np.array([cls])
        self.conf = np.array([conf])


class FakeYOLO:
    names = {0:'resident',1:'outsider'}
    def predict(self,*_,**kwargs):
        return [SimpleNamespace(boxes=[
            FakeBox((60,100,95,237),0,.94),   # 应保留
            FakeBox((400,240,437,270),0,.96), # 车辆高置信误检
            FakeBox((190,105,224,239),1,.90), # 应保留
        ])]


def test_yolo_detector_publishes_only_confirmed_boxes():
    model = PeopleDetector()
    model.model = FakeYOLO()
    img = np.zeros((480,640,3),dtype=np.uint8)
    assert model.detect(img) == []
    out = model.detect(img)
    summary, result = summarize(out)
    assert result == {'resident':1,'outsider':1,'total':2}
    assert model.last_raw_count == 3
    assert model.last_geometric_count == 2
    assert model.last_rejected['too_small'] == 1
    assert '2' in summary


def test_no_model_returns_empty():
    detector = PeopleDetector()
    assert detector.detect(np.zeros((300,400,3),dtype=np.uint8)) == []
