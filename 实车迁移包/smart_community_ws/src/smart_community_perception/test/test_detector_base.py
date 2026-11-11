import numpy as np
from smart_community_perception.detector_base import Detection, Detector


def test_detection_fields():
    d = Detection(0, 1, 2, 3, "person", 0.9, {"count": 4})
    assert d.x1 == 0 and d.y2 == 3 and d.label == "person"
    assert d.extra["count"] == 4


def test_detector_is_abstract():
    try:
        Detector()
        raised = False
    except TypeError:
        raised = True
    assert raised
