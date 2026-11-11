import numpy as np
from smart_community_perception.annotator import annotate
from smart_community_perception.detector_base import Detection


def test_annotate_draws_red_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = annotate(img, [Detection(10, 10, 50, 50, "person")], "found 1")
    assert out[10:12, 10:50].sum() > 0


def test_annotate_returns_new_array():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = annotate(img, [], "")
    assert out is not img
