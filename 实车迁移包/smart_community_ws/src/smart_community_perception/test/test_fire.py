import numpy as np

from smart_community_perception.detectors.fire import FireDetector, summarize
from smart_community_perception.detector_base import Detection


def _fire(n=3):
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    for i in range(n):
        x = 20 + i * 90
        img[60:120, x:x + 40] = (0, 51, 255)  # 橙色 BGR
    return img


def test_counts_fire():
    d = FireDetector().detect(_fire(3))
    assert len(d) == 3
    assert all(x.label == "火灾" for x in d)


def test_empty_returns_zero():
    d = FireDetector().detect(np.zeros((200, 300, 3), dtype=np.uint8))
    assert d == []


def test_summarize_counts():
    dets = [Detection(0, 0, 10, 10, "火灾") for _ in range(3)]
    s, p = summarize(dets)
    assert "3" in s and p["fire"] == 3
