import numpy as np

from smart_community_perception.detectors.plate import PlateDetector, summarize
from smart_community_perception.detector_base import Detection


def _plate():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[80:110, 90:210] = (255, 255, 255)  # 白色车牌 120x30
    return img


def test_detect_finds_plate():
    d = PlateDetector().detect(_plate())
    assert len(d) == 1
    assert d[0].label  # 无 OCR 时为 "未识别"，否则读号


def test_detect_empty_returns_empty():
    d = PlateDetector().detect(np.zeros((200, 300, 3), dtype=np.uint8))
    assert d == []


def test_summarize_unknown():
    s, p = summarize([Detection(0, 0, 10, 10, "未识别")])
    assert "车牌号未识别" in s and p["plate"] == "unknown"


def test_summarize_empty():
    s, p = summarize([])
    assert "未识别到车牌" in s and p["plate"] == "none"


def test_summarize_number():
    s, p = summarize([Detection(0, 0, 10, 10, "鲁A12345")])
    assert "鲁A12345" in s and p["plate"] == "鲁A12345"
