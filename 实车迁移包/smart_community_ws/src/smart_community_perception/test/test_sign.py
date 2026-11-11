import cv2
import numpy as np

from smart_community_perception.detectors.sign import SignDetector, summarize
from smart_community_perception.detector_base import Detection


def _ring():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)  # 白底
    cv2.circle(img, (100, 100), 50, (0, 0, 200), 8)    # 红圆环
    return img


def test_detects_no_straight():
    d = SignDetector().detect(_ring())
    assert d and d[0].label == "禁止直行"


def test_empty_returns_empty():
    d = SignDetector().detect(np.full((200, 200, 3), 255, dtype=np.uint8))
    assert d == []


def test_summarize():
    s, p = summarize([Detection(0, 0, 10, 10, "禁止直行")])
    assert "禁止直行" in s and p["sign"] == "禁止直行"
