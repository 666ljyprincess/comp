import numpy as np

from smart_community_perception.detectors.trash_bin import TrashBinDetector, summarize
from smart_community_perception.detector_base import Detection

H, W = 200, 200


def _bin(body=(0, 200, 0), lid_red=True, garbage=None):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[H // 5:, :] = body              # 桶身
    if lid_red:
        img[:H // 5, :] = (0, 0, 200)   # 红盖（打开）
    if garbage is not None:
        gy1, gy2 = 100, 130
        gx1, gx2 = 60, 140
        img[gy1:gy2, gx1:gx2] = garbage
    return img


def test_green_bin_open():
    d = TrashBinDetector().detect(_bin(body=(0, 200, 0), lid_red=True))
    assert d and d[0].label == "厨余"
    assert d[0].extra["state"] == "打开"


def test_red_garbage_in_green_bin_is_wrong():
    d = TrashBinDetector().detect(_bin(body=(0, 200, 0), lid_red=False, garbage=(0, 0, 200)))
    assert d and d[0].label == "厨余"
    assert d[0].extra["garbage"] == "有害"
    assert d[0].extra["disposal"] == "错误"


def test_empty_returns_empty():
    d = TrashBinDetector().detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert d == []


def test_summarize():
    s, p = summarize([Detection(0, 0, W, H, "厨余", 1.0, {"state": "打开", "garbage": "无", "disposal": "正确"})])
    assert "厨余" in s and p["bin"] == "厨余" and p["state"] == "打开"
