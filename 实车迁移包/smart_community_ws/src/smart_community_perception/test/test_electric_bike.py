import numpy as np

from smart_community_perception.detectors.electric_bike import ElectricBikeDetector, summarize
from smart_community_perception.detector_base import Detection

CYAN = (224, 217, 0)  # 青色 BGR


def _img():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[60:120, 30:50] = CYAN    # 竖块 20x60
    img[60:120, 80:100] = CYAN   # 竖块 20x60
    img[140:160, 120:180] = CYAN  # 横块 60x20
    return img


def test_counts_upright_and_fallen():
    d = ElectricBikeDetector().detect(_img())
    labels = [x.label for x in d]
    assert labels.count("正常") == 2
    assert labels.count("倒伏") == 1


def test_empty_returns_zero():
    d = ElectricBikeDetector().detect(np.zeros((200, 300, 3), dtype=np.uint8))
    assert len(d) == 0


def test_summarize():
    dets = [Detection(0, 0, 10, 10, "正常"), Detection(0, 0, 10, 10, "正常"),
            Detection(0, 0, 10, 10, "倒伏")]
    s, p = summarize(dets)
    assert "正常2辆" in s and "倒伏1辆" in s
    assert p["upright"] == 2 and p["fallen"] == 1
