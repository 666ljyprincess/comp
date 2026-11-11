"""Offline synthetic sanity tests, runnable without ROS 2: python3 test_traffic_light.py"""
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field

import cv2
import numpy as np

base = types.ModuleType('smart_community_perception')
base.__path__ = []
detectors = types.ModuleType('smart_community_perception.detectors')
detectors.__path__ = []
common = types.ModuleType('smart_community_perception.detector_base')


class Detector:
    pass


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float = 1.0
    extra: dict = field(default_factory=dict)


common.Detector = Detector
common.Detection = Detection
sys.modules.update({
    'smart_community_perception': base,
    'smart_community_perception.detectors': detectors,
    'smart_community_perception.detector_base': common,
})
spec = importlib.util.spec_from_file_location(
    'smart_community_perception.detectors.traffic_light',
    os.path.join(os.path.dirname(__file__), 'traffic_light.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def draw(state, order='RYG', background_red=False):
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    if background_red:
        image[180:470, 20:620] = (15, 15, 235)
    cv2.rectangle(image, (220, 65), (460, 150), (15, 15, 15), -1)
    colors = {
        'R': ((0, 0, 75), (0, 0, 255)),
        'Y': ((0, 75, 75), (0, 230, 255)),
        'G': ((0, 70, 0), (0, 255, 0)),
    }
    for i, c in enumerate(order):
        is_on = {'RED': 'R', 'YELLOW': 'Y', 'GREEN': 'G'}.get(state) == c
        cv2.circle(image, (260 + i * 80, 107), 24, colors[c][int(is_on)], -1)
    return image


def assert_state(img, state, roi=True, order='RYG'):
    detector = module.TrafficLightDetector(
        roi=(220 / 640, 65 / 480, 460 / 640, 150 / 480) if roi else None,
        order=order)
    dets = detector.detect(img)
    actual = module.summarize(dets)[1]['state']
    assert actual == state, f'expected {state}, got {actual}, detail={dets}'
    return module.summarize(dets)[1]


for state in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
    print('ROI:', state, assert_state(draw(state), state))
print('ROI reversed:', assert_state(draw('GREEN', 'GYR'), 'GREEN', order='GYR'))
print('ROI ignore red road:', assert_state(draw('GREEN', background_red=True), 'GREEN'))
for state in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
    print('AUTO:', state, assert_state(draw(state, background_red=state=='GREEN'), state, roi=False))
print('All 9 detector cases passed')
