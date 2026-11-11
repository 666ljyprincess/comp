from smart_community_perception.perception_node import build_result
from smart_community_perception.detector_base import Detection


def test_build_result_carries_boxes_and_summary():
    r = build_result("people", [Detection(1, 2, 3, 4, "外来人员")], "共16人", {"count": 16})
    assert r.task == "people"
    assert r.summary == "共16人"
    assert r.json == '{"count": 16}'
    assert len(r.boxes) == 1 and r.boxes[0].label == "外来人员"
