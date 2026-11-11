"""dataset_pipeline 纯逻辑测试（不依赖 ultralytics——它只允许函数内懒加载）。

钉住 Review Focus：批次不跨侧（防 mAP 虚高）、people 类名与检测器强制集合一致。
"""
import pytest

from dataset_pipeline import (
    assign_splits,
    boxes_to_yolo_lines,
    resolve_classes,
)


def test_resolve_classes_people_preset_and_override_and_error():
    assert resolve_classes('people', None) == ['resident', 'outsider']
    assert resolve_classes('trash', 'recyclable,kitchen,hazard,other') == [
        'recyclable', 'kitchen', 'hazard', 'other']
    with pytest.raises(ValueError):
        resolve_classes('fire', None)


def test_assign_splits_whole_batches_never_split_deterministic():
    batches = [(f'people_20261111-{i:02d}', 10) for i in range(10)]
    a = assign_splits(batches, seed=0)
    b = assign_splits(batches, seed=0)
    assert a == b                                    # 同 seed 确定
    placed = [x for side in a.values() for x in side]
    assert sorted(placed) == sorted(n for n, _ in batches)   # 每批恰好一侧一次
    counts = {s: sum(cnt for n, cnt in batches if n in a[s]) for s in a}
    assert counts['train'] == 70 and counts['val'] == 20 and counts['test'] == 10


def test_assign_splits_few_batches_all_train():
    a = assign_splits([('a', 5), ('b', 5)])
    assert a['train'] == ['a', 'b'] and a['val'] == [] and a['test'] == []


def test_boxes_to_yolo_lines_format():
    assert boxes_to_yolo_lines(
        [(1, 0.5, 0.25, 0.1, 0.2)]) == '1 0.500000 0.250000 0.100000 0.200000\n'
    assert boxes_to_yolo_lines([]) == ''
