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


# ---------------- Task 2: 划分落盘 / data.yaml / 训练参数集 ----------------
import os
from pathlib import Path

from PIL import Image

from dataset_pipeline import (
    AUGMENT_PARAMS,
    build_train_kwargs,
    do_split,
)


def _make_batch(root, name, n_imgs, labeled_stems=()):
    d = Path(root) / name
    d.mkdir(parents=True)
    for i in range(n_imgs):
        stem = f'{name}_{i:05d}'
        Image.new('RGB', (4, 4), (200, 30, 30)).save(d / f'{stem}.jpg')
        if stem in labeled_stems:
            (d / f'{stem}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    return d


def test_augment_params_flips_disabled():
    assert AUGMENT_PARAMS['fliplr'] == 0.0 and AUGMENT_PARAMS['flipud'] == 0.0


def test_do_split_copies_paired_and_background(tmp_path):
    src, out = tmp_path / 'src', tmp_path / 'out'
    src.mkdir()
    # 10 批次 × 3 张（seed=0 确定分层 train 21 / val 6 / test 3，三侧非空过守卫）
    b1 = _make_batch(src, 'people_b0', 3, labeled_stems=('people_b0_00000',))
    for i in range(1, 10):
        _make_batch(src, f'people_b{i}', 3)
    counts = do_split(str(src), 'people', ['resident', 'outsider'],
                      seed=0, out_root=str(out))
    assert counts == {'train': 21, 'val': 6, 'test': 3}
    # 批次不跨侧：b1 的三张图同侧
    sides_of_b1 = set()
    for side in ('train', 'val', 'test'):
        for img in (out / "people_dataset" / "images" / side).glob("people_b0_*.jpg"):
            sides_of_b1.add(side)
    assert len(sides_of_b1) == 1
    # 有 txt 的图，labels 同侧有同名 txt；无 txt 的图是背景样本（无 txt 但图在）
    all_imgs = {p.name for side in ('train', 'val', 'test')
                for p in (out / 'people_dataset' / 'images' / side).glob('*.jpg')}
    all_txts = {p.name for side in ('train', 'val', 'test')
                for p in (out / 'people_dataset' / 'labels' / side).glob('*.txt')}
    assert len(all_imgs) == 30
    assert all_txts == {'people_b0_00000.txt'}
    # 源目录保持只读母本
    assert len(list(b1.glob('*.jpg'))) == 3
    # data.yaml
    y = (out / 'people_dataset' / 'data.yaml').read_text()
    assert 'nc: 2' in y and 'resident' in y and 'outsider' in y
    assert 'train:' in y and 'val:' in y and 'test:' in y


def test_do_split_empty_side_raises(tmp_path):
    src, out = tmp_path / 'src', tmp_path / 'out'
    src.mkdir()
    _make_batch(src, 'people_only', 3)
    with pytest.raises(ValueError, match='train'):
        do_split(str(src), 'people', ['resident', 'outsider'], out_root=str(out))


def test_build_train_kwargs_merges_augment(tmp_path):
    kw = build_train_kwargs(str(tmp_path / 'data.yaml'), 100, 16, '0',
                            project=str(tmp_path / 'proj'))
    assert kw['device'] == '0' and kw['epochs'] == 100
    assert kw['flipud'] == 0.0 and kw['fliplr'] == 0.0
    assert kw['imgsz'] == 640 and kw['batch'] == 16
