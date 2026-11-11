"""dataset_capture.py 保存路径约定（方案 A：脚本在 PC 跑）。

钉住：SAVE_ROOT 必须落在 PC 用户目录（~/datasets），不得是车端/板端硬编码路径
（/userdata 在 PC 系统机上不存在，makedirs 会因权限失败）；两份副本（仓库根 +
实车迁移包）必须一致。
"""
import importlib.util
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPIES = [
    REPO / 'dataset_capture.py',
    REPO / '实车迁移包' / 'smart_community_ws' / 'dataset_capture.py',
]


def _load(path):
    name = 'dataset_capture_under_test_' + path.parent.name
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_save_root_is_pc_user_dir_not_board_path():
    for copy in COPIES:
        mod = _load(copy)
        assert mod.SAVE_ROOT == os.path.expanduser('~/datasets'), (
            f'{copy}: SAVE_ROOT={mod.SAVE_ROOT!r}，应为 ~/datasets（PC 用户目录）')


def test_both_copies_identical():
    a = COPIES[0].read_bytes()
    b = COPIES[1].read_bytes()
    assert a == b, '两份 dataset_capture.py 副本内容漂移，改完一处必须同步另一处'
