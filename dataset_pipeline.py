#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧社区数据集流水线（方案 A：在 PC 系统机上运行）

采集（dataset_capture.py）之后的后半段，一条命令走完 划分→data.yaml→GPU 训练：

  # 预标注（自迭代：用上一版模型给新图打框，人工只校正）
  python3 dataset_pipeline.py label --type people --model ~/weights/people_best.pt
  ✅ 应看到：每个批次目录打印 "目录名: 检出 N 框 / M 图"，txt 与 jpg 同名成对出现在源目录

  # 划分 + 一键 GPU 训练（70/20/10 按批次分层，在线增强、禁翻转）
  python3 dataset_pipeline.py run --type people
  ✅ 应看到：先打印每侧张数与 data.yaml 路径，随后每 epoch 一行进度，
  结束打印 "best.pt 已复制到 ~/weights/people_best.pt"

  # 仅划分 / 仅训练
  python3 dataset_pipeline.py split --type people
  python3 dataset_pipeline.py train --type people --epochs 100 --batch 16 --device 0

约定：数据在 ~/datasets/<类型>_<日期序号>/（每个目录=一个批次，整批不跨侧防泄漏）；
权重落位 ~/weights/<类型>_best.pt。trash/fire 等非预设类型首跑加 --classes a,b,c。
"""
import argparse
import random
import sys

# 类名预设（people 与检测器强制集合一致：people.py 校验 resident/outsider）
DATASET_CLASSES = {'people': ['resident', 'outsider']}


def resolve_classes(ds_type, classes_arg):
    """解析数据集类别：--classes（逗号分隔）优先，其次预设表；都没有则报错。"""
    if classes_arg:
        classes = [c.strip() for c in classes_arg.split(',') if c.strip()]
        if classes:
            return classes
    if ds_type in DATASET_CLASSES:
        return list(DATASET_CLASSES[ds_type])
    raise ValueError(
        f'数据集类型 {ds_type!r} 没有预设类名，请用 --classes a,b,c 指定（顺序定死后不再改）')


def assign_splits(batches, ratios=(0.7, 0.2, 0.1), seed=0):
    """批次分层：整批归属 train/val/test 某一侧，永不跨侧。

    batches: [(批次名, 张数), ...]；按 seed 洗牌后依 train→val→test 贪心装填配额。
    """
    order = list(batches)
    random.Random(seed).shuffle(order)
    total = sum(cnt for _, cnt in order) or 1
    quotas = {'train': total * ratios[0], 'val': total * ratios[1], 'test': total * ratios[2]}
    splits = {'train': [], 'val': [], 'test': []}
    filled = {'train': 0, 'val': 0, 'test': 0}
    side_order = ['train', 'val', 'test']
    idx = 0
    for name, cnt in order:
        # 跳过已满的侧（最后一侧不限量收尾）
        while idx < len(side_order) - 1 and filled[side_order[idx]] >= quotas[side_order[idx]]:
            idx += 1
        side = side_order[idx]
        splits[side].append(name)
        filled[side] += cnt
    return splits


def boxes_to_yolo_lines(boxes):
    """(cls, xc, yc, w, h) 归一化坐标列表 → YOLO txt 内容（每行 6 位小数）。"""
    return ''.join(
        f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in boxes)


def build_parser():
    ap = argparse.ArgumentParser(
        prog='dataset_pipeline.py',
        description='采集后流水线：label 预标注 / split 划分 / train GPU 训练 / run 一键')
    sub = ap.add_subparsers(dest='command')

    p = sub.add_parser('label', help='用已有模型给批次目录预标注（txt 落在图旁）')
    p.add_argument('--type', required=True)
    p.add_argument('--model', required=True, help='预标注用 .pt（自迭代：上一版 best）')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--src', default='~/datasets')

    p = sub.add_parser('split', help='按批次分层 70/20/10 复制到 <type>_dataset + data.yaml')
    p.add_argument('--type', required=True)
    p.add_argument('--classes', default=None, help='逗号分隔类名（非预设类型必传）')
    p.add_argument('--src', default='~/datasets')

    p = sub.add_parser('train', help='GPU 训练（在线增强、禁翻转），best.pt 落位 ~/weights')
    p.add_argument('--type', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0', help="GPU 编号，'0'；无 GPU 机器可传 'cpu'")
    p.add_argument('--pretrained', default='yolo11n.pt')

    p = sub.add_parser('run', help='split + train 一条龙')
    p.add_argument('--type', required=True)
    p.add_argument('--classes', default=None)
    p.add_argument('--src', default='~/datasets')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--pretrained', default='yolo11n.pt')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
