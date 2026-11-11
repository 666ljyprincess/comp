#!/usr/bin/env python3
"""HSV 阈值标定工具。

两种用法：

1) 抓一帧真实渲染画面（在仿真运行时执行）：
   python3 hsv_calibrate.py capture --topic /camera/image_raw --out /tmp/frame.jpg

2) 在抓到的图上用矩形 ROI 采样 HSV，打印建议的 cv2.inRange 上下界：
   python3 hsv_calibrate.py sample --image /tmp/frame.jpg --roi 320,240,80,60

   ROI 格式为 x,y,w,h（左上角 x/y + 宽/高）。用 `--expand 10` 可把建议上下界
   各向外放宽 N（H 单位是 0-180，S/V 是 0-255），用于覆盖光照变化。
"""
import argparse
import cv2
import numpy as np


def capture(topic, out):
    import rclpy
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node('hsv_capture')
    bridge = CvBridge()
    got = {}

    def cb(msg):
        if 'img' not in got:
            got['img'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    node.create_subscription(Image, topic, cb, 10)
    print(f'等待话题 {topic} 的一帧 ...')
    while rclpy.ok() and 'img' not in got:
        rclpy.spin_once(node, timeout_sec=0.1)
    if 'img' not in got:
        print('未收到图像，检查话题名/仿真是否在跑')
        return 1
    cv2.imwrite(out, got['img'])
    print(f'已保存 {out}  尺寸 {got["img"].shape[1]}x{got["img"].shape[0]}')
    node.destroy_node()
    rclpy.shutdown()
    return 0


def sample(image, roi, expand):
    img = cv2.imread(image)
    if img is None:
        print(f'无法读取 {image}')
        return 1
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lo = hsv.min(axis=0).astype(int)
    hi = hsv.max(axis=0).astype(int)
    mean = hsv.mean(axis=0)

    lo_e = np.clip([lo[0] - expand, lo[1] - expand, lo[2] - expand], 0, [180, 255, 255])
    hi_e = np.clip([hi[0] + expand, hi[1] + expand, hi[2] + expand], 0, [180, 255, 255])

    print(f'ROI {roi}  采样 {hsv.shape[0]} 像素')
    print(f'  实测 HSV  min = ({lo[0]},{lo[1]},{lo[2]})   max = ({hi[0]},{hi[1]},{hi[2]})   mean = ({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})')
    print(f'  建议 inRange 下界 = ({lo_e[0]},{lo_e[1]},{lo_e[2]})')
    print(f'  建议 inRange 上界 = ({hi_e[0]},{hi_e[1]},{hi_e[2]})')
    if hi[0] - lo[0] > 100 and lo[0] < 10 and hi[0] > 170:
        print('  注意：红色在 HSV 里跨越 0/180 边界，需拆成两段 (0..a)|(b..180)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--topic', default='/camera/image_raw')
    c.add_argument('--out', default='/tmp/frame.jpg')
    s = sub.add_parser('sample')
    s.add_argument('--image', required=True)
    s.add_argument('--roi', required=True, help='x,y,w,h')
    s.add_argument('--expand', type=int, default=10)
    args = ap.parse_args()

    if args.cmd == 'capture':
        return capture(args.topic, args.out)
    roi = [int(v) for v in args.roi.split(',')]
    assert len(roi) == 4, '--roi 必须是 x,y,w,h'
    return sample(args.image, roi, args.expand)


if __name__ == '__main__':
    raise SystemExit(main())
