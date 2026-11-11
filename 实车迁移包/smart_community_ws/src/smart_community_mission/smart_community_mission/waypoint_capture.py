#!/usr/bin/env python3
"""Interactive waypoint capture for Nav2 patrol calibration.

Teleop the robot to a target pose, then press Enter to save the current
/amcl_pose (map frame) as a waypoint line in a standalone YAML file.
"""
import math
import os
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from .mission_lib import format_waypoint_line, quat_to_yaw

TASKS = ('traffic_light', 'people', 'license_plate', 'parking', 'none')


def prompt(label, default=None, choices=None):
    if choices:
        hint = f" ({'/'.join(choices)}, 默认 {default})"
    elif default is not None:
        hint = f" [默认 {default}]"
    else:
        hint = ""
    value = input(f"  {label}{hint}: ").strip()
    if not value:
        value = default
    if choices and value not in choices:
        print(f"  !! 无效值，使用默认 {default}")
        value = default
    return value


class WaypointCapture(Node):
    def __init__(self):
        super().__init__('waypoint_capture')
        self.declare_parameter(
            'output_file',
            os.path.expanduser('~/.ros/smart_community/captured_waypoints.yaml'))
        self.declare_parameter('frame_id', 'map')
        self.out = self.get_parameter('output_file').value
        self.frame_id = self.get_parameter('frame_id').value
        self.latest = None
        self.count = 0
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._pose_cb, 10)

    def _pose_cb(self, msg):
        self.latest = msg.pose.pose

    def capture(self):
        if self.latest is None:
            print('  !! 尚未收到 /amcl_pose，请确认 AMCL 已启动并完成定位')
            return
        x = self.latest.position.x
        y = self.latest.position.y
        yaw = quat_to_yaw(self.latest.orientation.z, self.latest.orientation.w)
        self.count += 1
        name = prompt('name', f'wp_{self.count}')
        task = prompt('task', 'none', TASKS)
        wp = {'name': name, 'task': task, 'x': x, 'y': y, 'yaw': yaw}
        if task == 'people':
            wp['zone'] = prompt('zone', 'A')
        elif task == 'license_plate':
            wp['lot'] = prompt('lot', '1')
        dwell = prompt('dwell', '2.0')
        try:
            wp['dwell'] = float(dwell)
        except ValueError:
            wp['dwell'] = 2.0
        line = format_waypoint_line(wp)
        self._append(line)
        print(f'\n  已保存 {name}: x={x:.3f} y={y:.3f} yaw={yaw:.5f} rad ({math.degrees(yaw):.1f}°)')
        print('  行内容: ' + line)
        print('  输出文件: ' + self.out + '\n')

    def _append(self, line):
        if not os.path.exists(self.out):
            os.makedirs(os.path.dirname(self.out), exist_ok=True)
            with open(self.out, 'w', encoding='utf-8') as f:
                f.write(f'frame_id: {self.frame_id}\nwaypoints:\n')
        with open(self.out, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def main():
    rclpy.init()
    node = WaypointCapture()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    print('=' * 62)
    print('  巡航点取点工具')
    print('  1) 遥控机器人到位，调好车头朝向')
    print('  2) 回车 = 抓取当前 /amcl_pose 并保存')
    print('  3) 输入 q 回车 = 退出')
    print('  输出文件: ' + node.out)
    print('=' * 62)
    try:
        while True:
            cmd = input('\n[回车抓取 / q 退出] ').strip().lower()
            if cmd in ('q', 'quit', 'exit'):
                break
            node.capture()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
