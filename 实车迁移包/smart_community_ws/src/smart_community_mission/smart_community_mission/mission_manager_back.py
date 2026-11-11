#!/usr/bin/env python3
"""Mission manager with traffic-light confirmation and fail-safe waiting.

An UNKNOWN, RED, YELLOW, stale or low-confidence frame NEVER unlocks crossing.
Only fresh GREEN frames received AFTER arriving at a stop-line waypoint count.
"""
import json
import math
import os
import time
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from smart_community_interfaces.msg import PerceptionResult

from .mission_lib import format_announcement


def quat_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


PERCEPTION_TASKS = ['traffic_light', 'people', 'license_plate']


class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager')
        default_yaml = os.path.join(get_package_share_directory('smart_community_mission'),
                                    'config', 'waypoints.yaml')
        self.declare_parameter('waypoints_file', default_yaml)
        self.declare_parameter('start_delay', 3.0)
        self.declare_parameter('min_green_frames', 3)  # usually << 1s with a working camera
        self.declare_parameter('min_green_confidence', 0.55)
        self.declare_parameter('light_max_age', 1.0)
        self.declare_parameter('green_frame_max_gap', 0.8)
        with open(self.get_parameter('waypoints_file').value, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.frame_id = data.get('frame_id', 'map')
        self.waypoints = data.get('waypoints', [])
        self.index = 0
        self.current = None
        self.state = 'WAIT_NAV2'
        self.dwell_until = None
        self.start_at = time.monotonic() + float(self.get_parameter('start_delay').value)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.light_state = 'UNKNOWN'
        self.last_light_at = None
        self.green_count = 0
        self.last_green_at = None
        self.task_pub = self.create_publisher(String, '/mission/current_task', 10)
        self.result_pub = self.create_publisher(String, '/mission/task_result', 10)
        self.announce_pub = self.create_publisher(String, '/announce', 10)
        self.latest = {t: 'NO_DATA' for t in PERCEPTION_TASKS}
        self.latest['parking'] = 'POSE_CONTROL'
        for t in PERCEPTION_TASKS:
            self.create_subscription(PerceptionResult, f'/perception/{t}',
                                     lambda m, t=t: self._set(t, m), 10)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')

    def _set(self, k, m):
        self.latest[k] = m.summary
        if k != 'traffic_light':
            return
        try:
            payload = json.loads(m.json)
            new_state = payload.get('state', 'UNKNOWN')
            confidence = float(payload.get('confidence', 0.0))
        except (ValueError, TypeError, AttributeError):
            new_state, confidence = 'UNKNOWN', 0.0
        if new_state not in ('RED', 'YELLOW', 'GREEN'):
            new_state = 'UNKNOWN'
        self.light_state = new_state
        now = time.monotonic()
        self.last_light_at = now
        # Never count frames seen while driving toward the stop line.
        if self.state != 'WAIT_LIGHT':
            self.green_count = 0
            self.last_green_at = None
            return
        # 'last_green_at' is the timestamp of the previous valid GREEN frame.
        max_gap = float(self.get_parameter('green_frame_max_gap').value)
        min_conf = float(self.get_parameter('min_green_confidence').value)
        if new_state == 'GREEN' and confidence >= min_conf:
            if self.last_green_at is None or now - self.last_green_at > max_gap:
                self.green_count = 1
            else:
                self.green_count += 1
            self.last_green_at = now
        else:
            self.green_count = 0
            self.last_green_at = None

    def tick(self):
        s = String()
        s.data = self.state
        self.state_pub.publish(s)
        now = time.monotonic()
        if now < self.start_at:
            return
        if self.state == 'WAIT_NAV2':
            if self.nav.server_is_ready():
                self.send_next()
            else:
                self.nav.wait_for_server(timeout_sec=0.01)
        elif self.state == 'WAIT_LIGHT':
            max_age = float(self.get_parameter('light_max_age').value)
            if self.last_light_at is None or now - self.last_light_at > max_age:
                self.light_state = 'UNKNOWN'
                self.green_count = 0
                self.last_green_at = None
                return
            if self.green_count >= max(2, int(self.get_parameter('min_green_frames').value)):
                self.get_logger().info(
                    f'Confirmed GREEN ({self.green_count} frames); crossing {self.current["name"]}')
                self._finish_task()
        elif self.state == 'DWELL' and self.dwell_until is not None and now >= self.dwell_until:
            self._finish_task()

    def _finish_task(self):
        task = self.current.get('task', 'none')
        summary = self.latest.get(task, 'NO_DATA')
        sentence = format_announcement(task, summary, self.current)
        self.get_logger().info('TASK RESULT: ' + sentence)
        m = String()
        m.data = sentence
        self.result_pub.publish(m)
        a = String()
        a.data = sentence
        self.announce_pub.publish(a)
        self.index += 1
        self.send_next()

    def send_next(self):
        if self.index >= len(self.waypoints):
            self.state = 'FINISHED'
            self.get_logger().info('===== ALL MISSIONS FINISHED =====')
            return
        wp = self.waypoints[self.index]
        self.current = wp
        # Do not re-use signals confirmed at a previous intersection.
        self.green_count = 0
        self.last_green_at = None
        qz, qw = quat_from_yaw(float(wp.get('yaw', 0.0)))
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(wp['x'])
        pose.pose.position.y = float(wp['y'])
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal = NavigateToPose.Goal()
        goal.pose = pose
        t = String()
        t.data = f"{wp['name']}:{wp.get('task', 'none')}"
        self.task_pub.publish(t)
        self.state = 'NAVIGATING'
        self.get_logger().info(f"[{self.index + 1}/{len(self.waypoints)}] -> {wp['name']}")
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(self.goal_response)

    def goal_response(self, fut):
        gh = fut.result()
        if not gh.accepted:
            if self.current.get('task') == 'traffic_light':
                self.state = 'ERROR_STOP'
                self.get_logger().error('Traffic stop goal rejected: not crossing. Check waypoints/Nav2.')
            else:
                self.get_logger().error('Goal rejected; skip')
                self.index += 1
                self.state = 'WAIT_NAV2'
                self.start_at = time.monotonic() + 1.0
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self.result_cb)

    def result_cb(self, fut):
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            if self.current.get('task') == 'traffic_light':
                self.state = 'WAIT_LIGHT'
                self.light_state = 'UNKNOWN'
                self.green_count = 0
                self.last_green_at = None
                self.last_light_at = None
                self.get_logger().info(f'Arrived {self.current["name"]}; waiting for fresh GREEN')
            else:
                self.state = 'DWELL'
                dwell = float(self.current.get('dwell', 2.0))
                self.dwell_until = time.monotonic() + dwell
                self.get_logger().info(f'Arrived {self.current["name"]}; dwell={dwell}s')
        else:
            if self.current.get('task') == 'traffic_light':
                self.state = 'ERROR_STOP'
                self.get_logger().error(f'Traffic stop navigation failed ({status}); do NOT cross.')
            else:
                self.get_logger().warn(f'Navigation status={status}; skip')
                self.index += 1
                self.state = 'WAIT_NAV2'
                self.start_at = time.monotonic() + 1.0


def main():
    rclpy.init()
    n = MissionManager()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
