#!/usr/bin/env python3
"""Mission-controlled navigation and arrival-only perception for the contest.

Safety properties:
* Vision at people/plate waypoints is accepted ONLY after Nav2 success.
* Each vision result must identify the CURRENT waypoint (never use old results).
* Red/yellow/unknown/stale light frames never release a stop line.
* Any waypoint which fails its bounded retries enters ERROR_STOP, not 'done'.
"""
import json
import math
import os
import time
import yaml
from collections import Counter, defaultdict

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import String
from smart_community_interfaces.msg import PerceptionResult

from .mission_lib import format_announcement, quat_to_yaw
from .contest_rules import people_consensus, zone_totals, parking_pass, plate_consensus, valid_plate


def quat_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


PERCEPTION_TASKS = ('traffic_light', 'people', 'license_plate')


class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager')
        default_yaml = os.path.join(get_package_share_directory('smart_community_mission'),
                                    'config', 'waypoints.yaml')
        self.declare_parameter('waypoints_file', default_yaml)
        self.declare_parameter('start_delay', 3.0)
        self.declare_parameter('min_green_frames', 3)
        self.declare_parameter('min_green_confidence', 0.55)
        self.declare_parameter('light_max_age', 1.0)
        self.declare_parameter('green_frame_max_gap', 0.8)
        self.declare_parameter('max_nav_retries', 1)
        self.declare_parameter('retry_delay', 2.0)
        self.declare_parameter('min_detection_frames', 2)
        self.declare_parameter('min_plate_votes', 2)
        self.declare_parameter('max_dwell', 8.0)
        # 不使用全局 max_dwell 限制 people：YOLO 首次加载/预热可能明显慢于 8 秒。
        self.declare_parameter('people_first_frame_timeout', 30.0)
        self.declare_parameter('people_observation_seconds', 8.0)
        self.declare_parameter('people_min_consensus_frames', 2)
        # 本赛场每个观察点都应有实际人偶；空帧不能被误当成已经识别成功。
        # 真的存在空观察点时，可将此参数设为 True（仍需 >=2 帧有效空图）。
        self.declare_parameter('people_allow_empty', False)
        # Plate OCR may be much slower than the camera stream, especially the
        # first PaddleOCR inference.  Start the confirmation timeout only after
        # the FIRST valid license plate, not immediately upon Nav2 arrival.
        self.declare_parameter('plate_first_valid_timeout', 90.0)
        self.declare_parameter('plate_confirmation_timeout', 60.0)
        # Final parking requires Nav2 success AND independent pose/standstill verification.
        self.declare_parameter('parking_xy_tolerance', 0.12)
        self.declare_parameter('parking_yaw_tolerance', 0.20)
        self.declare_parameter('parking_linear_tolerance', 0.03)
        self.declare_parameter('parking_angular_tolerance', 0.10)
        self.declare_parameter('parking_stable_seconds', 0.8)
        self.declare_parameter('parking_max_wait', 10.0)
        with open(self.get_parameter('waypoints_file').value, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.frame_id = data.get('frame_id', 'map')
        self.waypoints = data.get('waypoints', [])
        self.zone_waypoints = defaultdict(list)
        for wp in self.waypoints:
            if wp.get('task') == 'people' and wp.get('zone'):
                self.zone_waypoints[str(wp['zone']).upper()].append(wp['name'])
        self.zone_results = defaultdict(dict)
        self.people_samples = []
        self.people_first_frame_at = None
        self.plate_first_valid_at = None
        self.people_last_error = ''
        self.parking_verified = False
        self.parking_stable_since = None
        self.last_odom = None
        self.last_odom_at = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.index = 0
        self.current = None
        self.state = 'WAIT_NAV2'
        self._prev_state = None
        self.start_at = time.monotonic() + float(self.get_parameter('start_delay').value)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.goal_generation = 0
        self.nav_retries = 0
        self.retry_at = 0.0
        self.dwell_start = None
        self.dwell_until = None
        self.dwell_deadline = None
        self.accepted_frames = 0
        self.plate_votes = Counter()
        self.light_state = 'UNKNOWN'
        self.last_light_at = None
        self.green_count = 0
        self.last_green_at = None
        self.last_reported_light = None
        self.latest = {t: 'NO_DATA' for t in PERCEPTION_TASKS}
        self.latest['parking'] = 'POSE_CONTROL'
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.task_pub = self.create_publisher(String, '/mission/current_task', 10)
        self.active_task_pub = self.create_publisher(String, '/mission/active_task', 10)
        self.result_pub = self.create_publisher(String, '/mission/task_result', 10)
        self.announce_pub = self.create_publisher(String, '/announce', 10)
        self._active_task = 'none'
        for t in PERCEPTION_TASKS:
            self.create_subscription(PerceptionResult, f'/perception/{t}',
                                     lambda m, kind=t: self._set(kind, m), 10)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f'已加载 {len(self.waypoints)} 个巡航点，坐标帧={self.frame_id}')

    def _set(self, kind, msg):
        if self.current is None:
            return
        current_task = self.current.get('task')
        required_state = 'WAIT_LIGHT' if kind == 'traffic_light' else 'DWELL'
        if current_task != kind or self.state != required_state:
            return
        try:
            payload = json.loads(msg.json)
            if not isinstance(payload, dict):
                return
        except (ValueError, TypeError, AttributeError):
            return
        # 感知节点若携带航点标记，则拒绝来自前一个视角的排队旧帧。
        if kind in ('people', 'license_plate') and payload.get('waypoint') != self.current['name']:
            # Reject queued results from an earlier viewpoint. All updated
            # perception nodes attach the waypoint to their result payload.
            return
        self.latest[kind] = msg.summary
        now = time.monotonic()
        if kind == 'traffic_light':
            new_state = payload.get('state', 'UNKNOWN')
            try:
                confidence = float(payload.get('confidence', 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if new_state not in ('RED', 'YELLOW', 'GREEN'):
                new_state = 'UNKNOWN'
            self.light_state = new_state
            self.last_light_at = now
            # Key contest actions are announced on state transition, not every frame.
            # GREEN is announced only AFTER the safety quorum is verified below.
            if new_state != self.last_reported_light:
                self.last_reported_light = new_state
                if new_state in ('RED', 'YELLOW'):
                    sentence = '前方红灯，停车等待。' if new_state == 'RED' else '前方黄灯，停车等待。'
                    self._publish(self.announce_pub, sentence)
            self.get_logger().info(
                f'红绿灯检测：{self.current["name"]}，状态={new_state}，置信度={confidence:.2f}',
                throttle_duration_sec=1.0)
            max_gap = float(self.get_parameter('green_frame_max_gap').value)
            min_conf = float(self.get_parameter('min_green_confidence').value)
            if new_state == 'GREEN' and confidence >= min_conf:
                self.green_count = (self.green_count + 1
                                    if self.last_green_at is not None and
                                    now - self.last_green_at <= max_gap else 1)
                self.last_green_at = now
            else:
                self.green_count = 0
                self.last_green_at = None
            return
        if kind == 'people':
            if payload.get('status') != 'OK':
                self.people_last_error = str(payload.get('status', 'UNKNOWN'))
                return
            # 首次有效帧到达后，才开始固定观察窗口（包括刚开始的 0 人空帧）。
            resident = payload.get('resident', 0)
            outsider = payload.get('outsider', 0)
            if (isinstance(resident, int) and not isinstance(resident, bool)
                    and isinstance(outsider, int) and not isinstance(outsider, bool)
                    and resident >= 0 and outsider >= 0):
                if self.people_first_frame_at is None:
                    self.people_first_frame_at = now
                    observe = max(
                        2.0, float(self.current.get('dwell', 2.0)),
                        float(self.get_parameter('people_observation_seconds').value))
                    self.dwell_deadline = now + observe
                    self.get_logger().info(
                        f'{self.current["name"]} 收到第一帧有效人偶结果，'
                        f'开始连续观察 {observe:.1f}s')
                self.accepted_frames += 1
                self.people_samples.append((resident, outsider))
        elif kind == 'license_plate':
            self.accepted_frames += 1
            plate = payload.get('plate', '')
            # Invalid OCR text / preliminary empty frames do NOT end the wait.
            if valid_plate(plate):
                self.plate_votes[plate] += 1
                if self.plate_first_valid_at is None:
                    self.plate_first_valid_at = now
                    confirmation_window = max(
                        2.0, float(self.get_parameter('plate_confirmation_timeout').value))
                    self.dwell_deadline = now + confirmation_window
                    self.get_logger().info(
                        f'{self.current["name"]} 收到首张合法车牌 {plate}，'
                        f'开始最多 {confirmation_window:.1f}s 的多帧确认')
                self.get_logger().info(
                    f'{self.current["name"]} OCR车牌候选={plate}，'
                    f'累计票数={self.plate_votes[plate]}')

    def tick(self):
        self._publish(self.state_pub, self.state)
        self._publish(self.active_task_pub, self._active_task)
        # 航点标记持续发布：感知节点 YOLO 预加载完成后也能及时取得当前航点。
        if self.current is not None:
            self._publish(self.task_pub,
                          f"{self.current['name']}:{self.current.get('task', 'none')}")
        if self.state != self._prev_state:
            self.get_logger().info(f'状态机切换：{self._prev_state or "初始"} -> {self.state}')
            self._prev_state = self.state
        now = time.monotonic()
        if now < self.start_at:
            return
        if self.state == 'WAIT_NAV2':
            if self.nav.server_is_ready():
                self.send_next()
            else:
                self.nav.wait_for_server(timeout_sec=0.01)
        elif self.state == 'RETRY_WAIT' and now >= self.retry_at:
            if self.nav.server_is_ready():
                self.send_next()
        elif self.state == 'WAIT_LIGHT':
            max_age = float(self.get_parameter('light_max_age').value)
            if self.last_light_at is None or now - self.last_light_at > max_age:
                self.light_state = 'UNKNOWN'
                self.green_count = 0
                self.last_green_at = None
                return
            if self.green_count >= max(2, int(self.get_parameter('min_green_frames').value)):
                self.get_logger().info(f'确认连续 {self.green_count} 帧绿灯，驶离 {self.current["name"]}')
                self._finish_task()
        elif self.state == 'DWELL':
            min_elapsed = self.dwell_until is not None and now >= self.dwell_until
            timed_out = self.dwell_deadline is not None and now >= self.dwell_deadline
            task = self.current.get('task')
            if task == 'parking':
                if self._verify_parking(now):
                    self.parking_verified = True
                    self._finish_task()
                elif timed_out:
                    self._parking_failed()
                return
            if task == 'people':
                # 没有第一帧有效结果，说明可能还在加载/预热/推理：最多等待初始化宽限期。
                if self.people_first_frame_at is None:
                    if timed_out:
                        self._recognition_failed()
                    return
                # 有效帧后的整个观察窗口都采样，防止早期空帧/不完整检测抢先结束。
                if not timed_out:
                    return
                count = people_consensus(
                    self.people_samples,
                    min_votes=int(self.get_parameter('people_min_consensus_frames').value),
                    allow_empty=bool(self.get_parameter('people_allow_empty').value),
                    min_valid_frames=int(self.get_parameter('min_detection_frames').value))
                if count is not None:
                    self._finish_task()
                else:
                    self._recognition_failed()
                return
            if task == 'license_plate':
                quorum = max(2, int(self.get_parameter('min_plate_votes').value))
                winner = plate_consensus(self.plate_votes, min_votes=quorum)
                # Never accept 0/unknown frames as successful recognition.
                if winner and min_elapsed:
                    self._finish_task()
                elif timed_out:
                    self._recognition_failed()
                return
            if timed_out or min_elapsed:
                self._finish_task()

    @staticmethod
    def _publish(pub, text):
        msg = String()
        msg.data = text
        pub.publish(msg)

    def _on_odom(self, msg):
        self.last_odom = msg
        self.last_odom_at = time.monotonic()

    def _verify_parking(self, now):
        """Independently check current map pose + fresh odom velocities.

        Do not label parking as a success on Nav2's goal result alone.
        """
        self._parking_check_reason = '等待有效 TF 和里程计'
        if (self.last_odom is None or self.last_odom_at is None
                or now - self.last_odom_at > 1.0):
            self.parking_stable_since = None
            return False
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame_id, 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.02))
        except Exception as exc:
            self.parking_stable_since = None
            self._parking_check_reason = f'无法获取停车位姿: {exc}'
            return False
        pos = tf.transform.translation
        rot = tf.transform.rotation
        yaw = quat_to_yaw(rot.z, rot.w)
        twist = self.last_odom.twist.twist
        xy_tol = float(self.get_parameter('parking_xy_tolerance').value)
        yaw_tol = float(self.get_parameter('parking_yaw_tolerance').value)
        v_tol = float(self.get_parameter('parking_linear_tolerance').value)
        w_tol = float(self.get_parameter('parking_angular_tolerance').value)
        actual_v = math.hypot(twist.linear.x, twist.linear.y)
        actual_w = twist.angular.z
        valid = parking_pass(
            pos.x, pos.y, yaw,
            float(self.current['x']), float(self.current['y']),
            float(self.current.get('yaw', 0)), actual_v, actual_w,
            xy_tolerance=xy_tol, yaw_tolerance=yaw_tol,
            linear_tolerance=v_tol, angular_tolerance=w_tol)
        if not valid:
            self.parking_stable_since = None
            self._parking_check_reason = (
                f'尚未满足停车位姿/静止要求：坐标=({pos.x:.3f},{pos.y:.3f}), '
                f'偏航={yaw:.2f}, 速度={actual_v:.3f}m/s')
            return False
        if self.parking_stable_since is None:
            self.parking_stable_since = now
        self._parking_check_reason = '已进入停车区域，等待持续静止'
        return now - self.parking_stable_since >= max(
            0.1, float(self.get_parameter('parking_stable_seconds').value))

    def _parking_failed(self):
        self.state = 'ERROR_STOP'
        self._active_task = 'none'
        reason = '终点停车校验未通过，巡检未完成：' + self._parking_check_reason
        self.get_logger().error(reason)
        self._publish(self.result_pub, reason)
        self._publish(self.announce_pub, reason)

    def _report_people_view(self):
        wp = self.current
        zone = str(wp['zone']).upper()
        view_names = self.zone_waypoints[zone]
        view_index = view_names.index(wp['name']) + 1
        count = people_consensus(
            self.people_samples,
            min_votes=int(self.get_parameter('people_min_consensus_frames').value),
            allow_empty=bool(self.get_parameter('people_allow_empty').value),
            min_valid_frames=int(self.get_parameter('min_detection_frames').value))
        if count is None:
            self._recognition_failed()
            return False
        # One result per viewpoint, not a per-video-frame sum.
        self.zone_results[zone][wp['name']] = count
        sentence = (f'{zone}区第{view_index}个视角：识别到{sum(count)}个人偶，'
                    f'其中居民{count[0]}个，非社区人员{count[1]}个。')
        self._publish(self.result_pub, sentence)
        self._publish(self.announce_pub, sentence)
        self.get_logger().info('人偶单视角统计：' + sentence)
        if view_index == len(view_names):
            if len(self.zone_results[zone]) != len(view_names):
                self._recognition_failed()
                return False
            residents, outsiders = zone_totals(
                [self.zone_results[zone][name] for name in view_names])
            sentence = (f'{zone}区{len(view_names)}个视角识别完成，'
                        f'共计{residents+outsiders}个人偶，'
                        f'居民{residents}个，非社区人员{outsiders}个。')
            self._publish(self.result_pub, sentence)
            self._publish(self.announce_pub, sentence)
            self.get_logger().info('人偶区域汇总（视角计数求和，未跨视角去重）：' + sentence)
        return True

    def _recognition_failed(self):
        wp = self.current
        task = wp.get('task', 'none')
        self.state = 'RECOGNITION_FAILED'
        self._active_task = 'none'
        if task == 'people':
            zone = str(wp.get('zone', '')).upper()
            positive = Counter(x for x in self.people_samples if sum(x) > 0)
            detail = ('尚未收到模型有效帧（可检查模型预热、相机订阅与 YOLO 权重）'
                      if self.people_first_frame_at is None else
                      f'共{self.accepted_frames}帧有效图像，非空样本{sum(positive.values())}帧，'
                      f'最高一致票数{max(positive.values(), default=0)}')
            if self.people_last_error:
                detail += f'；检测器状态={self.people_last_error}'
            reason = (f'{zone}区 {wp["name"]} 人偶识别失败：{detail}。'
                      f'小车停留原地待处理')
        elif task == 'license_plate':
            lot = wp.get('lot', '?')
            reason = (f'{lot}号停车场 {wp["name"]} 车牌识别失败：'
                      f'未获得两次一致的合法识别结果（已处理{self.accepted_frames}帧，'
                      f'有效候选={dict(self.plate_votes)}），小车停留原地待处理')
        else:
            reason = f'{wp["name"]} 识别失败，小车停留原地待处理'
        self._publish(self.result_pub, reason)
        self._publish(self.announce_pub, reason)
        self.get_logger().error(reason)

    def _finish_task(self):
        task = self.current.get('task', 'none')
        if task == 'people':
            if not self._report_people_view():
                return
        elif task == 'parking':
            if not self.parking_verified:
                self._parking_failed()
                return
            sentence = '车辆已到达指定终点，位姿与静止状态验收通过，停车成功。'
            self._publish(self.result_pub, sentence)
            self._publish(self.announce_pub, sentence)
            self.get_logger().info('任务结果：' + sentence)
        elif task in ('traffic_light', 'license_plate'):
            summary = self.latest.get(task, 'NO_DATA')
            if task == 'license_plate':
                quorum = max(2, int(self.get_parameter('min_plate_votes').value))
                summary = plate_consensus(self.plate_votes, min_votes=quorum)
                if not summary:
                    # Defense in depth: never announce an unverified plate.
                    self._recognition_failed()
                    return
            sentence = format_announcement(task, summary, self.current)
            self._publish(self.result_pub, sentence)
            self._publish(self.announce_pub, sentence)
            self.get_logger().info('任务结果：' + sentence)
        # turn / other transit waypoints: no meaningless NO_DATA announcements.
        self.index += 1
        self.nav_retries = 0
        self.send_next()

    def send_next(self):
        if self.index >= len(self.waypoints):
            self.state = 'FINISHED'
            self._active_task = 'none'
            sentence = (f'巡检完成！全部{len(self.waypoints)}个巡检点已处理，'
                        'A区和B区人偶统计已播报，车牌识别已结束，终点停车成功。')
            self._publish(self.result_pub, sentence)
            self._publish(self.announce_pub, sentence)
            self.get_logger().info('===== ' + sentence + ' =====')
            return
        wp = self.waypoints[self.index]
        self.current = wp
        self.dwell_start = self.dwell_until = self.dwell_deadline = None
        self.accepted_frames = 0
        self.plate_votes.clear()
        self.people_samples = []
        self.people_first_frame_at = None
        self.plate_first_valid_at = None
        self.people_last_error = ''
        self.parking_stable_since = None
        self.parking_verified = False
        self.latest[wp.get('task', 'none')] = 'NO_DATA'
        self.green_count = 0
        self.last_green_at = None
        self.last_light_at = None
        self.light_state = 'UNKNOWN'
        self.last_reported_light = None
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
        self.state = 'NAVIGATING'
        self._active_task = 'none'
        self.goal_generation += 1
        generation = self.goal_generation
        self._publish(self.task_pub, f"{wp['name']}:{wp.get('task', 'none')}")
        self.get_logger().info(
            f'[{self.index + 1}/{len(self.waypoints)}] 前往 {wp["name"]} '
            f'(重试 {self.nav_retries}，坐标 {float(wp["x"]):.3f}, {float(wp["y"]):.3f})')
        future = self.nav.send_goal_async(goal)
        future.add_done_callback(lambda fut, g=generation: self.goal_response(fut, g))

    def goal_response(self, future, generation):
        if generation != self.goal_generation or self.state != 'NAVIGATING':
            return
        try:
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                self._nav_failed('导航目标被拒绝')
                return
            result = goal_handle.get_result_async()
            result.add_done_callback(lambda fut, g=generation: self.result_cb(fut, g))
        except Exception as exc:
            self._nav_failed(f'发送导航目标异常：{exc}')

    def result_cb(self, future, generation):
        if generation != self.goal_generation or self.state != 'NAVIGATING':
            return
        try:
            status = future.result().status
        except Exception as exc:
            self._nav_failed(f'导航结果获取异常：{exc}')
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            self._nav_failed(f'导航返回状态码 {status}')
            return
        self._arrived()

    def _arrived(self):
        """Called after a confirmed navigation success (also used by FollowPath)."""
        wp = self.current
        if wp.get('task') == 'traffic_light':
            self.state = 'WAIT_LIGHT'
            self._active_task = 'traffic_light'
            self.light_state = 'UNKNOWN'
            self.last_reported_light = None
            self.green_count = 0
            self.last_light_at = None
            self.last_green_at = None
            self.get_logger().info(f'到达停止点 {wp["name"]}，只接受到点后的新绿灯帧')
        else:
            now = time.monotonic()
            self.dwell_start = now
            self.dwell_until = now + max(0.1, float(wp.get('dwell', 2.0)))
            task = wp.get('task')
            timeout_key = ('people_first_frame_timeout' if task == 'people' else
                           'parking_max_wait' if task == 'parking' else
                           'plate_first_valid_timeout' if task == 'license_plate' else
                           'max_dwell')
            self.dwell_deadline = now + max(
                float(wp.get('dwell', 2.0)), float(self.get_parameter(timeout_key).value))
            self.state = 'DWELL'
            self._active_task = wp.get('task')
            self.get_logger().info(f'到达 {wp["name"]}，停车开启 {wp.get("task")} 识别')

    def _nav_failed(self, reason):
        wp = self.current['name'] if self.current else '?'
        max_retries = max(0, int(self.get_parameter('max_nav_retries').value))
        if self.nav_retries < max_retries:
            self.nav_retries += 1
            self.retry_at = time.monotonic() + max(0.5, float(self.get_parameter('retry_delay').value))
            self.state = 'RETRY_WAIT'
            self.get_logger().warn(f'{wp}: {reason}；等待后重试 {self.nav_retries}/{max_retries}，不跳过该点')
        else:
            self.state = 'ERROR_STOP'
            self.get_logger().error(f'{wp}: {reason}；超过重试次数，巡检停止！请修复地图/航点，不会假报完成')
            self._publish(self.result_pub, f'{wp} 导航失败，任务未完成')


def main():
    rclpy.init()
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
