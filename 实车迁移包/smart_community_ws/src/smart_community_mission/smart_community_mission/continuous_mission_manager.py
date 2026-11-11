#!/usr/bin/env python3
"""Collision-checked continuous Nav2 patrol, stopping ONLY at inspection tasks.

Perception and contest result logic is intentionally reused unchanged from
MissionManager. Turn waypoints are pass-through locations, not NavigateToPose goals.
Flow: ComputePathThroughPoses -> collision-checked SmoothPath -> FollowPath ->
(stop + settle at task + recognition) -> replan from actual current pose.
Near-coincident camera viewpoints are handled by an in-place Nav2 Spin.
"""
import math
import time

import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathThroughPoses, SmoothPath, FollowPath, Spin
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from rclpy.duration import Duration

from .grid_collision import as_grid, verify_polyline
from .mission_manager import MissionManager, quat_from_yaw
from .mission_lib import quat_to_yaw


class ContinuousMissionManager(MissionManager):
    def __init__(self):
        super().__init__()
        self.declare_parameter('use_keepout', True)
        self.declare_parameter('controller_id', 'SmoothFollow')
        self.declare_parameter('smoother_id', 'simple_smoother')
        self.declare_parameter('footprint_check_radius', 0.075)
        self.declare_parameter('nearby_spin_distance', 0.065)
        self.declare_parameter('settle_linear_speed', 0.035)
        self.declare_parameter('settle_angular_speed', 0.12)
        self.declare_parameter('settle_duration', 0.25)
        self.declare_parameter('settle_timeout', 4.0)
        self.declare_parameter('map_wait_timeout', 12.0)
        self.declare_parameter('max_smoothing_duration', 0.5)
        self.plan_client = ActionClient(self, ComputePathThroughPoses,
                                        '/compute_path_through_poses')
        self.smooth_client = ActionClient(self, SmoothPath, '/smooth_path')
        self.follow_client = ActionClient(self, FollowPath, '/follow_path')
        self.spin_client = ActionClient(self, Spin, '/spin')
        self.path_pub = self.create_publisher(Path, '/mission/continuous_path', 10)
        self.route_pub = self.create_publisher(Path, '/mission/route_history', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map',
                                 lambda msg: self._update_grid('map', msg), qos)
        self.create_subscription(OccupancyGrid, '/keepout_mask',
                                 lambda msg: self._update_grid('keepout', msg), qos)
        self.grids = {}
        self.pending_path = None
        self.pending_raw_path = None
        self.wait_map_since = None
        self.history_poses = []
        self.segment_waypoints = []
        self.target_wp_index = 0
        self.settle_since = None
        self.settle_began = None
        self.get_logger().info(
            '连续巡航已启用：途经点不停车，巡检识别点与红灯处才停车；'
            '规划/平滑和足迹碰撞检查均通过后才执行FollowPath。')

    def _update_grid(self, name, msg):
        self.grids[name] = as_grid(msg)

    def _all_grids_ready(self):
        return ('map' in self.grids and
                (not bool(self.get_parameter('use_keepout').value)
                 or 'keepout' in self.grids))

    def _safe_grids(self):
        grids = [self.grids['map']]
        if bool(self.get_parameter('use_keepout').value):
            grids.append(self.grids['keepout'])
        return grids

    def _servers_ready(self):
        clients = (self.plan_client, self.smooth_client,
                   self.follow_client, self.spin_client)
        return all(c.server_is_ready() for c in clients)

    def _pose(self, wp):
        qz, qw = quat_from_yaw(float(wp.get('yaw', 0)))
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(wp['x'])
        pose.pose.position.y = float(wp['y'])
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _actual_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame_id, 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=0.02))
        except Exception:
            return None
        pos, rot = tf.transform.translation, tf.transform.rotation
        return pos.x, pos.y, quat_to_yaw(rot.z, rot.w)

    def tick(self):
        super().tick()
        now = time.monotonic()
        if self.state == 'WAIT_MAP':
            if self._all_grids_ready():
                path = self.pending_path
                self.pending_path = None
                self._validate_and_follow(path)
            elif self.wait_map_since and now - self.wait_map_since > float(
                    self.get_parameter('map_wait_timeout').value):
                self._nav_failed('地图/禁行栅格未收到，不允许绕过碰撞校验执行')
        elif self.state == 'SETTLING':
            if self.settle_began and now - self.settle_began > float(
                    self.get_parameter('settle_timeout').value):
                self._nav_failed('机器人到达目标后未能稳定停止')
                return
            odom = self.last_odom
            if odom is None or self.last_odom_at is None or now - self.last_odom_at > 0.8:
                self.settle_since = None
                return
            v = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
            w = abs(odom.twist.twist.angular.z)
            if (v > float(self.get_parameter('settle_linear_speed').value)
                    or w > float(self.get_parameter('settle_angular_speed').value)):
                self.settle_since = None
                return
            pose = self._actual_pose()
            if pose is None:
                self.settle_since = None
                return
            x, y, yaw = pose
            goal = self.current
            dist = math.hypot(x - float(goal['x']), y - float(goal['y']))
            angle = math.atan2(math.sin(yaw - float(goal['yaw'])),
                               math.cos(yaw - float(goal['yaw'])))
            # Final stopping pose must agree with waypoint, not merely action status.
            if dist > 0.115 or abs(angle) > 0.24:
                self.settle_since = None
                return
            if self.settle_since is None:
                self.settle_since = now
            elif now - self.settle_since >= float(self.get_parameter('settle_duration').value):
                self._arrived()

    def send_next(self):
        if self.index >= len(self.waypoints):
            return super().send_next()  # Original FINISHED, final report logic.
        if not self._servers_ready():
            self.state = 'WAIT_NAV2'
            return
        # Skip traversal-only waypoints as stopping goals, but retain them as
        # planner *via* locations (the route cannot cut straight across blocks).
        start_index = self.route_start_index if self.nav_retries and hasattr(self, "route_start_index") else self.index
        self.route_start_index = start_index
        while self.index < len(self.waypoints) and self.waypoints[self.index].get('task') in ('turn', 'none', None):
            self.index += 1
        if self.index >= len(self.waypoints):
            return super().send_next()
        end_index = self.index
        self.segment_waypoints = self.waypoints[start_index:end_index + 1]
        self.target_wp_index = end_index
        self.current = self.waypoints[end_index]
        self.dwell_start = self.dwell_until = self.dwell_deadline = None
        self.accepted_frames = 0
        self.plate_votes.clear()
        self.people_samples = []
        self.people_first_frame_at = None
        self.plate_first_valid_at = None
        self.people_last_error = ''
        self.parking_stable_since = None
        self.parking_verified = False
        self.latest[self.current.get('task', 'none')] = 'NO_DATA'
        self.green_count = 0
        self.last_green_at = self.last_light_at = None
        self.light_state = 'UNKNOWN'
        self.last_reported_light = None
        self._active_task = 'none'
        self.goal_generation += 1
        generation = self.goal_generation
        self._publish(self.task_pub, f"{self.current['name']}:{self.current.get('task', 'none')}")
        current = self._actual_pose()
        self.get_logger().info(
            f'连续段 {start_index + 1}-{end_index + 1} / {len(self.waypoints)}：'
            f'途经 {[wp["name"] for wp in self.segment_waypoints[:-1]]}，'
            f'最终任务 {self.current["name"]}')
        # Two task points have almost identical (x,y), but different camera
        # headings. No need for artificial forward/backward wandering.
        if current is not None:
            gx, gy = float(self.current['x']), float(self.current['y'])
            if (len(self.segment_waypoints) == 1 and
                    math.hypot(current[0] - gx, current[1] - gy) <=
                    float(self.get_parameter('nearby_spin_distance').value)):
                return self._spin_in_place(generation, current[2])
        goals = [self._pose(wp) for wp in self.segment_waypoints]
        # For non-task via locations, orient along the NEXT segment. This is
        # just planner guidance; FollowPath does not stop at these orientations.
        for j in range(len(goals) - 1):
            a, b = self.segment_waypoints[j:j+2]
            theta = math.atan2(float(b['y']) - float(a['y']),
                               float(b['x']) - float(a['x']))
            goals[j].pose.orientation.z, goals[j].pose.orientation.w = quat_from_yaw(theta)
        req = ComputePathThroughPoses.Goal()
        req.goals = goals
        req.planner_id = 'GridBased'
        req.use_start = False
        self.state = 'PLANNING'
        self._send_action(self.plan_client, req, generation, 'PLANNING',
                          self._planned, '连续路径规划')

    def _send_action(self, client, request, generation, expected_state,
                     result_handler, label):
        try:
            future = client.send_goal_async(request)
            future.add_done_callback(
                lambda f, g=generation, st=expected_state, cb=result_handler, n=label:
                self._accepted(f, g, st, cb, n))
        except Exception as exc:
            self._nav_failed(f'{label}发送异常：{exc}')

    def _accepted(self, future, generation, expected_state, handler, label):
        if generation != self.goal_generation or self.state != expected_state:
            return
        try:
            handle = future.result()
            if handle is None or not handle.accepted:
                self._nav_failed(label + '目标被拒绝')
                return
            result = handle.get_result_async()
            result.add_done_callback(
                lambda f, g=generation, st=expected_state, cb=handler, n=label:
                self._action_result(f, g, st, cb, n))
        except Exception as exc:
            self._nav_failed(f'{label}异常：{exc}')

    def _action_result(self, future, generation, expected_state, handler, label):
        if generation != self.goal_generation or self.state != expected_state:
            return
        try:
            reply = future.result()
            if reply.status != GoalStatus.STATUS_SUCCEEDED:
                if expected_state == 'SMOOTHING' and self.pending_raw_path is not None:
                    self.get_logger().warning('Nav2平滑失败，检查原始规划路线能否安全执行')
                    self.pending_path = self.pending_raw_path
                    if self._all_grids_ready():
                        self._validate_and_follow(self.pending_raw_path)
                    else:
                        self.wait_map_since = time.monotonic()
                        self.state = 'WAIT_MAP'
                    return
                self._nav_failed(f'{label}未完成，状态码={reply.status}')
                return
            handler(reply.result, generation)
        except Exception as exc:
            self._nav_failed(f'{label}结果处理失败：{exc}')

    def _planned(self, result, generation):
        path = result.path
        if len(path.poses) < 2:
            return self._nav_failed('规划器返回空路径')
        tx, ty = float(self.current['x']), float(self.current['y'])
        tail = path.poses[-1].pose.position
        if math.hypot(tail.x - tx, tail.y - ty) > 0.10:
            return self._nav_failed('规划路径没有到达指定识别点（拒绝宽容差替代）')
        self.pending_raw_path = path
        req = SmoothPath.Goal()
        req.path = path
        req.smoother_id = str(self.get_parameter('smoother_id').value)
        req.max_smoothing_duration.sec = 0
        req.max_smoothing_duration.nanosec = int(
            max(0.05, min(0.9, float(self.get_parameter('max_smoothing_duration').value))) * 1e9)
        req.check_for_collisions = True
        self.state = 'SMOOTHING'
        self._send_action(self.smooth_client, req, generation, 'SMOOTHING',
                          self._smoothed, '路径平滑')

    def _nav_failed(self, reason):
        # Keep transit via-point constraints when replanning the same segment.
        super()._nav_failed(reason)

    def _smoothed(self, result, generation):
        path = result.path
        if len(path.poses) < 2:
            return self._nav_failed('平滑器返回空路径')
        self.pending_path = path
        if not self._all_grids_ready():
            self.wait_map_since = time.monotonic()
            self.state = 'WAIT_MAP'
            return
        self._validate_and_follow(path)

    def _validate_and_follow(self, path):
        radius = float(self.get_parameter('footprint_check_radius').value)
        points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        ok, reason = verify_polyline(points, self._safe_grids(), radius)
        if math.hypot(points[-1][0] - float(self.current['x']), points[-1][1] - float(self.current['y'])) > 0.10:
            return self._nav_failed('平滑后的路径终点偏离识别点，拒绝执行')
        if not ok:
            # Smoother may creep across a forbidden painted line. If original
            # collision-checked Nav2 path is safe, prefer it over unsafe smooth.
            raw = self.pending_raw_path
            if raw is not None:
                raw_points = [(p.pose.position.x, p.pose.position.y) for p in raw.poses]
                raw_ok, _ = verify_polyline(raw_points, self._safe_grids(), radius)
                if raw_ok:
                    path = raw
                    self.get_logger().warning('平滑轨迹触碰禁行区域，自动回退为安全的原始Nav2路线')
                else:
                    return self._nav_failed('足迹碰撞检查失败：' + reason)
            else:
                return self._nav_failed('足迹碰撞检查失败：' + reason)
        # Force only final viewpoint orientation, intermediate rotations are
        # governed by tangent/path curvature, not the old turn_X yaw.
        path.poses[-1].pose.orientation.z, path.poses[-1].pose.orientation.w = quat_from_yaw(
            float(self.current.get('yaw', 0.0)))
        self.path_pub.publish(path)
        self.history_poses.extend(path.poses if not self.history_poses else path.poses[1:])
        hist = Path()
        hist.header.frame_id = self.frame_id
        hist.header.stamp = self.get_clock().now().to_msg()
        hist.poses = self.history_poses
        self.route_pub.publish(hist)
        req = FollowPath.Goal()
        req.path = path
        req.controller_id = str(self.get_parameter('controller_id').value)
        req.goal_checker_id = 'goal_checker'
        self.state = 'NAVIGATING'
        self.get_logger().info(f'安全连续轨迹已下发：{len(path.poses)} 个采样点，开始平滑跟踪')
        self._send_action(self.follow_client, req, self.goal_generation,
                          'NAVIGATING', self._followed, '连续轨迹跟踪')

    def _followed(self, result, generation):
        self.state = 'SETTLING'
        self.settle_began = time.monotonic()
        self.settle_since = None
        self.get_logger().info(f'{self.current["name"]}：已完成路径跟踪，确认静止和观察方向')

    def _spin_in_place(self, generation, actual_yaw):
        desired = float(self.current.get('yaw', 0.0))
        delta = math.atan2(math.sin(desired - actual_yaw),
                           math.cos(desired - actual_yaw))
        if abs(delta) < 0.08:
            return self._followed(None, generation)
        request = Spin.Goal()
        request.target_yaw = float(delta)
        request.time_allowance.sec = 15
        self.state = 'SPINNING'
        self.get_logger().info(f'{self.current["name"]}：近距离视角切换，原地旋转 {delta:.2f} rad')
        self._send_action(self.spin_client, request, generation, 'SPINNING',
                          self._followed, '安全原地转向')


def main():
    rclpy.init()
    node = ContinuousMissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
