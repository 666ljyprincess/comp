"""Zone-level multi-view fusion and one-time reporting after final people waypoint."""
import json
import math
import os

import rclpy
import yaml
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String
from smart_community_interfaces.msg import PerceptionResult
from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener

from .people_fusion import ZoneFusion


def quat_yaw(q):
    return math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                      1.0 - 2.0 * (q.y*q.y + q.z*q.z))


class PeopleZoneNode(Node):
    def __init__(self):
        super().__init__('people_zone_fusion')
        pkg = get_package_share_directory('smart_community_perception')
        mission_pkg = get_package_share_directory('smart_community_mission')
        self.declare_parameter('landmarks_yaml', os.path.join(pkg, 'config', 'people_landmarks.yaml'))
        self.declare_parameter('waypoints_yaml', os.path.join(mission_pkg, 'config', 'waypoints.yaml'))
        self.declare_parameter('hfov', 1.40)
        self.declare_parameter('min_frames_per_person', 2)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        # 汇总播报统一由 mission_manager 发布时置 False，避免双重语音
        self.declare_parameter('announce', True)
        with open(self.get_parameter('landmarks_yaml').value, encoding='utf-8') as f:
            landmarks = yaml.safe_load(f)['landmarks']
        with open(self.get_parameter('waypoints_yaml').value, encoding='utf-8') as f:
            waypoints = yaml.safe_load(f)['waypoints']
        self.name_zone = {w['name']: w['zone'] for w in waypoints
                          if w.get('task') == 'people' and 'zone' in w}
        self.fusion = ZoneFusion(landmarks, self.get_parameter('min_frames_per_person').value)
        self.tf = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.tf, self)
        self.task_name = ''
        self.state = ''
        self.ready = False
        self.result_pub = self.create_publisher(String, '/mission/people_zone_report', 10)
        self.speech_pub = self.create_publisher(String, '/announce', 10)
        self.create_subscription(String, '/mission/current_task', self.on_task, 10)
        self.create_subscription(String, '/mission/state', self.on_state, 10)
        self.create_subscription(String, '/mission/people_zone_done', self.on_done, 10)
        self.create_subscription(PerceptionResult, '/perception/people', self.on_people, 10)
        self.get_logger().info(f'人偶融合节点已就绪：{len(landmarks)} 个地标，{len(self.name_zone)} 个人偶巡航点')

    def on_task(self, msg):
        self.task_name = msg.data.split(':', 1)[0]

    def on_state(self, msg):
        self.state = msg.data

    def on_people(self, msg):
        zone = self.name_zone.get(self.task_name)
        if not zone or self.state != 'DWELL':
            return
        try:
            payload = json.loads(msg.json)
        except (ValueError, TypeError):
            return
        if payload.get('status') != 'OK':
            return
        # 与 mission_manager 相同的防串视角守卫：拒收上一视角的排队旧帧
        waypoint = payload.get('waypoint')
        if waypoint and waypoint != self.task_name:
            return
        boxes = [(b.x1, b.y1, b.x2, b.y2, b.label, b.score) for b in msg.boxes]
        if not boxes:
            self.fusion.frames_seen[zone] += 1
            return
        labels = [b[4] for b in boxes]
        self.get_logger().info(
            f'人偶识别帧：区域={zone}，检测到 {len(boxes)} 个目标：{labels}',
            throttle_duration_sec=2.0)
        try:
            trans = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time(),
                                             timeout=Duration(seconds=.07))
        except Exception as exc:
            self.get_logger().warn(f'无法获取 map->base_link 坐标变换：{exc}', throttle_duration_sec=5.0)
            return
        pose = (trans.transform.translation.x, trans.transform.translation.y,
                quat_yaw(trans.transform.rotation))
        self.fusion.add_frame(zone, boxes, pose,
                              int(self.get_parameter('image_width').value),
                              int(self.get_parameter('image_height').value),
                              float(self.get_parameter('hfov').value))

    def on_done(self, msg):
        zone = msg.data.strip().upper()
        if zone not in ('A', 'B'):
            return
        report = self.fusion.report(zone)
        text_msg = String()
        text_msg.data = json.dumps(report, ensure_ascii=False)
        self.result_pub.publish(text_msg)
        if report['total'] > 0:
            uncertain = (f'，另有{report["uncertain"]}名类别待核' if report['uncertain'] else '')
            sentence = (f'{zone}区已确认识别到{report["total"]}个人偶，'
                        f'其中居民{report["resident"]}名，外来人员{report["outsider"]}名'
                        f'{uncertain}。')
        else:
            sentence = f'{zone}区未获得可靠的人偶识别结果，无法确认人数，请检查模型和摄像头。'
        if self.get_parameter('announce').value:
            m = String()
            m.data = sentence
            self.speech_pub.publish(m)
        self.get_logger().info(f'人偶区域报告：{sentence}；数据={text_msg.data}')
        self.fusion.reset(zone)


def main():
    rclpy.init()
    node = PeopleZoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
