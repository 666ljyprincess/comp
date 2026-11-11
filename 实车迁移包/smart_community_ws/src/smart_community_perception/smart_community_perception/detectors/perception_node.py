import json
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from smart_community_interfaces.msg import PerceptionResult, PerceptionBox

from .annotator import annotate
from ament_index_python.packages import get_package_share_directory


def build_result(task: str, detections, summary: str, payload: dict) -> PerceptionResult:
    r = PerceptionResult()
    r.task = task
    r.summary = summary
    r.json = json.dumps(payload, ensure_ascii=False)
    for d in detections:
        b = PerceptionBox()
        b.x1, b.y1, b.x2, b.y2 = float(d.x1), float(d.y1), float(d.x2), float(d.y2)
        b.label = d.label
        b.score = float(d.score)
        r.boxes.append(b)
    return r


class PerceptionNode(Node):
    def __init__(self, task: str, detector, summary_fn):
        super().__init__(f"perception_{task}")
        self.task = task
        self.detector = detector
        self.summary_fn = summary_fn
        self.bridge = CvBridge()
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("enable_viz", True)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("save_prefix", task)
        self.declare_parameter("save_label", "")
        self.declare_parameter("save_interval", 1.0)
        # Normalized bbox of the THREE lamp lenses, for traffic_light only.
        self.declare_parameter("traffic_light_roi", "")
        self.declare_parameter("traffic_light_order", "AUTO")
        default_weights = os.path.join(get_package_share_directory('smart_community_perception'),
                                       'models', 'people_best.pt') if task == 'people' else ''
        self.declare_parameter("people_weights", default_weights)
        self.declare_parameter("people_confidence", 0.60)
        self.declare_parameter("people_min_box_height", 45)
        self.declare_parameter("people_min_aspect", 1.90)
        self.declare_parameter("people_max_aspect", 6.00)
        self.declare_parameter("people_edge_margin", 3)
        self.declare_parameter("people_confirm_frames", 2)
        self.declare_parameter("people_process_interval", 0.35)
        self._last_people_run = 0.0
        self._last_save = 0.0
        self._save_idx = 0
        self.res_pub = self.create_publisher(PerceptionResult, f"/perception/{task}", 10)
        self.viz_pub = self.create_publisher(Image, f"/perception/{task}/annotated", 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value,
                                 self.cb, qos_profile_sensor_data)
        self._active = False
        self.create_subscription(String, '/mission/active_task', self._on_active_task, 10)
        self.get_logger().info(f'感知节点启动：任务={task}，图像话题={self.get_parameter("image_topic").value}')
        if task == 'people':
            self.get_logger().info(
                f'人偶检测配置：权重={self.get_parameter("people_weights").value or "未设置"}，'
                f'置信度阈值={self.get_parameter("people_confidence").value}，'
                f'处理间隔={self.get_parameter("people_process_interval").value}s')
        elif task == 'traffic_light':
            self.get_logger().info(
                f'红绿灯检测配置：ROI={self.get_parameter("traffic_light_roi").value or "自动"}，'
                f'灯序={self.get_parameter("traffic_light_order").value}')
        self.get_logger().info(f'发布话题：结果=/perception/{task}，标注图像=/perception/{task}/annotated')

    def _on_active_task(self, msg):
        active = (msg.data == self.task)
        if self.task == 'people' and active != self._active:
            self.detector.reset_temporal()  # 切换任务时不能延续上一次画面的确认状态
        self._active = active

    def cb(self, msg):
        if not self._active:
            return
        if self.task == 'people':
            now = time.monotonic()
            if now - self._last_people_run < float(self.get_parameter('people_process_interval').value):
                return
            self._last_people_run = now
            self.detector.configure(
                self.get_parameter('people_weights').value,
                self.get_parameter('people_confidence').value,
                min_height=self.get_parameter('people_min_box_height').value,
                min_aspect=self.get_parameter('people_min_aspect').value,
                max_aspect=self.get_parameter('people_max_aspect').value,
                edge_margin=self.get_parameter('people_edge_margin').value,
                confirm_frames=self.get_parameter('people_confirm_frames').value)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.task == "traffic_light":
            # Parameters can be adjusted with ros2 param set (no restart needed).
            self.detector.set_roi(self.get_parameter("traffic_light_roi").value)
            self.detector.set_order(self.get_parameter("traffic_light_order").value)
        detections = self.detector.detect(img)
        summary, payload = self.summary_fn(detections)
        if self.task == 'people':
            payload['status'] = 'OK' if not self.detector.error else self.detector.error
            payload['raw_candidates'] = self.detector.last_raw_count
            payload['geometry_passed'] = self.detector.last_geometric_count
            payload['filtered_reasons'] = self.detector.last_rejected
            if self.detector.error:
                summary = '人偶YOLO模型未准备好（请设置people_weights并训练）'
                self.get_logger().warn(f'人偶检测器异常：{self.detector.error}', throttle_duration_sec=2.0)
        labels = [f'{d.label}({d.score:.2f})' for d in detections]
        self.get_logger().info(
            f'检测结果[{self.task}]：{summary}；目标数={len(detections)}；目标={labels or "无"}；详情={payload}',
            throttle_duration_sec=1.0)
        self.res_pub.publish(build_result(self.task, detections, summary, payload))
        if self.get_parameter("enable_viz").value or self.get_parameter("save_dir").value:
            annotated = annotate(img, detections, summary)
            if self.get_parameter("enable_viz").value:
                self.viz_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
            save_dir = self.get_parameter("save_dir").value
            if save_dir:
                self._maybe_save(annotated, detections, save_dir)

    def _maybe_save(self, annotated, detections, save_dir):
        label = self.get_parameter("save_label").value
        if label and not any(d.label == label for d in detections):
            return
        now = time.monotonic()
        if now - self._last_save < self.get_parameter("save_interval").value:
            return
        self._last_save = now
        os.makedirs(save_dir, exist_ok=True)
        prefix = self.get_parameter("save_prefix").value
        path = os.path.join(save_dir, f"{prefix}_{self._save_idx}.jpg")
        self._save_idx += 1
        cv2.imwrite(path, annotated)
        self.get_logger().info(f'已保存标注图像：{path}')


def main():
    import importlib

    import yaml
    from ament_index_python.packages import get_package_share_directory
    rclpy.init()
    cfg_path = os.path.join(get_package_share_directory('smart_community_perception'),
                            'config', 'tasks.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    launcher = Node('perception_launcher')
    launcher.declare_parameter('task', cfg['default_task'])
    task = launcher.get_parameter('task').value
    launcher.get_logger().info(f'读取任务配置：{cfg_path}')
    if task not in cfg['tasks']:
        launcher.get_logger().warn(f'未知任务「{task}」，回退到默认任务「{cfg["default_task"]}」')
        task = cfg['default_task']
    launcher.get_logger().info(f'本次运行任务：{task}')
    launcher.destroy_node()
    mod = importlib.import_module(f"smart_community_perception.detectors.{cfg['tasks'][task]['detector']}")
    node = PerceptionNode(task, getattr(mod, cfg['tasks'][task]['class'])(),
                          getattr(mod, 'summarize'))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
