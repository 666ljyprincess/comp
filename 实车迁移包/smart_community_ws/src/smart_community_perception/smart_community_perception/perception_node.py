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
        self.declare_parameter("people_weights", "")
        self.declare_parameter("people_confidence", 0.35)
        self.declare_parameter("people_process_interval", 0.35)
        self._last_people_run = 0.0
        self._last_save = 0.0
        self._save_idx = 0
        self.res_pub = self.create_publisher(PerceptionResult, f"/perception/{task}", 10)
        self.viz_pub = self.create_publisher(Image, f"/perception/{task}/annotated", 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value,
                                 self.cb, qos_profile_sensor_data)
        self._active = False
        self._current_waypoint = ''
        self.create_subscription(String, '/mission/current_task', self._on_current_task, 10)
        self.create_subscription(String, '/mission/active_task', self._on_active_task, 10)
        if self.task == 'people':
            # 模型装载/首轮推理在节点启动时进行，不占用到点后的 8 秒观察时间。
            self._prepare_people_model()
        elif self.task == 'license_plate':
            # PaddleOCR/TextRecognition startup must not consume the parking
            # waypoint's recognition timeout. Warm up BEFORE rclpy.spin().
            self._prepare_plate_model()
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

    def _prepare_people_model(self):
        self.detector.configure(self.get_parameter('people_weights').value,
                                self.get_parameter('people_confidence').value)
        if self.detector.error:
            self.get_logger().error('人偶模型预加载失败：' + self.detector.error)
            return
        try:
            # Ultralytics 首次 predict 有耗时初始化；空图预热结果不计入人偶统计。
            import numpy as np
            blank = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detector.model.predict(blank, conf=self.detector.conf,
                                        imgsz=self.detector.imgsz,
                                        verbose=False, device='cpu')
            self.detector.reset_temporal()
            self.get_logger().info('人偶 YOLO 模型已预加载并完成推理预热')
        except Exception as exc:
            self.get_logger().warning(f'人偶模型预热失败，仍可尝试正常识别：{exc}')

    def _prepare_plate_model(self):
        """Preload and warm up the same OCR recognizer used by PlateDetector.

        This is a synthetic test image, never a contest detection or vote.
        PaddleOCR full-pipeline fallback remains lazy to avoid preloading a
        second huge model unnecessarily.
        """
        try:
            from .detectors.plate import _get_line_recognizer
            model = _get_line_recognizer()
            if model is None:
                self.get_logger().warning(
                    '车牌 TextRecognition 预加载失败，检测时将尝试 OCR 备用路径')
                return
            import numpy as np
            blank = np.full((80, 320, 3), 255, dtype=np.uint8)
            model.predict(input=blank, batch_size=1)
            self.get_logger().info('车牌 OCR 模型已预加载并完成首次推理预热')
        except Exception as exc:
            self.get_logger().warning(
                f'车牌 OCR 预热异常（仍可尝试正常识别）：{exc}')

    def _on_current_task(self, msg):
        name = msg.data.split(':', 1)[0]
        if name != self._current_waypoint:
            self._current_waypoint = name
            if self.task == 'people':
                self.detector.reset_temporal()

    def _on_active_task(self, msg):
        active = msg.data == self.task
        if self.task == 'people' and active and not self._active:
            self.detector.reset_temporal()
            self._last_people_run = 0.0
        self._active = active

    def cb(self, msg):
        if not self._active:
            return
        if self.task == 'people':
            now = time.monotonic()
            if now - self._last_people_run < float(self.get_parameter('people_process_interval').value):
                return
            self._last_people_run = now
            self.detector.configure(self.get_parameter('people_weights').value,
                                    self.get_parameter('people_confidence').value)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.task == "traffic_light":
            # Parameters can be adjusted with ros2 param set (no restart needed).
            self.detector.set_roi(self.get_parameter("traffic_light_roi").value)
            self.detector.set_order(self.get_parameter("traffic_light_order").value)
        detections = self.detector.detect(img)
        summary, payload = self.summary_fn(detections)
        if self.task in ('people', 'license_plate'):
            payload['waypoint'] = self._current_waypoint
        if self.task == 'people':
            payload['status'] = 'OK' if not self.detector.error else self.detector.error
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
