import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ViewerNode(Node):
    def __init__(self):
        super().__init__("perception_viewer")
        self.bridge = CvBridge()
        # Subscribe directly: ROS topic discovery may not be complete at startup.
        subs = ['/perception/traffic_light/annotated',
                '/perception/people/annotated',
                '/perception/license_plate/annotated']
        self.windows = {}
        for t in subs:
            self.create_subscription(Image, t, self.make_cb(t), 10)
        self.get_logger().info(f'可视化窗口订阅到 {len(subs)} 个标注图像话题：{subs}')

    def make_cb(self, topic):
        def cb(msg):
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow(topic, img)
            cv2.waitKey(1)
        return cb


def main():
    rclpy.init()
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
