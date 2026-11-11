import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import pyttsx3
    _engine = pyttsx3.init()
    _TTS_AVAILABLE = True
except Exception:
    _engine = None
    _TTS_AVAILABLE = False


def _say_tts(text: str):
    _engine.say(text)
    _engine.runAndWait()


def speak(text: str) -> str:
    # A dedicated event marker is read by run.sh; all other output goes to launch.log.
    print(f"COMPETITION_EVENT|{text}", flush=True)
    if _TTS_AVAILABLE:
        try:
            _say_tts(text)
            return "tts"
        except Exception as exc:
            print(f"[播报] TTS 播放失败，仅记录日志：{exc}")
            return "log"
    return "log"


class AnnounceNode(Node):
    def __init__(self):
        super().__init__("announce_node")
        self.create_subscription(String, "/announce", lambda m: speak(m.data), 10)
        self.get_logger().info("播报节点已就绪，TTS 可用=" + str(_TTS_AVAILABLE))


def main():
    rclpy.init()
    node = AnnounceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
