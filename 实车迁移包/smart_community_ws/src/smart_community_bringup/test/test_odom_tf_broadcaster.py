"""odom_tf_broadcaster 行为单测（stub 掉 rclpy/nav_msgs/geometry_msgs/tf2_ros）。

WSL 无 ROS 运行时：用 stub 模块加载被测脚本，验证节点接线与 TF 内容。
钉住两条 Review Focus：entry 可执行名、TF 时间戳必须取 /odom 消息的 header.stamp。
"""
import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'odom_tf_broadcaster'


class _Param:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    """最小 Node 桩：记录 declare/create_subscription 调用。"""

    def __init__(self, node_name):
        self.node_name = node_name
        self.declared = {}
        self.subscriptions = []

    def declare_parameter(self, name, value):
        self.declared[name] = value

    def get_parameter(self, name):
        return _Param(self.declared[name])

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append(
            {'msg_type': msg_type, 'topic': topic, 'callback': callback, 'qos': qos})

    def get_logger(self):
        return SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)


def _install_stubs(capture):
    rclpy = types.ModuleType('rclpy')
    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = _FakeNode
    rclpy.node = node_mod
    rclpy.init = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None

    tf2_ros = types.ModuleType('tf2_ros')
    sent = []

    class _FakeBroadcaster:
        def __init__(self, node):
            pass

        def sendTransform(self, t):
            sent.append(t)

    tf2_ros.TransformBroadcaster = _FakeBroadcaster

    nav_msgs = types.ModuleType('nav_msgs')
    nav_msgs_msg = types.ModuleType('nav_msgs.msg')
    nav_msgs_msg.Odometry = type('Odometry', (), {})
    nav_msgs.msg = nav_msgs_msg

    geometry_msgs = types.ModuleType('geometry_msgs')
    gm_msg = types.ModuleType('geometry_msgs.msg')

    def _make_transform_stamped():
        # 模拟真实消息：实例化即带嵌套默认子对象
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
            child_frame_id='',
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    gm_msg.TransformStamped = _make_transform_stamped
    geometry_msgs.msg = gm_msg

    capture['sent'] = sent
    sys.modules.update({
        'rclpy': rclpy, 'rclpy.node': node_mod, 'tf2_ros': tf2_ros,
        'nav_msgs': nav_msgs, 'nav_msgs.msg': nav_msgs_msg,
        'geometry_msgs': geometry_msgs, 'geometry_msgs.msg': gm_msg,
    })


def _load_module():
    name = 'odom_tf_broadcaster_under_test'
    loader = SourceFileLoader(name, str(SCRIPT))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _make_odom_msg():
    stamp = SimpleNamespace(sec=112233, nanosec=456000000)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=1.5, y=-0.25, z=0.08),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.3826834, w=0.9238795))))


def test_subscribes_odom_and_broadcasts_tf_with_msg_stamp():
    capture = {}
    _install_stubs(capture)
    mod = _load_module()

    node = mod.OdomTfBroadcaster()

    # 接线：订阅 /odom，类型 nav_msgs/Odometry
    assert len(node.subscriptions) == 1
    sub = node.subscriptions[0]
    assert sub['topic'] == 'odom'
    assert sub['msg_type'].__name__ == 'Odometry'

    # TF 内容：stamp 必须是消息自带时间（不是墙钟），frame 默认 odom/base_link
    msg = _make_odom_msg()
    sub['callback'](msg)
    assert len(capture['sent']) == 1
    t = capture['sent'][0]
    assert t.header.stamp is msg.header.stamp, 'TF 时间戳必须取 /odom 的 header.stamp'
    assert t.header.frame_id == 'odom'
    assert t.child_frame_id == 'base_link'
    assert abs(t.transform.translation.x - 1.5) < 1e-9
    assert abs(t.transform.translation.y - (-0.25)) < 1e-9
    assert abs(t.transform.translation.z - 0.08) < 1e-9
    assert t.transform.rotation is msg.pose.pose.orientation


def test_frame_names_come_from_parameters():
    capture = {}
    _install_stubs(capture)
    mod = _load_module()

    node = mod.OdomTfBroadcaster()
    # 覆盖参数（模拟 launch parameters={'odom_frame': 'wheel_odom', ...}）
    node.declared['odom_frame'] = 'wheel_odom'
    node.declared['base_frame'] = 'chassis'

    sub = node.subscriptions[0]
    sub['callback'](_make_odom_msg())
    t = capture['sent'][-1]
    assert t.header.frame_id == 'wheel_odom'
    assert t.child_frame_id == 'chassis'
