"""实车驱动启动：底盘 + 激光雷达 + 相机。

★ 根据你的实车实际驱动改这个文件，改完的验收标准（计划.md §1.4）：
    ros2 topic hz /scan    >= 5 Hz
    ros2 topic hz /odom    30–50 Hz
    ros2 topic hz /image_raw  15–30 Hz
    RViz 里 TF 树 map->odom->base_link->laser_link/camera_link 完整

以下按 OriginBot（RDK X5 + TROS）写的常见组合：
- 底盘：originbot_bringup（发 /odom、TF odom->base_link、收 /cmd_vel、IMU）
- 雷达：以实物型号为准（示例 websocket_lidar / ldlidar）
- 相机：以实物为准（示例 hobot_usb_cam，话题 /image_raw）
如果启动报找不到包，先 ros2 pkg list | grep -E "originbot|lidar|cam" 看板上实际装了哪些驱动。
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    actions = []

    # ---- 1. 底盘（OriginBot：编码器里程计 + IMU + TF odom->base_link）----
    try:
        chassis = os.path.join(get_package_share_directory('originbot_bringup'),
                               'launch', 'originbot.launch.py')
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(chassis)))
    except Exception:
        print('[real_drivers] 找不到 originbot_bringup，请改成你实车底盘驱动的 launch')

    # ---- 2. 激光雷达（发布 /scan，frame: laser_link）----
    # 按实物型号启用其一（注释掉不用的）：
    # try:
    #     lidar = os.path.join(get_package_share_directory('websocket_lidar'),
    #                          'launch', 'lidar.launch.py')
    #     actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(lidar)))
    # except Exception:
    #     print('[real_drivers] 找不到雷达驱动包，按实物型号修改此处')
    print('[real_drivers] TODO: 在此加入你的激光雷达驱动 launch（需发布 /scan）')

    # ---- 3. 相机（发布 /image_raw）----
    # try:
    #     cam = os.path.join(get_package_share_directory('hobot_usb_cam'),
    #                        'launch', 'usb_cam.launch.py')
    #     actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(cam)))
    # except Exception:
    #     print('[real_drivers] 找不到相机驱动包，按实物型号修改此处')
    print('[real_drivers] TODO: 在此加入你的相机驱动 launch（需发布 /image_raw）')

    # 雷达/相机稍后启动，等底盘先就绪
    return LaunchDescription([actions[0] if actions else TimerAction(period=0.0, actions=[])] +
                             [TimerAction(period=2.0, actions=actions[1:])])
