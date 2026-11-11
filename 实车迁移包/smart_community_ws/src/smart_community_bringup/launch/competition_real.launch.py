"""实车一键启动（比赛用 / 分阶段调试用）。

分阶段调试用法（推荐按此顺序调通）：
  ① 只起驱动 + URDF 看话题/TF：
      ros2 launch smart_community_bringup competition_real.launch.py use_navigation:=false
  ② 加导航遥控验证（RViz 里 2D Goal Pose 试走）：
      ros2 launch smart_community_bringup competition_real.launch.py
  ③ 纯巡航丝滑测试（无识别任务！所有点位只停不走任务）：
      ros2 launch smart_community_bringup competition_real.launch.py start_mission:=true waypoints:=<巡航yaml绝对路径>
  ④ 全任务（识别开启，weights 指向实车训练模型）：
      ros2 launch smart_community_bringup competition_real.launch.py start_mission:=true \
          use_perception:=true people_weights:=/userdata/smart_community_ws/weights/people_best.pt

关键差异（相对仿真 competition.launch.py）：
  - 无 gazebo；改为 real_drivers.launch.py（底盘/雷达/相机实车驱动，需按实物改）
  - use_sim_time 全部 false
  - 导航默认不开 keepout（纯巡航阶段；跑顺后 use_keepout:=true 并重画实车禁行 mask）
  - waypoints 作为参数传入，默认巡航版（全部 task: turn，无识别依赖）
  - 感知默认不启动（阶段④才开）
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PERCEPTION_TASKS = ["traffic_light", "people", "license_plate"]


def generate_launch_description():
    use_navigation = LaunchConfiguration('use_navigation')
    use_slam = LaunchConfiguration('use_slam')
    start_mission = LaunchConfiguration('start_mission')
    use_perception = LaunchConfiguration('use_perception')
    people_weights = LaunchConfiguration('people_weights')
    use_keepout = LaunchConfiguration('use_keepout')

    bringup_share = get_package_share_directory('smart_community_bringup')
    drivers = os.path.join(bringup_share, 'launch', 'real_drivers.launch.py')
    desc = os.path.join(get_package_share_directory('smart_community_description'),
                        'launch', 'description.launch.py')
    nav = os.path.join(get_package_share_directory('smart_community_navigation'),
                       'launch', 'navigation.launch.py')
    slam = os.path.join(get_package_share_directory('smart_community_slam'),
                        'launch', 'slam.launch.py')

    def launch_mission(context):
        if LaunchConfiguration('start_mission').perform(context).lower() not in ('true', '1', 'yes'):
            return []
        mode = LaunchConfiguration('navigation_mode').perform(context).lower()
        if mode not in ('continuous', 'legacy'):
            raise RuntimeError('navigation_mode must be continuous or legacy')
        wp = LaunchConfiguration('waypoints').perform(context)
        if wp and not os.path.isfile(wp):
            raise RuntimeError('waypoints 文件不存在: ' + wp)
        return [TimerAction(period=8.0, actions=[Node(
            package='smart_community_mission',
            executable='continuous_mission_manager' if mode == 'continuous' else 'mission_manager',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'use_keepout': LaunchConfiguration('use_keepout').perform(context).lower() in ('true', '1', 'yes'),
                'waypoints_file': wp,
            }])])]

    actions = [
        DeclareLaunchArgument('use_navigation', default_value='true'),
        DeclareLaunchArgument('use_slam', default_value='false',
                              description='建图模式（此时应关导航）'),
        DeclareLaunchArgument('start_mission', default_value='false'),
        DeclareLaunchArgument('use_perception', default_value='false'),
        DeclareLaunchArgument('use_keepout', default_value='false'),
        DeclareLaunchArgument('navigation_mode', default_value='continuous'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument(
            'waypoints',
            default_value=os.path.join(get_package_share_directory('smart_community_mission'),
                                       'config', 'waypoints_cruise.yaml'),
            description='巡航/任务航点 yaml 绝对路径；巡航版=全部 turn，任务版=识别任务点'),
        DeclareLaunchArgument('map', default_value='/userdata/maps/community_real.yaml'),
        # 1. 实车驱动（底盘/雷达/相机 —— 按 real_drivers.launch.py 里的 TODO 改好）
        IncludeLaunchDescription(PythonLaunchDescriptionSource(drivers)),
        # 2. 实车 URDF TF
        IncludeLaunchDescription(PythonLaunchDescriptionSource(desc)),
        # 3. SLAM（建图时才开）
        TimerAction(period=2.0, actions=[
            IncludeLaunchDescription(PythonLaunchDescriptionSource(slam),
                                     condition=IfCondition(use_slam))]),
        # 4. Nav2（实车地图，默认 no keepout）
        TimerAction(period=4.0, actions=[
            IncludeLaunchDescription(PythonLaunchDescriptionSource(nav),
                launch_arguments={'use_keepout': use_keepout,
                                  'map': LaunchConfiguration('map')}.items(),
                condition=IfCondition(use_navigation))]),
    ]

    # 感知节点（阶段④才启动）
    for t in PERCEPTION_TASKS:
        params = [{'use_sim_time': False, 'task': t, 'image_topic': '/image_raw'}]
        if t == "people":
            params[0]['people_weights'] = people_weights
            params[0]['save_dir'] = os.path.expanduser('~/.ros/smart_community')
            params[0]['save_prefix'] = 'noncommunity'
            params[0]['save_label'] = '外来人员'
        actions.append(TimerAction(period=6.0, actions=[
            Node(package='smart_community_perception', executable='perception_node',
                 output='screen', parameters=params,
                 condition=IfCondition(use_perception))]))

    actions.append(OpaqueFunction(function=launch_mission))
    return LaunchDescription(actions)
