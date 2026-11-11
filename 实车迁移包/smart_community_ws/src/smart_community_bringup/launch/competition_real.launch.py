"""实车一键启动（比赛用 / 分阶段调试用）。

分阶段调试用法（推荐按此顺序调通）：
  ① 车端先起 ROS1 驱动（计划.md §1.5）+ PC 起桥（§1.4），本 launch 只起 TF/导航/任务；
      起来后先看桥来话题/TF：
      ros2 launch smart_community_bringup competition_real.launch.py use_navigation:=false
  ② 加导航遥控验证（RViz 里 2D Goal Pose 试走）：
      ros2 launch smart_community_bringup competition_real.launch.py
  ③ 纯巡航丝滑测试（无识别任务！所有点位只停不走任务）：
      ros2 launch smart_community_bringup competition_real.launch.py start_mission:=true waypoints:=<巡航yaml绝对路径>
  ④ 全任务（识别开启，weights 指向 PC 本地训练模型，~ 会自动展开；hfov/地标用实测值）：
      ros2 launch smart_community_bringup competition_real.launch.py start_mission:=true \
          use_perception:=true people_weights:=~/weights/people_best.pt \
          map:=~/maps/community_real.yaml \
          people_hfov:=<§2.3实测弧度> people_landmarks:=<§5.3重测yaml绝对路径>

关键差异（相对仿真 competition.launch.py）：
  - 无 gazebo；无实车驱动（车端 ROS1 自起，经桥转发，计划.md 方案 A）
  - TF 全 PC 自建：URDF 静态 TF + odom_tf_broadcaster（桥来 /odom → odom→base_link）
  - use_sim_time 全部 false
  - 导航默认不开 keepout（纯巡航阶段；跑顺后 use_keepout:=true 并重画实车禁行 mask）
  - waypoints 作为参数传入，默认巡航版（全部 task: turn，无识别依赖）
  - 感知默认不启动（阶段④才开；含人偶跨视角去重融合 people_zone_node——
    mission_manager 区域汇总自动优先其融合结果，节点不在/地标未重测/无匹配时
    1.5s 后回退为视角计数求和，行为与不装相同）
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PERCEPTION_TASKS = ["traffic_light", "people", "license_plate"]


def generate_launch_description():
    use_navigation = LaunchConfiguration('use_navigation')
    use_slam = LaunchConfiguration('use_slam')
    start_mission = LaunchConfiguration('start_mission')
    use_perception = LaunchConfiguration('use_perception')
    people_weights = LaunchConfiguration('people_weights')
    use_keepout = LaunchConfiguration('use_keepout')

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
        DeclareLaunchArgument('people_weights', default_value='',
                              description='Trained 2-class YOLO .pt（PC 本地路径，~ 可用）；空=无模型，people 检测报 MODEL_MISSING'),
        DeclareLaunchArgument('use_keepout', default_value='false'),
        DeclareLaunchArgument('navigation_mode', default_value='continuous'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('people_hfov', default_value='1.40',
                              description='相机水平 FOV 弧度；1.40 是 Gazebo 仿真值，实车按 计划.md §2.3 实测后覆盖'),
        DeclareLaunchArgument('image_width', default_value='1280',
                              description='相机实际输出分辨率宽（人偶投影参照系，须与检测框像素坐标一致）'),
        DeclareLaunchArgument('image_height', default_value='720',
                              description='相机实际输出分辨率高'),
        DeclareLaunchArgument(
            'people_landmarks',
            default_value=os.path.join(
                get_package_share_directory('smart_community_perception'),
                'config', 'people_landmarks.yaml'),
            description='人偶地标 yaml；默认是仿真坐标，实车重测后换绝对路径（计划.md §5.3）'),
        DeclareLaunchArgument(
            'waypoints',
            default_value=os.path.join(get_package_share_directory('smart_community_mission'),
                                       'config', 'waypoints_cruise.yaml'),
            description='巡航/任务航点 yaml 绝对路径；巡航版=全部 turn，任务版=识别任务点'),
        DeclareLaunchArgument('map', default_value=os.path.expanduser('~/maps/community_real.yaml')),
        # 1. 实车 URDF 静态 TF（base_link→雷达/相机）
        IncludeLaunchDescription(PythonLaunchDescriptionSource(desc)),
        # 2. odom→base_link TF（桥来 /odom，替代车端 TF；车端驱动按计划.md §1.5 自起）
        Node(package='smart_community_bringup', executable='odom_tf_broadcaster',
             output='screen', parameters=[{'use_sim_time': False}]),
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

    # 人偶跨视角去重融合（阶段④与感知一起开；汇总播报统一归 mission_manager，
    # 故 announce=False；区域汇总优先融合结果，无匹配/无响应自动回退视角求和）
    actions.append(TimerAction(period=6.5, actions=[
        Node(package='smart_community_perception', executable='people_zone_node',
             output='screen',
             condition=IfCondition(use_perception),
             parameters=[{
                 'use_sim_time': False,
                 'announce': False,
                 'hfov': ParameterValue(LaunchConfiguration('people_hfov'), value_type=float),
                 'image_width': ParameterValue(LaunchConfiguration('image_width'), value_type=int),
                 'image_height': ParameterValue(LaunchConfiguration('image_height'), value_type=int),
                 'landmarks_yaml': LaunchConfiguration('people_landmarks'),
             }])]))

    actions.append(OpaqueFunction(function=launch_mission))
    return LaunchDescription(actions)
