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
    people_weights = LaunchConfiguration('people_weights')
    use_keepout = LaunchConfiguration('use_keepout')

    def launch_mission(context):
        if LaunchConfiguration('start_mission').perform(context).lower() not in ('true', '1', 'yes'):
            return []
        mode = LaunchConfiguration('navigation_mode').perform(context).lower()
        if mode not in ('continuous', 'legacy'):
            raise RuntimeError('navigation_mode must be continuous or legacy')
        return [Node(package='smart_community_mission',
                     executable='continuous_mission_manager' if mode == 'continuous' else 'mission_manager',
                     output='screen', parameters=[{
                         'use_sim_time': True,
                         'use_keepout': LaunchConfiguration('use_keepout').perform(context).lower() in ('true', '1', 'yes')
                     }])]

    gazebo_launch = os.path.join(get_package_share_directory('smart_community_gazebo'), 'launch', 'gazebo.launch.py')
    nav_launch = os.path.join(get_package_share_directory('smart_community_navigation'), 'launch', 'navigation.launch.py')
    slam_launch = os.path.join(get_package_share_directory('smart_community_slam'), 'launch', 'slam.launch.py')

    actions = [
        DeclareLaunchArgument('use_navigation', default_value='true'),
        DeclareLaunchArgument('use_slam', default_value='false'),
        DeclareLaunchArgument('start_mission', default_value='false'),
        DeclareLaunchArgument('people_weights', default_value='', description='Trained 2-class YOLO .pt for resident/outsider'),
        DeclareLaunchArgument('use_keepout', default_value='true'),
        DeclareLaunchArgument('navigation_mode', default_value='continuous',
                              description='continuous (new) or legacy (old waypoint NavigateToPose)'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gazebo_launch)),
        TimerAction(period=4.0, actions=[
            IncludeLaunchDescription(PythonLaunchDescriptionSource(slam_launch), condition=IfCondition(use_slam))
        ]),
        TimerAction(period=5.0, actions=[
            IncludeLaunchDescription(PythonLaunchDescriptionSource(nav_launch),
                launch_arguments={'use_keepout': use_keepout}.items(),
                condition=IfCondition(use_navigation))
        ]),
    ]

    for t in PERCEPTION_TASKS:
        params = [{'use_sim_time': True, 'task': t}]
        if t == "people":
            params[0]['people_weights'] = people_weights
            params[0]['save_dir'] = os.path.expanduser('~/.ros/smart_community')
            params[0]['save_prefix'] = 'noncommunity'
            params[0]['save_label'] = '外来人员'
        actions.append(TimerAction(period=3.0, actions=[
            Node(package='smart_community_perception', executable='perception_node',
                 output='screen', parameters=params)]))

    # Person viewpoint and zone sums are now handled by mission_manager.
    # Do not launch the old fusion reporter or it would duplicate announcements.
    actions.append(TimerAction(period=4.0, actions=[
        Node(package='smart_community_perception', executable='announce_node', output='screen')]))
    actions.append(TimerAction(period=5.0, actions=[
        Node(package='smart_community_perception', executable='viewer_node', output='screen')]))
    actions.append(TimerAction(period=12.0, actions=[OpaqueFunction(function=launch_mission)]))
    return LaunchDescription(actions)
