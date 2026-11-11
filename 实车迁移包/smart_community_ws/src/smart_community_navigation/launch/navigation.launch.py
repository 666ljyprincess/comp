"""Nav2 bringup（实车版）。

- use_sim_time 恒为 false
- map 作为 launch 参数传入（实车地图 /userdata/maps/community_real.yaml）
- use_keepout 默认 false：纯巡航阶段不开禁行层，跑顺后再开（需要实车 lane_keepout 地图）
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nav(context):
    nav2_share = get_package_share_directory('nav2_bringup')
    our_share = get_package_share_directory('smart_community_navigation')
    keepout = LaunchConfiguration('use_keepout').perform(context).lower() in ('true', '1', 'yes', 'on')
    map_yaml = LaunchConfiguration('map').perform(context)
    if not map_yaml or not os.path.isfile(map_yaml):
        raise RuntimeError(
            '地图文件不存在: %s\n请先用 slam.launch.py 建图并保存，再传 map:=<你的地图.yaml>' % map_yaml)
    params_name = 'nav2_params.yaml' if keepout else 'nav2_params_no_keepout.yaml'
    params = os.path.join(our_share, 'config', params_name)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_share, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': map_yaml, 'use_sim_time': 'false',
            'params_file': params, 'autostart': 'true'
        }.items())
    if not keepout:
        return [nav2]

    # keepout 模式：mask 地图必须换成实车地图上重画的版本
    mask_yaml = os.path.join(our_share, 'maps', 'lane_keepout_tuned.yaml')
    if not os.path.isfile(mask_yaml):
        raise RuntimeError('Missing keepout mask YAML: ' + mask_yaml)
    mask_server = Node(
        package='nav2_map_server', executable='map_server',
        name='keepout_mask_server', output='screen',
        parameters=[{'use_sim_time': False, 'yaml_filename': mask_yaml,
                     'topic_name': '/keepout_mask', 'frame_id': 'map'}])
    info_server = Node(
        package='nav2_map_server', executable='costmap_filter_info_server',
        name='keepout_filter_info_server', output='screen',
        parameters=[{'use_sim_time': False, 'type': 0,
                     'filter_info_topic': '/costmap_filter_info',
                     'mask_topic': '/keepout_mask', 'base': 0.0, 'multiplier': 1.0}])
    keepout_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_keepout', output='screen',
        parameters=[{'use_sim_time': False, 'autostart': True,
                     'node_names': ['keepout_mask_server', 'keepout_filter_info_server']}])
    return [mask_server, info_server, keepout_lifecycle, nav2]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='/userdata/maps/community_real.yaml',
                              description='实车地图 yaml 路径'),
        DeclareLaunchArgument('use_keepout', default_value='false'),
        OpaqueFunction(function=_launch_nav)
    ])
