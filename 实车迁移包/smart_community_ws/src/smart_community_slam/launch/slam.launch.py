import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    slam_share = get_package_share_directory('slam_toolbox')
    our_share = get_package_share_directory('smart_community_slam')
    params = os.path.join(our_share, 'config', 'mapper_params_online_async.yaml')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_share, 'launch', 'online_async_launch.py')
            ),
            launch_arguments={'slam_params_file': params, 'use_sim_time': 'false'}.items()
        )
    ])
