from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import Command
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    # urdf 参数：实车用 real_robot.urdf.xacro（默认），仿真对照用 smart_community_robot.urdf.xacro
    xacro_file = PathJoinSubstitution([
        FindPackageShare('smart_community_description'), 'urdf',
        LaunchConfiguration('urdf')
    ])
    return LaunchDescription([
        DeclareLaunchArgument('urdf', default_value='real_robot.urdf.xacro'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[
                {'robot_description': Command(['xacro ', xacro_file])},
                {'use_sim_time': False}
            ]
        )
    ])
