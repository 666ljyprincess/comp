from setuptools import setup

package_name = 'smart_community_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/waypoints.yaml', 'config/waypoints_cruise.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'mission_manager = smart_community_mission.mission_manager:main',
            'continuous_mission_manager = smart_community_mission.continuous_mission_manager:main',
            'waypoint_capture = smart_community_mission.waypoint_capture:main',
        ],
    },
)
