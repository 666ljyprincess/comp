from setuptools import setup

package_name = 'smart_community_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.detectors'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/tasks.yaml', 'config/people_landmarks.yaml']),
        ('share/' + package_name + '/models', ['models/people_best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'perception_node = smart_community_perception.perception_node:main',
            'people_zone_node = smart_community_perception.people_zone_node:main',
            'announce_node = smart_community_perception.announce_node:main',
            'viewer_node = smart_community_perception.viewer_node:main',
        ],
    },
)
