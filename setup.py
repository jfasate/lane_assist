import os
from glob import glob
from setuptools import setup

package_name = 'lane_assist'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='jayesh',
    maintainer_email='jayesh@todo.com',
    description='ViT lane assist: one camera image -> reference path + stop, with a lidar safety brake',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lane_follow_node = lane_assist.lane_follow_node:main',
            'road_collect = lane_assist.road_collect:main',
            'vit_lane_node = lane_assist.vit_lane_node:main',
            'lidar_guard = lane_assist.lidar_guard:main',
        ],
    },
)
