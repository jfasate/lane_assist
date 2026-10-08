"""Camera lane keeping with the trained ViT: vit_lane_node -> pure pursuit -> /drive.

lane_detector is NOT started: the reference path comes only from the model.

  ros2 launch f1tenth_gym_gazebo sim_launch.py
  ros2 launch lane_assist vit_follow_launch.py        # lookahead:=1.2 max_speed:=1.5 lidar_guard:=true

The checkpoint is vit_lane_node.vit_checkpoint in lane_assist_params.yaml.
Watch /vit_lane/debug_image in RViz (prediction in blue on the camera frame)
and /planning/ref_path_viz (the path in the map).

Every run is logged to its own folder, log/vit_run_<stamp>/ (vit.csv, frames/,
follow_*.csv). Score it afterwards:
  /usr/bin/python3 src/lane_assist/tools/analysis/analyze_run.py      # newest run
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('lane_assist'), 'config',
                          'lane_assist_params.yaml')
    log_root = os.environ.get('LANE_ASSIST_LOG_DIR',
                              os.path.expanduser('~/sim_gazebo/src/lane_assist/log'))
    log_dir = os.path.join(log_root, 'vit_run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    return LaunchDescription([
        DeclareLaunchArgument('lookahead', default_value='1.2',
                              description='Pure-pursuit lookahead [m].'),
        DeclareLaunchArgument('max_speed', default_value='1.5',
                              description='Cap on the speed the ViT path asks for [m/s].'),
        DeclareLaunchArgument('lidar_guard', default_value='true',
                              description='Lidar emergency brake (lidar_guard -> follower speed cap).'),
        Node(package='lane_assist', executable='lidar_guard', name='lidar_guard', output='screen',
             parameters=[config], condition=IfCondition(LaunchConfiguration('lidar_guard'))),
        Node(package='lane_assist', executable='vit_lane_node', name='vit_lane_node',
             output='screen', parameters=[config, {'log_dir': log_dir}]),
        Node(package='lane_assist', executable='lane_follow_node', name='lane_follow_node',
             output='screen',
             parameters=[{'use_sim_time': True, 'log_dir': log_dir,
                          'lookahead': LaunchConfiguration('lookahead'),
                          'max_speed': LaunchConfiguration('max_speed'),
                          'free_topic': PythonExpression(["'/lane_assist/lidar_free' if '",
                                                          LaunchConfiguration('lidar_guard'),
                                                          "' == 'true' else ''"])}]),
    ])
