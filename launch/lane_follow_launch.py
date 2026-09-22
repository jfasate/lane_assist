"""Camera lane keeping, no MPC: lane_detector -> pure pursuit -> /drive.

The diagnostic twin of lane_keeping_launch.py. Same detector, same reference
topic, but the controller is 40 lines of pure pursuit instead of the LPV-MPC
stack. Run it to find out which half of the pipeline is at fault:

  car keeps the lane  -> perception is sound; the MPC integration is the problem
  car does not        -> the fault is in lane_detector, debug it here where
                         there are two stages between camera and wheels

  ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:=superspeedway
  ros2 launch lane_assist lane_follow_launch.py

Writes two CSVs to src/lane_assist/log/:
  lane_<stamp>.csv    what the camera saw and the reference it published
  follow_<stamp>.csv  the goal point, steering and speed that came out
Both carry sim_t, so they join on the sim clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _log_dir():
    d = os.environ.get('LANE_ASSIST_LOG_DIR',
                       os.path.expanduser('~/sim_gazebo/src/lane_assist/log'))
    os.makedirs(d, exist_ok=True)
    return d


def generate_launch_description():
    lane_config = os.path.join(
        get_package_share_directory('lane_assist'), 'config',
        'lane_assist_params.yaml')
    log_dir = _log_dir()

    return LaunchDescription([
        DeclareLaunchArgument(
            'lookahead', default_value='1.2',
            description='Pure-pursuit lookahead [m]. Lower = tighter tracking '
                        'and more weave; the reference is only ~2.3 m long, so '
                        'much above 1.5 aims at the noisiest end of the fit.'),
        DeclareLaunchArgument(
            'max_speed', default_value='1.5',
            description='Cap on the speed the detector asks for [m/s].'),
        Node(
            package='lane_assist', executable='lane_detector',
            name='lane_detector', output='screen',
            parameters=[lane_config, {'log_dir': log_dir}],
        ),
        Node(
            package='lane_assist', executable='lane_follow_node',
            name='lane_follow_node', output='screen',
            parameters=[{
                'use_sim_time': True,
                'log_dir': log_dir,
                'lookahead': LaunchConfiguration('lookahead'),
                'max_speed': LaunchConfiguration('max_speed'),
            }],
        ),
    ])
