"""Camera lane keeping: lane_detector -> /planning/ref_path -> LPV-MPC.

Detector params come from lane_assist/config/lane_assist_params.yaml.
MPC params come from lpv_mpc_gazebo/config/lpv_mpc_params.yaml, with three
overridden here so that tuned file stays untouched:

  ref_source    csv -> topic   take the horizon from the camera, not a CSV
  ref_generator -> external    lane_detector owns /planning/ref_path
  speed_scale   -> 1.0         so the detector's target_speed is the ONE speed
                               knob (the YAML's 0.55 would silently scale it)

window_republisher is deliberately NOT started — two publishers on the same
topic would fight.

  ros2 launch f1tenth_gym_gazebo sim_launch.py map_name:=superspeedway
  ros2 launch lane_assist lane_keeping_launch.py

Each run writes two CSVs, and nothing else:
  src/lane_assist/log/lane_<stamp>.csv  detector view — the horizon it published
                                        and the MPC prediction that came back
  src/lpv_mpc_gazebo/log/run.csv        controller view — tracking and timing
Both carry sim_t, so they join on the sim clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _resolve_log_dir():
    log_dir = os.environ.get(
        'LPV_MPC_LOG_DIR',
        os.path.expanduser('~/sim_gazebo/src/lpv_mpc_gazebo/log'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _resolve_lane_log_dir():
    log_dir = os.environ.get(
        'LANE_ASSIST_LOG_DIR',
        os.path.expanduser('~/sim_gazebo/src/lane_assist/log'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lpv_mpc_gazebo'), 'config',
        'lpv_mpc_params.yaml')
    lane_config = os.path.join(
        get_package_share_directory('lane_assist'), 'config',
        'lane_assist_params.yaml')

    return LaunchDescription([
        Node(
            package='lane_assist', executable='lane_detector',
            name='lane_detector', output='screen',
            parameters=[lane_config, {'log_dir': _resolve_lane_log_dir()}],
        ),
        Node(
            package='lpv_mpc_gazebo', executable='lpv_mpc_node',
            name='lpv_mpc_node', output='screen',
            parameters=[config, {
                'log_dir': _resolve_log_dir(),
                'config_file': config,
                'ref_source': 'topic',
                'ref_generator': 'external',
                'speed_scale': 1.0,
                # Camera-pipeline tuning. Lives here, not in the shared yaml,
                # because the CSV raceline runs on a 12 m+ reference and tracks
                # to 0.02 m with the yaml's values — detuning it for a 3 m
                # sightline would regress that run.
                'cmd_accel_horizon': 0.05,
                'speed_lookahead_time': 0.8,
                # [x_dot, psi, X, Y]. Ego frame: Y is cross-track, X along-track.
                'Q_diag': [10.0, 1000.0, 100.0, 500.0],
            }],
        ),
    ])
