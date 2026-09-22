"""Controller-only baseline: perfect reference from CSV, no camera.

Feeds lpv_mpc the ground-truth MIDDLE lane through the SAME /planning/ref_path
contract lane_assist uses (window_republisher slices a rolling open horizon),
but with geometry that is exact by construction. lane_detector is not started.

The point is to split one question into two:

  does the controller track a correct reference?      <- this launch
  does lane_assist produce a correct reference?       <- lane_keeping_launch

If tracking is good here, the controller is sound and lane_assist's job is
fully specified: match what window_republisher puts on the topic. If tracking
is bad here too, no amount of perception work will fix it.

window_m is the knob that matters. The camera can only ever see ~3 m, while
window_republisher's own default is 12 m because the MPC's horizon reach is
hz*Ts*v. Run it twice:

  ros2 launch lane_assist csv_reference_launch.py                  # 12 m, as designed
  ros2 launch lane_assist csv_reference_launch.py window_m:=2.3    # camera-length

If 12 m tracks well and 2.3 m reproduces the corner-cutting seen with the
camera, the fault is the horizon LENGTH, not the controller and not perception.

Ground truth used as a reference here deliberately. It normally lives outside
src/csv_data/ precisely so it cannot be selected by accident; this launch names
the full path so the choice is explicit.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LANE_CSV = os.path.expanduser(
    '~/sim_gazebo/src/lane_assist/ground_truth/superspeedway_lane_middle.csv')


def _log_dir():
    d = os.environ.get('LPV_MPC_LOG_DIR',
                       os.path.expanduser('~/sim_gazebo/src/lpv_mpc_gazebo/log'))
    os.makedirs(d, exist_ok=True)
    return d


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lpv_mpc_gazebo'), 'config',
        'lpv_mpc_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'window_m', default_value='12.0',
            description='Arc length of the horizon published ahead of the car. '
                        '12.0 = window_republisher default; 2.3 = what the '
                        'camera can actually see.'),
        DeclareLaunchArgument(
            'reference_csv', default_value=LANE_CSV,
            description='Ground-truth lane to follow (full path).'),
        DeclareLaunchArgument(
            'target_speed', default_value='1.5',
            description='Matches lane_assist target_speed so the two runs are '
                        'comparable.'),

        LogInfo(msg=['csv reference: ', LaunchConfiguration('reference_csv')]),
        LogInfo(msg=['window_m: ', LaunchConfiguration('window_m')]),

        Node(
            package='lpv_mpc_gazebo', executable='window_republisher',
            name='window_republisher', output='screen',
            parameters=[{
                'reference_csv': LaunchConfiguration('reference_csv'),
                'ref_path_topic': '/planning/ref_path',
                'window_m': LaunchConfiguration('window_m'),
                'use_sim_time': True,
            }],
        ),
        Node(
            package='lpv_mpc_gazebo', executable='lpv_mpc_node',
            name='lpv_mpc_node', output='screen',
            # Same three overrides as lane_keeping_launch, so the controller is
            # configured identically and only the reference source differs.
            parameters=[config, {
                'log_dir': _log_dir(),
                'config_file': config,
                'ref_source': 'topic',
                'ref_generator': 'external',
                'speed_scale': 1.0,
            }],
        ),
    ])
