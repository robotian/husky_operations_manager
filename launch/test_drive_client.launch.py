"""
Launch file for TestDriveNode — DriveClient field testing and parameter tuning.

Usage:
  ros2 launch husky_operations_manager test_drive_client.launch.py
  ros2 launch husky_operations_manager test_drive_client.launch.py namespace:=/j100_0921

Tune parameters via config/test_drive_client.yaml without recompiling.
To flip center.y correction direction: set drive.center_y_correction_sign: -1.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = FindPackageShare('husky_operations_manager')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'test', 'test_drive_client.yaml'])

    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='/j100_0921', description='Robot namespace — must match live robot topics'
    )

    test_drive_node = Node(
        package='husky_operations_manager',
        executable='test_drive_client',
        name='test_drive_client',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[config_file],
        arguments=['--ros-args', '--log-level', 'DriveClient:=debug'],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )

    return LaunchDescription(
        [
            namespace_arg,
            test_drive_node,
        ]
    )
