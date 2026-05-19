"""Launch file for Husky Operations Manager node."""

import os

from ament_index_python.packages import get_package_share_directory

from husky_operations_manager.substitutions import NamespacedYaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """Generate launch description for HuskyOperationsManager node."""
    config_path = os.path.join(
        get_package_share_directory('husky_operations_manager'),
        'config', 'test_lavender_harvest.yaml'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/a300_00036',
        description='Robot namespace'
    )

    namespace = LaunchConfiguration('namespace')

    group_action = GroupAction(
        [
            PushRosNamespace(namespace),
            Node(
                package='husky_operations_manager',
                executable='test_husky_ops_parameter_fetch',
                name='test_husky_ops_parameter_fetch',
                output='screen',
                parameters=[NamespacedYaml(source_file=config_path, namespace=namespace)],
                remappings=[
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static'),
                ],
            ),
        ]
    )

    return LaunchDescription(
        [
            namespace_arg,
            group_action,
        ]
    )
