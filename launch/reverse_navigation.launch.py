"""
reverse_navigation_launch.py

Launches the ReverseNavigationNode.

Only the 'namespace' parameter is declared here.
All other parameters are sourced from reverse_navigation.yaml.

Usage:
    ros2 launch <package> reverse_navigation_launch.py namespace:=/husky1
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:

    # docking_param = PathJoinSubstitution([get_package_share_directory('mtu32_bringup'), 'config', 'a300', 'nav2.yaml'])

    # ── Only argument declared in the launch file ────────────────────────────
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="",
        description="Robot namespace (e.g. /husky1)",
    )

    namespace = LaunchConfiguration("namespace")

    # ── Path to parameter file ───────────────────────────────────────────────
    config_file = PathJoinSubstitution([
        get_package_share_directory("husky_operations_manager"),
        "config",
        "reverse_navigation.yaml",
        ])

    # ── Node ─────────────────────────────────────────────────────────────────
    reverse_nav_node = Node(
        package="husky_operations_manager",
        executable="reverse_navigation_node",
        name="reverse_navigation_node",
        namespace=namespace,
        parameters=[config_file],
        output="screen",
        remappings=[
            ('/tf','tf'),
            ('/tf_static','tf_static'),
        ]
    )

    return LaunchDescription([
        namespace_arg,
        reverse_nav_node,
    ])