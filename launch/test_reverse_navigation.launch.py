"""
reverse_navigation_launch.py

Launches the ReverseNavigationNode.

Only the 'namespace' parameter is declared here.
All other parameters are sourced from reverse_navigation.yaml.

Usage:
    ros2 launch <package> reverse_navigation_launch.py namespace:=/husky1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    
    # Get package share directory
    pkg_share = FindPackageShare('husky_operations_manager')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'reverse_navigation.yaml'])

    # ── Only argument declared in the launch file ────────────────────────────
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="/a300_00036",
        description="Robot namespace (e.g. /husky1)",
    )
    
    # ── Node ─────────────────────────────────────────────────────────────────
    reverse_nav_node = Node(
        package="husky_operations_manager",
        executable="reverse_navigation_node",
        name="reverse_navigation_node",
        namespace=LaunchConfiguration("namespace"),
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