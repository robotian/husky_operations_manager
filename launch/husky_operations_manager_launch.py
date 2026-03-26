"""
Launch file for Husky Operations Manager node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for HuskyOperationsManager node"""
    
    # Get package share directory
    pkg_share = FindPackageShare('husky_operations_manager')
    config_file = PathJoinSubstitution([pkg_share, 'config', 'config.yaml'])
    
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/a300_00036',
        description='Robot namespace'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to configuration file'
    )
    
    # Create HuskyOperationsManager node
    husky_operations_manager_node = Node(
        package='husky_operations_manager',
        executable='husky_operations_manager',
        namespace=LaunchConfiguration('namespace'),
        name='husky_operations_manager',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
        ],
        remappings=[
            ('/tf','tf'),
            ('/tf_static','tf_static'),
        ]
    )
    
    # Create launch description
    ld = LaunchDescription([
        namespace_arg,
        config_file_arg,
        husky_operations_manager_node,
    ])
    
    return ld
