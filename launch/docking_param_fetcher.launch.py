from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace (e.g. /a300_00036)'
    )

    namespace = LaunchConfiguration('namespace')

    docking_param_fetcher_node = Node(
        package='husky_operations_manager',
        executable='docking_param_fetcher',
        name='docking_param_fetcher',
        namespace=namespace,
        parameters=[{'namespace': namespace}],
        output='screen',
        remappings=[
            ('/tf','tf'),
            ('/tf_static','tf_static'),
        ]
    )

    return LaunchDescription([
        namespace_arg,
        docking_param_fetcher_node,
    ])