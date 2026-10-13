"""
Launch file for the Husky Operations Manager node.

Dock / motion configuration is sourced from the Nav2 docking_server params file
(nav2.yaml) at launch time, so there is a single source of truth. The
docking_server:ros__parameters block is translated into this node's motion.*
parameters and applied AFTER config.yaml, so nav2.yaml wins for any key it
provides. config.yaml keeps the motion.* keys as fallback defaults and owns
the task->dock mapping (motion.dock_for_charging / dock_for_unloading), which
Nav2 does not define.
"""

import os
import tempfile

import launch
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_logger = launch.logging.get_logger('husky_operations_manager.launch')


ROBOT_YAML = '/etc/clearpath/robot.yaml'


def _robot_yaml() -> dict:
    try:
        with open(ROBOT_YAML) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _robot_namespace() -> str:
    """robot.yaml's system.ros2.namespace, else $ROBOT_NAMESPACE, else ''."""
    ns = ((_robot_yaml().get('system') or {}).get('ros2') or {}).get('namespace') or os.environ.get('ROBOT_NAMESPACE', '')
    return str(ns).strip('/')


def _platform_model() -> str:
    """The platform model (j100 / a200 / a300) from robot.yaml's serial number, else 'a300'."""
    serial = str(_robot_yaml().get('serial_number') or '')
    model = serial.replace('cpr-', '').split('-')[0]
    return model or 'a300'


def _default_nav2_params_file() -> str:
    """The Nav2 params this robot's docking server actually runs with: the file bringup_nav2_map.launch.py writes
    (/tmp/<namespace>_nav2_map.yaml: the platform's nav2.yaml with the simulator's docks, when it runs in the
    sim), else the platform's installed config/<platform>/nav2.yaml from mtu32_bringup, else ''."""
    ns = _robot_namespace()
    rewritten = os.path.join(tempfile.gettempdir(), f'{ns}_nav2_map.yaml') if ns else ''
    if rewritten and os.path.isfile(rewritten):
        return rewritten
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
        return os.path.join(
            get_package_share_directory('mtu32_bringup'),
            'config', _platform_model(), 'nav2.yaml')
    except (ImportError, PackageNotFoundError):
        return ''


def _translate_docking_params(nav2_file: str) -> dict:
    """
    Read docking_server:ros__parameters from nav2_file and return a flat dict of
    this node's motion.* parameters. Returns {} (node falls back to config.yaml)
    if the file or the docking_server block cannot be read. Missing individual
    keys are warned about and left out so config.yaml supplies them.
    """
    if not nav2_file or not os.path.isfile(nav2_file):
        _logger.warning(
            f"nav2 params file not found: '{nav2_file}' — "
            "motion.* comes entirely from config.yaml")
        return {}

    try:
        with open(nav2_file, 'r') as f:
            data = yaml.safe_load(f) or {}
        if 'docking_server' not in data:  # a file written with a namespace root key
            data = next((v for v in data.values() if isinstance(v, dict) and 'docking_server' in v), data)
        ds = data['docking_server']['ros__parameters']
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        _logger.warning(
            f"could not read docking_server params from '{nav2_file}': {e} — "
            "motion.* comes entirely from config.yaml")
        return {}

    params: dict = {}
    missing: list = []

    def take(dst_key, container, *path, cast=None, default=None):
        node = container
        for p in path:
            if not isinstance(node, dict) or p not in node:
                if default is not None:
                    params[dst_key] = default
                else:
                    missing.append('.'.join(str(x) for x in path))
                return
            node = node[p]
        params[dst_key] = cast(node) if cast else node

    # Dock instances
    dock_names = ds.get('docks')
    if isinstance(dock_names, list) and dock_names:
        params['motion.dock_names'] = [str(n) for n in dock_names]
        for name in dock_names:
            d = ds.get(name)
            if not isinstance(d, dict):
                missing.append(name)
                continue
            take(f'motion.dock_configs.{name}.type', d, 'type', cast=str)
            take(f'motion.dock_configs.{name}.frame', d, 'frame', cast=str)
            if isinstance(d.get('pose'), list) and len(d['pose']) >= 3:
                params[f'motion.dock_configs.{name}.pose'] = [float(v) for v in d['pose'][:3]]
            else:
                missing.append(f'{name}.pose')
    else:
        missing.append('docks')

    # Staging offsets live under the first dock plugin
    plugins = ds.get('dock_plugins')
    plugin_name = plugins[0] if isinstance(plugins, list) and plugins else None
    plugin = ds.get(plugin_name, {}) if plugin_name else {}
    take('motion.staging_x_offset', plugin, 'staging_x_offset', cast=float)
    take('motion.staging_yaw_offset', plugin, 'staging_yaw_offset', cast=float, default=0.0)

    # Top-level docking_server params
    take('motion.base_frame', ds, 'base_frame', cast=str)
    take('motion.dock_backwards', ds, 'dock_backwards', cast=bool)
    take('motion.controller_frequency', ds, 'controller_frequency', cast=float)
    take('motion.linear_tolerance', ds, 'undock_linear_tolerance', cast=float)
    take('motion.angular_tolerance', ds, 'undock_angular_tolerance', cast=float)

    # Controller sub-block
    take('motion.v_linear_min', ds, 'controller', 'v_linear_min', cast=float)
    take('motion.v_angular_max', ds, 'controller', 'v_angular_max', cast=float)

    if missing:
        _logger.warning(
            f"nav2.yaml missing expected docking keys {missing} — "
            "config.yaml defaults used for those")

    _logger.info(
        f"motion params sourced from {nav2_file}: "
        + ', '.join(f'{k}={v}' for k, v in params.items()))
    return params


def _launch_setup(context, *args, **kwargs):
    config_file  = LaunchConfiguration('config_file').perform(context)
    nav2_file    = LaunchConfiguration('nav2_params_file').perform(context) or _default_nav2_params_file()
    docking_params = _translate_docking_params(nav2_file)

    # The dock a charging task undocks from when the task doesn't say: the launch argument, else the first dock
    # Nav2 knows (the one the robot's AprilTag matches in the sim), else config.yaml's.
    dock_for_charging = LaunchConfiguration('dock_for_charging').perform(context)
    if not dock_for_charging and docking_params.get('motion.dock_names'):
        dock_for_charging = docking_params['motion.dock_names'][0]
    extra = {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context) == 'true'}
    if dock_for_charging:
        extra['motion.dock_for_charging'] = dock_for_charging
    namespace = LaunchConfiguration('namespace').perform(context) or f'/{_robot_namespace()}'

    node = Node(
        package='husky_operations_manager',
        executable='husky_operations_manager',
        namespace=namespace,
        name='husky_operations_manager',
        output='screen',
        # config.yaml first (defaults + dock_for_* mapping),
        # nav2-derived motion.* second so it wins, then the launch arguments.
        parameters=[config_file, docking_params, extra],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        # a crash mid-task restarts the node: the fleet server republishes the task it was on
        respawn=True,
        respawn_delay=5.0,
    )
    return [node]


def generate_launch_description():
    """Generate launch description for the HuskyOperationsManager node."""
    config_file = PathJoinSubstitution([
        FindPackageShare('husky_operations_manager'), 'config', 'config.yaml'])

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description="Robot namespace (default: robot.yaml's system.ros2.namespace)")

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value=os.environ.get('USE_SIM_TIME', 'false'),
        choices=['true', 'false'],
        description='Run on the simulator clock (default: $USE_SIM_TIME or false)')

    dock_for_charging_arg = DeclareLaunchArgument(
        'dock_for_charging',
        default_value='',
        description='Dock a charging task undocks from when the task does not say (default: the first dock in '
                    'the Nav2 params file)')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to husky_operations_manager config.yaml')

    nav2_params_file_arg = DeclareLaunchArgument(
        'nav2_params_file',
        default_value='',
        description='Nav2 params file whose docking_server block is translated into motion.* params (default: '
                    'the file bringup_nav2_map.launch.py wrote, /tmp/<namespace>_nav2_map.yaml, else the '
                    "platform's nav2.yaml from mtu32_bringup)")

    return LaunchDescription([
        namespace_arg,
        use_sim_time_arg,
        dock_for_charging_arg,
        config_file_arg,
        nav2_params_file_arg,
        OpaqueFunction(function=_launch_setup),
    ])
