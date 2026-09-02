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

import launch
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_logger = launch.logging.get_logger('husky_operations_manager.launch')


def _default_nav2_params_file() -> str:
    """Installed nav2.yaml from mtu32_bringup, or '' if that package is absent."""
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
        return os.path.join(
            get_package_share_directory('mtu32_bringup'),
            'config', 'a300', 'nav2.yaml')
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
    nav2_file    = LaunchConfiguration('nav2_params_file').perform(context)
    docking_params = _translate_docking_params(nav2_file)

    node = Node(
        package='husky_operations_manager',
        executable='husky_operations_manager',
        namespace=LaunchConfiguration('namespace'),
        name='husky_operations_manager',
        output='screen',
        # config.yaml first (defaults + dock_for_* mapping),
        # nav2-derived motion.* second so it wins.
        parameters=[config_file, docking_params],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )
    return [node]


def generate_launch_description():
    """Generate launch description for the HuskyOperationsManager node."""
    config_file = PathJoinSubstitution([
        FindPackageShare('husky_operations_manager'), 'config', 'config.yaml'])

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/a300_00036',
        description='Robot namespace')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to husky_operations_manager config.yaml')

    nav2_params_file_arg = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=_default_nav2_params_file(),
        description='Nav2 params file (nav2.yaml) whose docking_server block '
                    'is translated into motion.* params. Empty -> config.yaml only.')

    return LaunchDescription([
        namespace_arg,
        config_file_arg,
        nav2_params_file_arg,
        OpaqueFunction(function=_launch_setup),
    ])
