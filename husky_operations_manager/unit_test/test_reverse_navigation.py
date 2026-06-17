#!/usr/bin/env python3
"""
Test node for ReverseDriveClient (action_clients/reverse_drive.py).

Dock poses and motion parameters are hardcoded below. Set the active dock
via the 'active_dock' ROS2 parameter at launch:

    ros2 run husky_operations_manager reverse_navigation_node \
        --ros-args -p active_dock:=unloading_station \
        -r __ns:=/husky_0

Valid values for active_dock: husky_charger, unloading_station
"""

import rclpy
from rclpy.node import Node

from husky_operations_manager.types import DockInstanceConfig, DockPose, ReverseDriveConfig
from husky_operations_manager.action_clients.reverse_drive import ReverseDriveClient
from husky_operations_manager.robot_enums import ReverseDriveStatus

# =============================================================================
# GLOBAL CONFIG — edit here instead of a YAML file
# =============================================================================

DOCK_CONFIGS: dict[str, DockInstanceConfig] = {
    'husky_charger': DockInstanceConfig(
        instance_name='husky_charger',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=0.8, y=-1.945, theta=0.0),
    ),
    'unloading_station': DockInstanceConfig(
        instance_name='unloading_station',
        type='simple_charging_dock',
        frame='map',
        pose=DockPose(x=0.85, y=1.60, theta=1.571),
    ),
}

MOTION_CONFIG = dict(
    dock_configs=DOCK_CONFIGS,
    plugin_name='simple_charging_dock',
    staging_x_offset=-1.5,
    staging_yaw_offset=0.0,
    base_frame='base_link',
    controller_frequency=50.0,
    v_linear_min=0.15,
    v_angular_max=0.25,
    linear_tolerance=0.05,
    angular_tolerance=0.1,
    dock_backwards=False,
)

DEFAULT_ACTIVE_DOCK = 'husky_charger'
TIMER_PERIOD        = 1.0  # seconds


# =============================================================================
# NODE
# =============================================================================

class ReverseNavigationNode(Node):

    def __init__(self):
        super().__init__('reverse_navigation_node')

        self.declare_parameter('active_dock', DEFAULT_ACTIVE_DOCK)
        active_dock = str(self.get_parameter('active_dock').value)

        if active_dock not in DOCK_CONFIGS:
            self.get_logger().error(
                f"Unknown active_dock='{active_dock}'. "
                f"Valid options: {list(DOCK_CONFIGS.keys())}. Shutting down."
            )
            raise SystemExit(1)

        config = ReverseDriveConfig(
            dock_names=[active_dock],
            **MOTION_CONFIG,
        )

        self.reverse_drive = ReverseDriveClient(self, config)
        self._started = False

        self._timer = self.create_timer(TIMER_PERIOD, self._tick)
        self.get_logger().info(
            f"ReverseNavigationNode started | "
            f"active_dock='{active_dock}' | "
            f"staging_x_offset={config.staging_x_offset}"
        )

    def _tick(self):
        if not self._started:
            self.reverse_drive.drive_to_staging()
            self._started = True
            return

        status = self.reverse_drive.get_status()

        if status == ReverseDriveStatus.DONE:
            self.get_logger().info("Reverse drive DONE — shutting down")
            self._shutdown()
        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error("Reverse drive ERROR — shutting down")
            self._shutdown()
        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning("Reverse drive CANCELED — shutting down")
            self._shutdown()

    def _shutdown(self):
        self._timer.cancel()
        self.destroy_node()
        rclpy.shutdown()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = ReverseNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
