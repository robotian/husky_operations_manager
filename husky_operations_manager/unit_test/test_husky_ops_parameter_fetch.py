"""Debug node for verifying Husky Operations Manager parameter fetch."""

from husky_operations_manager.dataclass import DriveConfig

import rclpy
from rclpy.node import Node

DOCKING_CONFIG_POLL_TIMEOUT_SEC = 30.0


class HuskyOperationsManager(Node):
    """Debug node: declares and reads all operations manager parameters."""

    def __init__(self):
        """Initialize the node, declare parameters, and start the timer."""
        super().__init__('test_husky_ops_parameter_fetch')

        self.namespace = self.get_namespace().rstrip('/')
        self.get_logger().info(f'Node namespace: {self.namespace}')

        self._declare_parameters()
        self._get_parameters()
        self._get_farm_row_parameters()

        self.create_timer(self.timing_timer_period, self.timer_callback)

    # =========================================================================
    # PARAMS AND VARIABLES INITIALIZATION
    # =========================================================================

    def _declare_parameters(self):
        """Declare all ROS2 parameters sourced from YAML at launch."""
        # General
        self.declare_parameter('num_rows', 10)
        self.declare_parameter('no_detection_timeout', 10.0)
        self.declare_parameter('harvest_duration', 5.0)
        self.declare_parameter('unload_duration', 5.0)
        self.declare_parameter('loading.increment', 20.0)

        # DriveClient
        self.declare_parameter('drive.base_frame', 'base_link')
        self.declare_parameter('drive.tf_polling_rate', 10.0)
        self.declare_parameter('drive.tf_base_frame', 'arm_0_base_link')
        self.declare_parameter('drive.tf_detection_frame', 'arm_0_detections')
        self.declare_parameter('drive.tolerance', 0.05)
        self.declare_parameter('drive.timeout', 30.0)
        self.declare_parameter('drive.v_linear', 0.2)
        self.declare_parameter('drive.v_angular', 0.5)

        # Navigation
        self.declare_parameter('navigation.max_retries', 3)
        self.declare_parameter('navigation.retry_delay', 5.0)

        # Docking
        self.declare_parameter('docking.max_retries', 2)
        self.declare_parameter('docking.retry_delay', 3.0)
        self.declare_parameter('docking.threshold', 0.25)

        # Battery
        self.declare_parameter('battery.low_threshold', 50.0)
        self.declare_parameter('battery.full_threshold', 99.0)

        # Timing
        self.declare_parameter('timing.timer_period', 1.0)
        self.declare_parameter('timing.initial_position_check_delay', 2.0)

    def _get_parameters(self):
        """Read all declared parameters into instance variables."""
        self.num_rows = int(self.get_parameter('num_rows').value)
        self.no_detection_timeout = float(self.get_parameter('no_detection_timeout').value)
        self.harvest_duration = float(self.get_parameter('harvest_duration').value)
        self.unload_duration = float(self.get_parameter('unload_duration').value)
        self.loading_increment = float(self.get_parameter('loading.increment').value)

        self.navigation_max_retries = int(self.get_parameter('navigation.max_retries').value)
        self.navigation_retry_delay = float(self.get_parameter('navigation.retry_delay').value)
        self.docking_max_retries = int(self.get_parameter('docking.max_retries').value)
        self.docking_retry_delay = float(self.get_parameter('docking.retry_delay').value)
        self.docking_threshold = float(self.get_parameter('docking.threshold').value)
        self.battery_low_threshold = float(self.get_parameter('battery.low_threshold').value)
        self.battery_full_threshold = float(self.get_parameter('battery.full_threshold').value)
        self.timing_timer_period = float(self.get_parameter('timing.timer_period').value)
        self.timing_initial_check_delay = float(self.get_parameter('timing.initial_position_check_delay').value)

        # DriveConfig
        self._drive_config = DriveConfig(
            base_frame=str(self.get_parameter('drive.base_frame').value),
            tolerance=float(self.get_parameter('drive.tolerance').value),
            timeout=float(self.get_parameter('drive.timeout').value),
            tf_polling_rate=float(self.get_parameter('drive.tf_polling_rate').value),
            tf_base_frame=str(self.get_parameter('drive.tf_base_frame').value),
            tf_detection_frame=str(self.get_parameter('drive.tf_detection_frame').value),
            v_linear=float(self.get_parameter('drive.v_linear').value),
            v_angular=float(self.get_parameter('drive.v_angular').value),
        )

        self.get_logger().info(
            f'Parameters loaded | rows={self.num_rows} | '
            f'harvest_duration={self.harvest_duration}s '
            f'unload_duration={self.unload_duration}s | '
            f'load_increment={self.loading_increment}% | '
            f'no_detection_timeout={self.no_detection_timeout}s'
        )

    def _get_farm_row_parameters(self):
        """Declare and load per-row waypoint parameters."""
        # Row waypoints — flat structure, indexed by row number
        # Supports up to 10 rows; unused rows are ignored at runtime
        for i in range(self.num_rows):
            self.declare_parameter(f'row_{i}_side_a_start', [0.0, 0.0, 0.0])
            self.declare_parameter(f'row_{i}_side_a_end', [0.0, 5.0, 0.0])
            self.declare_parameter(f'row_{i}_side_b_start', [0.5, 5.0, 3.14])
            self.declare_parameter(f'row_{i}_side_b_end', [0.5, 0.0, 3.14])

        # Load all row waypoints into a list of dicts
        self.row_waypoints: list[dict] = []
        for i in range(self.num_rows):
            self.row_waypoints.append(
                {
                    'side_a_start': list(self.get_parameter(f'row_{i}_side_a_start').value),
                    'side_a_end': list(self.get_parameter(f'row_{i}_side_a_end').value),
                    'side_b_start': list(self.get_parameter(f'row_{i}_side_b_start').value),
                    'side_b_end': list(self.get_parameter(f'row_{i}_side_b_end').value),
                }
            )

    # =========================================================================
    # MAIN CALLBACK METHODS
    # =========================================================================

    def timer_callback(self):
        """Log loaded parameter values for verification."""
        self.get_logger().info(
            f'rows={self.num_rows} | '
            f'drive_base={self._drive_config.base_frame} '
            f'tolerance={self._drive_config.tolerance} | '
            f'battery_low={self.battery_low_threshold}% '
            f'full={self.battery_full_threshold}% | '
            f'waypoints={self.row_waypoints}'
        )


def main():
    """Entry point: spin the parameter fetch debug node."""
    rclpy.init()
    node = HuskyOperationsManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
