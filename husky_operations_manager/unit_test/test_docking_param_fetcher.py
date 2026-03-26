#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from husky_operations_manager.docking_param_fetcher import DockingParamFetcher
from husky_operations_manager.action_clients.reverse_drive_client import ReverseDriveClient
from husky_operations_manager.dataclass import DockingConfig
from husky_operations_manager.enum import DockingParamFetcherStatus, ReverseDriveStatus


class TestReverseDriveNode(Node):

    def __init__(self):
        super().__init__('test_reverse_drive')

        self.declare_parameter('namespace', '')
        namespace = (
            self.get_parameter('namespace')
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(f"TestReverseDriveNode started | namespace={namespace}")

        self._reverse_client: ReverseDriveClient | None = None
        self._last_status: ReverseDriveStatus | None    = None

        # Step 1 — fetch docking config
        self._param_fetcher = DockingParamFetcher(self)
        self._param_fetcher.fetch()

        # Poll at 2Hz — handles both fetch wait and drive status monitoring
        self.create_timer(0.5, self._poll)

    def _poll(self):
        # ---- Phase 1: wait for DockingConfig -------------------------
        if self._reverse_client is None:
            fetch_status = self._param_fetcher.get_status()

            if fetch_status == DockingParamFetcherStatus.ERROR:
                self.get_logger().error("DockingParamFetcher failed — cannot start test")
                return

            if fetch_status != DockingParamFetcherStatus.DONE:
                self.get_logger().info(
                    f"Waiting for DockingConfig... [{fetch_status.name}]"
                )
                return

            # Config ready — initialise and start reverse drive
            config: DockingConfig = self._param_fetcher.get_config()
            self._reverse_client  = ReverseDriveClient(self, config)

            self.get_logger().info("DockingConfig ready — starting reverse drive")
            success = self._reverse_client.drive_to_staging()

            if not success:
                self.get_logger().error(
                    "drive_to_staging() returned False — "
                    f"status: {self._reverse_client.get_status().name}"
                )
            return

        # ---- Phase 2: monitor drive status ---------------------------
        status = self._reverse_client.get_status()

        if status != self._last_status:
            self.get_logger().info(f"ReverseDriveClient status: {status.name}")
            self._last_status = status

        if status == ReverseDriveStatus.DONE:
            self.get_logger().info("=" * 50)
            self.get_logger().info("Test PASSED — staging pose reached")
            self.get_logger().info("=" * 50)

        elif status == ReverseDriveStatus.ERROR:
            self.get_logger().error("=" * 50)
            self.get_logger().error("Test FAILED — ReverseDriveClient in ERROR state")
            self.get_logger().error("=" * 50)

        elif status == ReverseDriveStatus.CANCELED:
            self.get_logger().warning("Drive was canceled")


def main(args=None):
    rclpy.init(args=args)
    node = TestReverseDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Stop robot on exit
        if node._reverse_client and node._reverse_client.is_active():
            node._reverse_client.cancel()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()