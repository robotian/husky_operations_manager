"""TaskGenerator — generates typed Task messages for the robot fleet."""

import csv
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np

from rclpy.impl.rcutils_logger import RcutilsLogger

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from status_interfaces.msg import DockGoal, SubTask, Task, UndockGoal, WayPoint


logger = RcutilsLogger(os.path.basename(__file__))

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_COST_MATRIX_CSV = os.path.join(_DATA_DIR, 'cost_matrix.csv')
_NAV_NODES_CSV = os.path.join(_DATA_DIR, 'farm_node_for_image_detection_nav.csv')

_NODE_TYPE_HARVEST = 4
_NODE_TYPE_CHARGING = 5
_NODE_TYPE_UNLOADING = 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TaskGeneratorConfig:
    """Deployment-specific parameters for task generation."""

    dock_id: str = 'husky_charger'
    dock_type: str = 'simple_charging_dock'
    undock_timeout: float = 30.0
    robot_id: int = 4
    crop_type: str = 'lavender'


@dataclass
class FarmNodes:
    """Class representing a farm edge."""

    id: int
    zone_prefix: str
    x: float
    y: float
    type: float
    theta: float


# ---------------------------------------------------------------------------
# RouteCalculator
# ---------------------------------------------------------------------------


class RouteCalculator:
    """Load the farm graph from CSV files and compute shortest paths between nodes."""

    def __init__(
        self,
        nav_nodes_csv: str,
        cost_matrix_csv: str,
    ):
        """Load node positions and pre-compute the all-pairs shortest-path matrix."""
        self._nodes = self._load_nav_nodes(nav_nodes_csv)

        csgraph = self._load_adjacency_matrix(cost_matrix_csv)
        dist_matrix, predecessors = shortest_path(
            csgraph=csgraph, method='auto', directed=True, return_predecessors=True
        )
        self._dist_matrix = dist_matrix
        self._predecessors = predecessors

        logger.debug(f'RouteCalculator ready — {len(self._nodes)} nodes')

    @classmethod
    def from_default_data_dir(cls) -> 'RouteCalculator':
        """Construct from the package-default data directory."""
        return cls(
            nav_nodes_csv=_NAV_NODES_CSV,
            cost_matrix_csv=_COST_MATRIX_CSV,
        )

    def build_route_to_node(
        self, robot_x: float, robot_y: float, destination_type: int
    ) -> list[WayPoint] | None:
        """
        Return WayPoints from the nearest farm node to (robot_x, robot_y) to the closest
        node of destination_type.

        Returns None when no matching destination node exists or no path is reachable.
        """
        dest_nodes = [n for n in self._nodes if n.type == destination_type]
        if not dest_nodes:
            logger.error(f'build_route_to_node: no node with type={destination_type} in farm graph')
            return None
        dest_node_idx = self._nodes.index(dest_nodes[0])

        distances = [math.hypot(robot_x - n.x, robot_y - n.y) for n in self._nodes]
        start_node_idx = int(np.argmin(distances))

        path_indices = self._shortest_path_to_destination(start_node_idx, dest_node_idx)

        if path_indices is None:
            logger.warning(
                f'No path from node idx {start_node_idx} (id={self._nodes[start_node_idx].id}) '
                f'to destination node idx {dest_node_idx} (type={destination_type})'
            )
            return None

        waypoints = []
        for idx in path_indices:
            node = self._nodes[idx]
            wp = WayPoint()
            wp.x = node.x
            wp.y = node.y
            wp.theta = node.theta
            wp.node_id = node.id
            waypoints.append(wp)

        return waypoints

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _load_nav_nodes(nav_nodes_csv: str) -> list[FarmNodes]:
        """Load FarmNodes from the navigation nodes CSV, sorted by id."""
        nodes = []
        with open(nav_nodes_csv, newline='') as f:
            for row in csv.DictReader(f):
                nodes.append(FarmNodes(
                    id=int(row['id']),
                    zone_prefix=row['zone_prefix'],
                    x=float(row['x']),
                    y=float(row['y']),
                    type=float(row['type']),
                    theta=float(row['theta']),
                ))
        return sorted(nodes, key=lambda n: n.id)

    @staticmethod
    def _load_adjacency_matrix(cost_matrix_csv: str) -> coo_matrix:
        """Load cost_matrix.csv as a sparse adjacency matrix."""
        cost_matrix = np.loadtxt(cost_matrix_csv, delimiter=',')
        return coo_matrix(cost_matrix)

    def _reconstruct_path(self, start: int, end: int) -> list[int] | None:
        """Reconstruct the path from start to end using the predecessors array."""
        if np.isinf(self._dist_matrix[start, end]):
            return None

        path = []
        i = end
        while i != start:
            path.append(int(i))
            i = self._predecessors[start, i]
        path.append(start)
        return path[::-1]

    def _shortest_path_to_destination(self, start_node: int, end_node: int) -> list[int] | None:
        """Return the node index list for the shortest route, or None if unreachable."""
        logger.debug(
            f'Routing: node idx {start_node} (id={self._nodes[start_node].id}) '
            f'→ node idx {end_node} (id={self._nodes[end_node].id})'
        )
        path_nodes = None
        try:
            path_nodes = self._reconstruct_path(start_node, end_node)
            logger.info(f'Route node idx {start_node} → node idx {end_node}: {path_nodes}')
        except Exception as e:
            logger.error(f'Routing error: {e}\n{traceback.format_exc()}')
        return path_nodes


# ---------------------------------------------------------------------------
# TaskGenerator
# ---------------------------------------------------------------------------


class TaskGenerator:
    """Build Task messages (with sub-tasks and waypoints) for the robot fleet."""

    def __init__(
        self,
        route_calculator: RouteCalculator | None = None,
        config: TaskGeneratorConfig | None = None,
    ):
        """Initialise with an optional RouteCalculator and config.

        Defaults to the package CSV files and the TaskGeneratorConfig defaults.
        """
        self._router = route_calculator or RouteCalculator.from_default_data_dir()
        self._config = config or TaskGeneratorConfig()

    # =========================================================================
    # FLEET TASK GENERATORS — multi-subtask combined tasks for JobPublisher
    # =========================================================================

    def generate_harvest_task(
        self,
        job_id: int,
        robot_x: float,
        robot_y: float,
        need_row_navigation: bool = True,
    ) -> Task | None:
        """Create a harvest task: MOVING → HARVESTING.

        When need_row_navigation is True, routes from the nearest graph node to
        (robot_x, robot_y) to the closest harvest node (type=4). When False,
        move_sub carries empty waypoints — DriveClient starts immediately.
        Returns None if need_row_navigation is True but no route exists.
        """
        if need_row_navigation:
            waypoints = self._router.build_route_to_node(robot_x, robot_y, _NODE_TYPE_HARVEST)
            if waypoints is None:
                logger.error(
                    f'generate_harvest_task: no route from ({robot_x:.2f}, {robot_y:.2f})'
                )
                return None
        else:
            waypoints = []

        move_sub = SubTask()
        move_sub.sub_task_id = 1
        move_sub.type = SubTask.MOVING
        move_sub.description = 'Move to harvest row'
        move_sub.data = waypoints

        harvest_sub = SubTask()
        harvest_sub.sub_task_id = 2
        harvest_sub.type = SubTask.HARVESTING
        harvest_sub.description = 'Harvesting the bush'
        harvest_sub.data_str = 'harvest data'

        task = Task(
            task_id=job_id,
            task_type=Task.HARVESTING_TASK,
            assigned_robot_id=self._config.robot_id,
            description=f'Harvesting Job {job_id}',
            crop_type=self._config.crop_type,
            sub_tasks=[move_sub, harvest_sub],
        )

        logger.debug(f'Harvest task generated: {task}')
        return task

    def generate_charging_task(self, robot_x: float, robot_y: float) -> Task | None:
        """
        Create a charging task: MOVING → DOCKING → CHARGING.

        Routes from the nearest graph node to (robot_x, robot_y) to the staging_pose.
        Returns None if no route exists.
        """
        waypoints = self._router.build_route_to_node(robot_x, robot_y, _NODE_TYPE_CHARGING)
        if waypoints is None:
            logger.error(
                f'generate_charging_task: no route from ({robot_x:.2f}, {robot_y:.2f})'
            )
            return None

        dock_goal = DockGoal(
            use_dock_id=True,
            dock_id=self._config.dock_id,
            navigate_to_staging_pose=True,
        )
        undock_goal = UndockGoal(
            dock_type=self._config.dock_type,
            max_undocking_time=self._config.undock_timeout,
        )

        move_sub = SubTask()
        move_sub.sub_task_id = 1
        move_sub.type = SubTask.MOVING
        move_sub.description = 'Move to charging node'
        move_sub.data = waypoints

        docking_sub = SubTask()
        docking_sub.sub_task_id = 2
        docking_sub.type = SubTask.DOCKING
        docking_sub.description = 'Dock at charging station'
        docking_sub.dock_goal = dock_goal

        charging_sub = SubTask()
        charging_sub.sub_task_id = 3
        charging_sub.type = SubTask.CHARGING
        charging_sub.description = 'Charge at charging station'
        charging_sub.undock_goal = undock_goal

        task = Task(
            task_id=1,
            task_type=Task.CHARGING_TASK,
            assigned_robot_id=self._config.robot_id,
            description='Charging Job',
            crop_type='',
            sub_tasks=[move_sub, docking_sub, charging_sub],
        )

        logger.debug(f'Charging task generated: {task}')
        return task

    def generate_unloading_task(self, robot_x: float, robot_y: float) -> Task | None:
        """
        Create an unloading task: MOVING → DOCKING → UNLOADING.

        Routes from the nearest graph node to (robot_x, robot_y) to the staging_pose.
        Returns None if no route exists.
        """
        waypoints = self._router.build_route_to_node(robot_x, robot_y, _NODE_TYPE_UNLOADING)
        if waypoints is None:
            logger.error(
                f'generate_unloading_task: no route from ({robot_x:.2f}, {robot_y:.2f})'
            )
            return None

        dock_goal = DockGoal(
            use_dock_id=True,
            dock_id=self._config.dock_id,
            navigate_to_staging_pose=True,
        )
        undock_goal = UndockGoal(
            dock_type=self._config.dock_type,
            max_undocking_time=self._config.undock_timeout,
        )

        move_sub = SubTask()
        move_sub.sub_task_id = 1
        move_sub.type = SubTask.MOVING
        move_sub.description = 'Move to unloading node'
        move_sub.data = waypoints

        docking_sub = SubTask()
        docking_sub.sub_task_id = 2
        docking_sub.type = SubTask.DOCKING
        docking_sub.description = 'Dock at unloading station'
        docking_sub.dock_goal = dock_goal

        unloading_sub = SubTask()
        unloading_sub.sub_task_id = 3
        unloading_sub.type = SubTask.UNLOADING
        unloading_sub.description = 'Unload at unloading station'
        unloading_sub.undock_goal = undock_goal

        task = Task(
            task_id=1,
            task_type=Task.UNLOADING_TASK,
            assigned_robot_id=self._config.robot_id,
            description='Unloading Job',
            crop_type='',
            sub_tasks=[move_sub, docking_sub, unloading_sub],
        )

        logger.debug(f'Unloading task generated: {task}')
        return task

    @staticmethod
    def build_row_nav_waypoints(
        target_x: float,
        target_y: float,
        target_theta: float,
        via_pose: list[float] | None = None,
    ) -> list[WayPoint]:
        """Build Nav2 waypoints for navigating to a row start position.

        When via_pose ([x, y, theta]) is provided, it is prepended as an
        intermediate waypoint so the robot exits the dock area cleanly.
        """
        target_wp = WayPoint()
        target_wp.x = target_x
        target_wp.y = target_y
        target_wp.theta = target_theta

        if via_pose is not None:
            via_wp = WayPoint()
            via_wp.x = via_pose[0]
            via_wp.y = via_pose[1]
            via_wp.theta = via_pose[2]
            return [via_wp, target_wp]

        return [target_wp]
