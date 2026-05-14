"""TaskGenerator — generates typed Task messages for the robot fleet."""

import os

from rclpy.impl.rcutils_logger import RcutilsLogger

from status_interfaces.msg import DockGoal, SubTask, Task, UndockGoal, WayPoint


logger = RcutilsLogger(os.path.basename(__file__))

# ---------------------------------------------------------------------------
# Hardcoded fallback dock parameters.
# Used only when the live dock config fetched from a robot's docking_server is
# empty or does not contain a matching dock type.  Shared station for charging
# and unloading is intentional during the current development phase.
# ---------------------------------------------------------------------------
_FALLBACK_DOCK_ID = 'husky_charger'
_FALLBACK_DOCK_TYPE = 'simple_charging_dock'
_FALLBACK_UNDOCK_TIMEOUT = 30.0  # seconds


# ---------------------------------------------------------------------------
# TaskGenerator
# ---------------------------------------------------------------------------

class TaskGenerator:
    """Generates Task messages (with sub-tasks and waypoints) for the robot fleet."""

    # =========================================================================
    # FLEET TASK GENERATORS — multi-subtask combined tasks for JobPublisher
    # =========================================================================

    def generate_harvest_task(self, job_id: int) -> Task | None:
        """Create a harvest task: HARVESTING (manipulator cut-and-load cycle)."""
        harvest_sub_task = SubTask()
        harvest_sub_task.sub_task_id = 2
        harvest_sub_task.type = SubTask.HARVESTING
        harvest_sub_task.description = 'Harvesting the bush'
        harvest_sub_task.data_str = 'harvest data'

        task = Task(
            task_id=job_id,
            task_type=Task.HARVESTING_TASK,
            assigned_robot_id=4,
            description=f'Harvesting Job {job_id}',
            crop_type='lavender',
            sub_tasks=[harvest_sub_task],
        )

        logger.debug(f'Harvest task generated: {task}')
        return task

    def generate_charging_task(self, waypoint_list: list[WayPoint]) -> Task | None:
        """
        Create a charging task: MOVING → DOCKING → CHARGING.

        Pass waypoints from build_route_to_dock() to route through farm nodes.
        """
        dock_goal = DockGoal(
            use_dock_id=True,
            dock_id=_FALLBACK_DOCK_ID,
            navigate_to_staging_pose=True,
        )
        undock_goal = UndockGoal(
            dock_type=_FALLBACK_DOCK_TYPE,
            max_undocking_time=_FALLBACK_UNDOCK_TIMEOUT,
        )

        logger.debug(
            f"Charging dock: id='{_FALLBACK_DOCK_ID}' type='{_FALLBACK_DOCK_TYPE}' "
            f'waypoints={[w.node_id for w in waypoint_list]}'
        )

        move_sub_task = SubTask()
        move_sub_task.sub_task_id = 1
        move_sub_task.type = SubTask.MOVING
        move_sub_task.description = 'Move to charging node'
        move_sub_task.data = waypoint_list

        docking_sub_task = SubTask()
        docking_sub_task.sub_task_id = 2
        docking_sub_task.type = SubTask.DOCKING
        docking_sub_task.description = 'Dock at charging station'
        docking_sub_task.dock_goal = dock_goal

        charging_sub_task = SubTask()
        charging_sub_task.sub_task_id = 3
        charging_sub_task.type = SubTask.CHARGING
        charging_sub_task.description = 'Charge at charging station'
        charging_sub_task.undock_goal = undock_goal

        task = Task(
            task_id=1,
            task_type=Task.CHARGING_TASK,
            assigned_robot_id=4,
            description='Charging Job',
            crop_type='',
            sub_tasks=[move_sub_task, docking_sub_task, charging_sub_task],
        )

        logger.debug(f'Charging task generated: {task}')
        return task

    def generate_unloading_task(self, waypoint_list: list[WayPoint]) -> Task | None:
        """
        Create an unloading task: MOVING → DOCKING → UNLOADING.

        Pass waypoints from build_route_to_dock() to route through farm nodes.
        """
        dock_goal = DockGoal(
            use_dock_id=True,
            dock_id=_FALLBACK_DOCK_ID,
            navigate_to_staging_pose=True,
        )
        undock_goal = UndockGoal(
            dock_type=_FALLBACK_DOCK_TYPE,
            max_undocking_time=_FALLBACK_UNDOCK_TIMEOUT,
        )

        logger.debug(
            f"Unloading dock: id='{_FALLBACK_DOCK_ID}' type='{_FALLBACK_DOCK_TYPE}' "
            f'waypoints={[w.node_id for w in waypoint_list]}'
        )

        move_sub_task = SubTask()
        move_sub_task.sub_task_id = 1
        move_sub_task.type = SubTask.MOVING
        move_sub_task.description = 'Move to unloading node'
        move_sub_task.data = waypoint_list

        docking_sub_task = SubTask()
        docking_sub_task.sub_task_id = 2
        docking_sub_task.type = SubTask.DOCKING
        docking_sub_task.description = 'Dock at unloading station'
        docking_sub_task.dock_goal = dock_goal

        unloading_sub_task = SubTask()
        unloading_sub_task.sub_task_id = 3
        unloading_sub_task.type = SubTask.UNLOADING
        unloading_sub_task.description = 'Unload at unloading station'
        unloading_sub_task.undock_goal = undock_goal

        task = Task(
            task_id=1,
            task_type=Task.UNLOADING_TASK,
            assigned_robot_id=4,
            description='Unloading Job',
            crop_type='',
            sub_tasks=[move_sub_task, docking_sub_task, unloading_sub_task],
        )

        logger.debug(f'Unloading task generated: {task}')
        return task

    # =========================================================================
    # ROUTING HELPER
    # =========================================================================

    def build_route_to_dock(self, robot_x: float, robot_y: float) -> list[WayPoint]:
        """
        Return a waypoint list routed through farm topology to the dock staging area.

        Pass the result directly to generate_charging_task() or generate_unloading_task().

        Requires farm_layout_path at construction time.
        """
        if self._farm_layout is None:
            logger.warning('build_route_to_dock called but FarmLayout not loaded')
            return []
        return self._farm_layout.build_route_to_dock(robot_x, robot_y)
