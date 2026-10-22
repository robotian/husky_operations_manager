"""NavigationActionClient: pose 0 takes the robot's heading, and a retry resumes after the waypoints Nav2 passed."""
import math
from unittest.mock import MagicMock

from builtin_interfaces.msg import Time
from nav2_msgs.action import NavigateThroughPoses
from status_interfaces.msg import RobotStatus, SubTask, Task, WayPoint

from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.enum import NavigationStatus

# Task 640's move (2026-10-21): the dock's heading on waypoint 0, the lane entry, plant 260
DOCK = WayPoint(node_id=884, x=-8.467, y=-2.012, theta=math.pi)
ENTRY = WayPoint(node_id=-1, x=-1.38, y=-8.33, theta=0.0)
PLANT = WayPoint(node_id=260, x=9.172, y=-8.45, theta=-0.003)


def client(x=0.633, y=-8.321, yaw=0.0):
    c = object.__new__(NavigationActionClient)  # no node: only the goal bookkeeping is under test
    c.logger, c.node, c.client = MagicMock(), MagicMock(), MagicMock()
    c.node.get_clock().now().to_msg.return_value = Time()
    c.robot_status = RobotStatus()
    c.robot_status.topo_map_position.x, c.robot_status.topo_map_position.y = x, y
    c.robot_status.topo_map_orientation.z, c.robot_status.topo_map_orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    c.navigation_status, c.goal_handle, c._send_future = NavigationStatus.IDLE, None, None
    c._resume_key, c._next_wp, c.last_waypoint_index = None, None, 0
    return c


def task(sub_task_id=1):
    return Task(task_id=640, sub_tasks=[SubTask(sub_task_id=sub_task_id, type=SubTask.MOVING,
                                                data=[DOCK, ENTRY, PLANT])])


def test_pose_0_has_the_robot_heading_not_the_start_nodes():
    pose0 = client(yaw=0.0)._generate_goal_poses([DOCK, ENTRY, PLANT])[0].pose
    assert abs(pose0.orientation.w - 1.0) < 1e-9 and (pose0.position.x, pose0.position.y) == (0.633, -8.321)


def test_retry_resumes_after_the_passed_lane_entry():
    c = client()
    assert c.send_goal(task())
    assert c.waypoints_list == [DOCK, ENTRY, PLANT]
    # Nav2 passed the lane entry, then the goal failed
    c._feedback_callback(MagicMock(feedback=NavigateThroughPoses.Feedback(number_of_poses_remaining=1)))
    c.reset()
    assert c.send_goal(task())
    assert c.waypoints_list == [DOCK, PLANT]


def test_a_new_sub_task_starts_from_its_first_waypoint():
    c = client()
    c.send_goal(task())
    c._feedback_callback(MagicMock(feedback=NavigateThroughPoses.Feedback(number_of_poses_remaining=1)))
    c.reset()
    c.send_goal(task(sub_task_id=2))
    assert c.waypoints_list == [DOCK, ENTRY, PLANT]
