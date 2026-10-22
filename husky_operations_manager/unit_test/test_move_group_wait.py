"""Arm goals wait for move_group's move_action server, for at most arm.move_group_wait."""
from unittest.mock import MagicMock

from rclpy.time import Time

from husky_operations_manager.husky_operations_manager import HuskyOperationsManager


def node(up=False):
    n = object.__new__(HuskyOperationsManager)  # no ROS node: only the gate is under test
    n.get_logger, n.get_clock = MagicMock(), MagicMock()
    n.manipulator_client = MagicMock()
    n.manipulator_client.move_group_ready.return_value = up
    n.arm_move_group_wait, n._move_group_wait_since, n._dwell_deadlines = 180.0, None, {}
    return n


def at(n, t):
    n.get_clock().now.return_value = Time(seconds=t)


def test_holds_while_move_group_is_down_then_sends_when_it_comes_up():
    n = node()
    at(n, 10)
    assert not n._arm_send_ready()
    at(n, 30)
    assert not n._arm_send_ready()
    n.manipulator_client.move_group_ready.return_value = True
    assert n._arm_send_ready() and n._move_group_wait_since is None


def test_gives_up_waiting_after_move_group_wait():
    n = node()
    at(n, 10)
    assert not n._move_group_ready()
    at(n, 189)
    assert not n._move_group_ready()
    at(n, 190)
    assert n._move_group_ready()  # the goal goes out and fails the usual way
    assert not n._move_group_ready()  # a new wait starts


def test_no_wait_when_move_group_is_up():
    n = node(up=True)
    at(n, 10)
    assert n._arm_send_ready()
