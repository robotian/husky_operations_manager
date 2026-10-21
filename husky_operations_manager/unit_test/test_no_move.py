"""is_no_move: a dock task's move that only names the start node is skipped (status_server _build_dock_task)."""
from status_interfaces.msg import SubTask, Task, WayPoint

from husky_operations_manager.husky_operations_manager import is_no_move


def task(task_type, nodes, move_type=SubTask.MOVING):
    move = SubTask(sub_task_id=1, type=move_type, data=[WayPoint(node_id=n) for n in nodes])
    return Task(task_type=task_type, sub_tasks=[move, SubTask(sub_task_id=2, type=SubTask.DOCKING)])


def test_dock_task_from_here_is_no_move():
    assert is_no_move(task(Task.CHARGING_TASK, [80, 80]))
    assert is_no_move(task(Task.UNLOADING_TASK, [80, 80]))


def test_leaving_the_lane_moves():
    assert not is_no_move(task(Task.CHARGING_TASK, [80, 81, 82]))


def test_harvest_task_always_moves():
    assert not is_no_move(task(Task.HARVESTING_TASK, [80, 80]))


def test_no_task_or_no_move_subtask():
    assert not is_no_move(None)
    assert not is_no_move(Task(task_type=Task.CHARGING_TASK))
    assert not is_no_move(task(Task.CHARGING_TASK, [80, 80], move_type=SubTask.DOCKING))
