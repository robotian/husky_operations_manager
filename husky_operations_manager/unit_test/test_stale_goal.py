"""NavigationActionClient: a goal the server accepts after it was cancelled must not take the client over."""
from unittest.mock import MagicMock

from husky_operations_manager.action_clients.navigation import NavigationActionClient
from husky_operations_manager.enum import NavigationStatus


def client_with_pending_goal():
    c = object.__new__(NavigationActionClient)  # no node: only the goal bookkeeping is under test
    c.logger = MagicMock()
    c.goal_handle = None
    c.navigation_status = NavigationStatus.SENDING
    c._send_future = MagicMock()
    return c


def accepted_future(c):
    f = c._send_future
    f.result.return_value = MagicMock(accepted=True)
    return f


def test_late_acceptance_after_cancel_is_cancelled_and_ignored():
    c = client_with_pending_goal()
    f = accepted_future(c)
    c.cancel_goal()
    assert c.navigation_status == NavigationStatus.CANCELED
    c._goal_response_callback(f)
    f.result().cancel_goal_async.assert_called_once()
    assert c.goal_handle is None and c.navigation_status == NavigationStatus.CANCELED


def test_late_acceptance_after_reset_does_not_override_idle():
    c = client_with_pending_goal()
    f = accepted_future(c)
    c.task = c.sub_task = None
    c.reset()
    c._goal_response_callback(f)
    f.result().cancel_goal_async.assert_called_once()
    assert c.navigation_status == NavigationStatus.IDLE


def test_current_acceptance_is_taken():
    c = client_with_pending_goal()
    f = accepted_future(c)
    c._goal_response_callback(f)
    assert c.goal_handle is f.result() and c.navigation_status == NavigationStatus.ACCEPTED
    f.result().cancel_goal_async.assert_not_called()


def test_unknown_result_status_is_aborted():
    c = client_with_pending_goal()
    c.navigation_status = NavigationStatus.ACTIVE
    c.waypoints_list = []
    c.last_waypoint_index = 0
    c._result_callback(MagicMock(result=MagicMock(return_value=MagicMock(status=0))))
    assert c.navigation_status == NavigationStatus.ABORTED
