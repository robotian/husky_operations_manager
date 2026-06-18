# Plan: New Combined-Navigation Operations Node (v2)

## Context

This node replaces the failed `test_lavender_harvest.py` with a clean implementation. It takes the startup-undocking foundation from `test_startup_init.py`, extends it with full task execution (tasks received from external JobPublisher), and adds dual-mode navigation: `NavigateThroughPoses` for navigating to row start or side change, and `DriveClient` for camera-guided row traversal.

**Do not reference `test_lavender_harvest.py` for implementation.** It is kept only as historical reference.

Sources to draw from:
- Startup undock: `test_startup_init.py`
- Task execution flow, action-client handlers: `test_husky_ops_navigation.py`
- YAML parameter loading pattern: `husky_operations_manager.py:80-178`
- YAML structure reference: `config/husky_operations_manager.yaml`

---

## New File

`src/husky_operations_manager/husky_operations_manager/unit_test/test_lavender_harvest.py`

**Override the existing file.** The old `test_lavender_harvest.py` is superseded.

**Class name**: `LavenderHarvestNode`

Register (or update) in `setup.py` console_scripts — entry already exists, no change needed:
```python
'test_lavender_harvest = husky_operations_manager.unit_test.test_lavender_harvest:main',
```

---

## Parameters: YAML (no hardcoded config, no module-level constants)

All configuration is loaded via `declare_parameter` / YAML. No hardcoded module-level `DOCK_CONFIGS`, `MOTION_CONFIG`, or `DRIVE_CONFIG` dictionaries.

**New YAML file**: `config/test/test_lavender_harvest.yaml`

Structure mirrors `husky_operations_manager.yaml` with additions for `drive`, `unloading`, and `task` sections:

```yaml
/*/test_lavender_harvest:
  ros__parameters:

    docks:
      names: [husky_charger, unloading_station]
      husky_charger:
        type:  simple_charging_dock
        frame: map
        pose:  [0.8, -1.945, 0.0]
      unloading_station:
        type:  simple_charging_dock
        frame: map
        pose:  [0.85, 1.60, 1.571]

    plugin:
      name:               simple_charging_dock
      staging_x_offset:  -1.5
      staging_yaw_offset: 0.0

    controller:
      base_frame:           base_link
      controller_frequency: 50.0
      v_linear_min:         0.15
      v_angular_max:        0.25

    undocking:
      linear_tolerance:  0.05
      angular_tolerance: 0.1
      dock_backwards:    false

    drive:
      detection_topic:       manipulators/arm_0_detection/image_annotated/detection_pose
      odom_topic:            platform/odom
      base_frame:            base_link
      cmd_vel_rate:          10.0
      v_linear_min:          0.05
      v_linear_max:          0.2
      v_angular_max:         0.5
      k_rho:                 1.0
      ex_tolerance:          0.05
      stop_lookahead:        0.05
      ex_coast_gate:         0.1
      ex_angular_gate:       0.05
      departure_clearance:   0.3
      no_detection_distance: 0.5

    navigation:
      max_retries: 3
      retry_delay: 5.0

    docking:
      max_retries: 2
      retry_delay: 3.0
      threshold:   0.25

    battery:
      low_threshold:  50.0
      full_threshold: 99.0

    loading:
      increment: 20.0

    unloading:
      home_delay_s: 2.0   # delay between AT_END result and sending HOME goal

    timing:
      timer_period:                 1.0
      initial_position_check_delay: 2.0

    task:
      trigger_topic: job/trigger    # topic (or service) name for server-side task generation

    # Subscription topics (relative to robot namespace)
    topics:
      battery:   platform/bms/state
      pose:      ground_truth/pose
      imu:       sensors/gps_0/imu
      estop:     platform/emergency_stop
      task:      status/task
      detection: manipulators/arm_0_detection/image_annotated/detection_pose
```

The node reads these in `__init__` with `declare_parameter`, then constructs `DockInstanceConfig`, `ReverseDriveConfig`, and `DriveConfig` objects from the loaded values. Pattern follows `husky_operations_manager.py:80-178` (now updated to also load `topics.*` params).

Note: `drive.detection_topic` and `topics.detection` refer to the same physical topic. `drive.detection_topic` is passed into `DriveConfig` for the `DriveClient`; `topics.detection` is the node-level subscription for `_detection_callback`. Both should be kept in sync in the YAML — set them to the same value.

---

## Task Generation (Deviation #1)

**Old**: Node passively subscribes to `/status/task` and processes whatever arrives.

**New**: Node actively triggers task generation from a server (JobPublisher). The server generates the task, stores it in a database, and delivers it back via `/status/task`.

Two interface options (YAML `task.trigger_topic` selects which):

### Option A — Topic Trigger
Node publishes a lightweight trigger message (e.g. `std_msgs/String` with robot namespace) to `{namespace}/{task.trigger_topic}` to signal readiness. Server receives, generates, persists to DB, and publishes back on `/status/task`.

### Option B — Service Trigger
Node calls a ROS2 service (e.g. `/job_server/request_task`) with the robot namespace. Service response carries the Task directly; server also stores it in DB.

**When triggers are sent by the node**:
- Startup undock complete → send trigger (request first task)
- Each `JOB_DONE` → send trigger (request next task)
- After CHARGING/UNLOADING task interruption resolves (back to IDLE) → send trigger

**Node does not interact with any database directly.** DB persistence is the server's responsibility.

---

## `__init__` Differences vs `test_startup_init.py`

| Item | `test_startup_init.py` | New node |
|------|----------------------|----------|
| Dock/motion params | hardcoded module constants | `declare_parameter` + YAML |
| `active_dock` set in | `_check_initial_position()` | same |
| `ReverseDriveClient` built in | `_check_initial_position()` | same |
| `DriveClient` | not present | built in `__init__()` from YAML drive params |
| `UnloaderActionClient` | not present | built in `__init__()` |
| Task subscription | not present | added (`/status/task`) |
| Task trigger publisher/client | not present | added (topic or service) |
| `RobotStatus` publisher | not present | added |
| `_init_state_variables()` | minimal inline | expanded (see below) |
| Subscription topics | hardcoded strings | loaded from `topics.*` YAML params |

`DriveClient` and `UnloaderActionClient` are instantiated in `__init__()` — they do not depend on `active_dock` so they can be built before dock selection.

`_declare_parameter()` includes a `topics.*` block (matching `husky_operations_manager.py` updated pattern). `_get_paramters()` reads them into `self.topic_battery`, `self.topic_pose`, `self.topic_imu`, `self.topic_estop`, `self.topic_task`, `self.topic_detection`. `_init_subscriptions()` uses `f'{self.namespace}/{self.topic_*}'` — no hardcoded topic strings anywhere.

---

## State Variables Beyond `test_startup_init.py`

Full set from `test_husky_ops_navigation.py` `_init_state_variables()`, plus:

```python
# Navigation routing
self._need_row_navigation: bool = True
# True  = next HARVESTING MOVING uses Nav2 first
# False = skip Nav2, start DriveClient directly

# DriveClient detection tracking
self._detection_received: bool  = False
# Reset False on each scan()/resume().
# Set True by _detection_callback on valid detection.
# Distinguishes STOPPED-at-bush from STOPPED-by-no-detection-timeout.

self.last_detection_time: float | None = None

# Unloader phase flag
self._unloader_at_end: bool = False
# False = awaiting AT_END; True = AT_END received, now commanding HOME.

# Arm state (mirrors husky_operations_manager.py)
self.last_confirmed_arm_command: str = ArmCommand.UNKNOWN
# Boot assumption: arm configuration unknown — STOW gate will always fire on first undock.

self.arm_stow_pending: bool  = False
# True while waiting for a STOW goal to complete via _handle_manipulator.
# Gates _subtask_undocking from proceeding until arm is safe.

self.arm_ready_pending: bool = False
# True while waiting for a READY goal to complete via _handle_manipulator.
# Gates _subtask_harvesting from advancing past DESTINATION_REACHED.
```

No `RowSide`, `_advance_row`, `return_row/side`, `_all_rows_complete`, `_harvest_complete` — row management is entirely the server's responsibility.

---

## Startup Sequence

Identical to `test_startup_init.py`:
- `_check_initial_position()` uses `min()` over all loaded dock configs by Euclidean distance — **not** `dock_names[0]`
- `ReverseDriveClient` built here with nearest dock at index 0
- If within `docking.threshold` → startup undocking fires; otherwise skip directly to ready
- When `startup_undock_complete = True` → send task trigger to server to request first task

---

## Task Execution

Unchanged from `test_husky_ops_navigation.py`:
- `_task_callback`
- `_handle_task_execution`
- `_handle_task_start` (see `_need_row_navigation` management below)
- `_execute_current_subtask`
- `_check_and_handle_low_battery`
- `_process_action_clients` — **extended**: polls `unloader_action_client.get_status()` and calls `_handle_manipulator()` when `arm_stow_pending`, `arm_ready_pending`, or manipulator status is `HARVESTING`
- `_update_current_subtask`

`_cancel_all_motion()`: cancels `navigation.cancel_goal()`, `drive_client.cancel()`, and `manipulator_client.cancel_goal()` (if active). Also resets `arm_stow_pending = False`, `arm_ready_pending = False`. Called from `_handle_error_recovery` and low-battery interrupt.

---

## `_subtask_moving` — Dual Navigation Logic

```
JOB_START:
    if task_type != HARVESTING_TASK OR _need_row_navigation:
        → START_MOVING  (Nav2 path)
    else:
        → DESTINATION_REACHED  (skip Nav2; DriveClient path)

START_MOVING:
    send_goal() → MOVING
    (Nav2 result transitions handled by _handle_navigation → DESTINATION_REACHED)

DESTINATION_REACHED:
    if task_type == HARVESTING_TASK:
        _need_row_navigation = False
        if drive_client.get_status() == STOPPED:
            drive_client.resume()   # depart from previous bush
        else:
            drive_client.scan()     # fresh row start
        _detection_received = False
        last_detection_time = now()
        → MOVING  (DriveClient active)

    else (CHARGING / UNLOADING):
        *** TASK INTERRUPTION ***
        - cancel all motion
        - send task trigger to server requesting the appropriate task type
          (CHARGING_TASK or UNLOADING_TASK)
        - clear current_task, reset subtask state
        - → IDLE
        Rationale: docking is NOT embedded as the next subtask index. The server
        generates a fresh CHARGING or UNLOADING task with its own full subtask list
        (MOVING → DOCKING → CHARGING/UNLOADING → UNDOCKING). The node resets
        cleanly to IDLE and waits for that task to arrive on /status/task.

MOVING (DriveClient active, _need_row_navigation=False):
    drive_status = drive_client.get_status()

    SCANNING / DEPARTING → no-op (drive continues)

    STOPPED:
        if _detection_received:
            current_sub_task_index += 1   # advance to HARVESTING subtask
            → DESTINATION_REACHED
        else:
            # No-detection timeout (row end)
            _need_row_navigation = True
            drive_client.reset()
            send task trigger to server (row complete / request next task)
            → JOB_DONE

    CANCELED / ERROR → ERROR
```

`_detection_callback`: on `detection_valid=True` → `_detection_received = True`, update `last_detection_time`. DriveClient stops internally; the node does NOT call `stop()` or `forward()`.

**API constraint**: use `drive.py` public API only — `scan()`, `resume()`, `cancel()`, `reset()`, `get_status()`, `is_active()`, `is_ready()`.

---

## `_handle_task_start` — `_need_row_navigation` Management

After JOB_DONE, before clearing `current_task`:

```python
if task_type == Task.HARVESTING_TASK:
    self._need_row_navigation = False   # resume row, DriveClient next
elif task_type in (Task.CHARGING_TASK, Task.UNLOADING_TASK):
    self._need_row_navigation = True    # must navigate back to row after
```

After transitioning to IDLE: send task trigger to server to request the next task.

---

## `_subtask_unloading` — `UnloaderActionClient` (Deviation #5)

Replaces the simulated timer with `UnloaderActionClient` from `action_clients/unloader.py`.

**API summary**:
- `send_goal(OperateUnloader.Goal.END)` — move carriage to end (dump) position
- `send_goal(OperateUnloader.Goal.HOME)` — move carriage back to home position
- `get_status()` → `RobotStatusEnum`: `IDLE`, `START_UNLOADING`, `UNLOADING`, `DONE_UNLOADING`, `ERROR`
- Both AT_END and AT_HOME results yield `DONE_UNLOADING` — node tracks phase via `_unloader_at_end`

**State machine**:

```
DONE_DOCKING:
    → START_UNLOADING

START_UNLOADING:
    send_goal(END)
    _unloader_at_end = False
    → UNLOADING

UNLOADING (polling unloader_action_client.get_status() each tick):
    if DONE_UNLOADING:
        if not _unloader_at_end:
            _unloader_at_end = True
            sleep(params.unloading.home_delay_s)   # configured pause
            send_goal(HOME)
            # status stays UNLOADING — client resets to START_UNLOADING → UNLOADING internally
        else:
            # HOME confirmed — carriage returned
            current_load_status = 0.0
            → DONE_UNLOADING
    if ERROR:
        → ERROR

DONE_UNLOADING:
    last_undocking_subtask = current_sub_task
    undocking_after_task_type = Task.UNLOADING_TASK
    → START_UNDOCKING → _subtask_undocking()
```

---

## `_subtask_charging`

Unchanged from `test_husky_ops_navigation.py`. Polls battery percentage until `battery.full_threshold` is reached, then triggers undocking.

---

## `_subtask_harvesting` — `ManipulatorTaskActionClient` with STOW/READY gating

Uses `ManipulatorTaskActionClient` from `action_clients/manipulator.py`. Gating logic is identical to `husky_operations_manager.py`.

**API summary**:
- `send_harvesting_goal(subtask)` — sends `START_HARVEST`
- `send_ready_goal(subtask)` — sends `GO_READY`
- `send_stow_goal(subtask)` — sends `GO_STOW`
- `cancel_goal()`, `reset()`, `get_status()` → `RobotStatusEnum`

Internal client status flow:
- `send_*_goal()` → `START_HARVESTING` → (accepted) → `HARVESTING`
- Result SUCCEEDED → `DONE_HARVESTING`; ABORTED → `ERROR`; CANCELED → `IDLE`

**State machine**:

```
DESTINATION_REACHED:
    READY gate:
      if last_confirmed_arm_command != GO_READY:
          if not arm_ready_pending:
              send_ready_goal(current_sub_task) → arm_ready_pending = True
          return  # hold here until _handle_manipulator clears arm_ready_pending
    # Arm confirmed READY → advance
    → START_HARVESTING

START_HARVESTING:
    guard: skip if manipulator already HARVESTING
    send_harvesting_goal(current_sub_task)
    → HARVESTING  (or ERROR)

HARVESTING:
    no-op — result handled by _handle_manipulator → DONE_HARVESTING

DONE_HARVESTING:
    STOW gate (guard with arm_stow_pending to avoid re-incrementing each tick):
      if not arm_stow_pending and last_confirmed != GO_STOW:
          current_load_status += loading_increment
          send_stow_goal(current_sub_task) → arm_stow_pending = True
          return  # hold until _handle_manipulator clears arm_stow_pending
    # Arm confirmed STOW → advance
    if last_confirmed == GO_STOW and not arm_stow_pending:
        → JOB_DONE
```

---

## `_subtask_undocking` — STOW gate (same as `husky_operations_manager.py`)

**START_UNDOCKING gate** — arm must be confirmed STOW before the undocking goal is sent:

```
START_UNDOCKING:
    if last_confirmed_arm_command != GO_STOW:
        if not arm_stow_pending:
            undock_ref = current_sub_task or last_undocking_subtask
            send_stow_goal(undock_ref) → arm_stow_pending = True
        return  # hold until _handle_manipulator confirms STOW and re-enters this method
    # Arm confirmed STOW — proceed with undocking
    send undocking goal → UNDOCKING

DONE_UNDOCKING:
    clear last_undocking_subtask, undocking_after_task_type
    → JOB_DONE
```

`_handle_manipulator` re-enters `_subtask_undocking()` directly after STOW is confirmed in `START_UNDOCKING` context (same as production node).

---

## `_handle_manipulator` — context-aware handler

Copy verbatim from `husky_operations_manager.py:1226-1337`. Three branches:

| Condition | Action |
|-----------|--------|
| `arm_stow_pending` and client `DONE_HARVESTING` | Clear `arm_stow_pending`, set `last_confirmed = GO_STOW`, reset client. If context is `START_UNDOCKING` → re-enter `_subtask_undocking()`. If `DONE_HARVESTING` → no-op (subtask picks up next tick). |
| `arm_ready_pending` and client `DONE_HARVESTING` | Clear `arm_ready_pending`, set `last_confirmed = GO_READY`, reset client. No-op otherwise (subtask picks up next tick at `DESTINATION_REACHED`). |
| Neither flag, client `DONE_HARVESTING`, node in `HARVESTING` | Reset client, transition node → `DONE_HARVESTING`. |
| Client `ERROR` | Clear both flags, reset client, → `ERROR`. |

---

## Action Client Handlers

Copy verbatim from `test_husky_ops_navigation.py` (**not** from `test_lavender_harvest.py`):
- `_handle_navigation`, `_handle_navigation_retry`, `_retry_navigation`
- `_handle_docking`, `_handle_docking_retry`, `_retry_docking`
- `_handle_undocking`, `_handle_undocking_retry`
- `_handle_reverse_drive`
- `_handle_error_recovery` — extended: calls `_cancel_all_motion()` before resetting state

`_process_action_clients` additions:
- Poll `unloader_action_client.get_status()` → call `_handle_unloader_client()`
- Call `_handle_manipulator()` when `arm_stow_pending`, `arm_ready_pending`, or `manipulator_client.get_status() == HARVESTING`

`_handle_manipulator` is copied verbatim from `husky_operations_manager.py:1226-1337` (see `_handle_manipulator` section above).

---

## Files to Reference During Implementation

| Purpose | File |
|---------|------|
| Startup undock + nearest-dock | `unit_test/test_startup_init.py` |
| Task execution + all action-client handlers | `unit_test/test_husky_ops_navigation.py` |
| YAML param loading pattern | `husky_operations_manager.py:80-178` |
| YAML structure reference | `config/husky_operations_manager.yaml` |
| DriveClient public API | `action_clients/drive.py` |
| UnloaderActionClient API | `action_clients/unloader.py` |
| ManipulatorTaskActionClient API | `action_clients/manipulator.py` |
| `DriveConfig` + other types | `types.py` |
| `DriveStatus`, `RobotStatusEnum` | `robot_enums.py` |

**Never reference `test_lavender_harvest.py` (old) for any logic.**

---

## New / Modified Files Summary

| File | Action |
|------|--------|
| `unit_test/test_lavender_harvest.py` | Override (full rewrite) |
| `config/test/test_lavender_harvest.yaml` | Create new YAML config |
| `setup.py` | No change (entry already present) |

---

## Verification

1. Build: `colcon build --packages-select husky_operations_manager --symlink-install`
2. Launch:
   ```
   ros2 run husky_operations_manager test_lavender_harvest \
       --ros-args -r __ns:=/a200_0284 \
       --params-file config/test/test_lavender_harvest.yaml
   ```
3. Startup: nearest-dock selection fires from YAML poses, undock completes, task trigger sent to server.
4. HARVESTING flow: Nav2 goal sent (first task, `_need_row_navigation=True`); DriveClient starts at `DESTINATION_REACHED`; detection stops drive and advances to HARVESTING subtask.
5. Consecutive HARVESTING: `_need_row_navigation=False` confirmed — Nav2 skipped, `drive_client.resume()` called.
6. Row end (no detection): `JOB_DONE`, `_need_row_navigation=True` reset, task trigger sent.
7. CHARGING/UNLOADING interruption: `DESTINATION_REACHED` in MOVING triggers task interruption → node resets to IDLE → server delivers CHARGING/UNLOADING task.
8. Unloading: `UnloaderActionClient` sends END goal, waits for `DONE_UNLOADING` (AT_END), delays `home_delay_s`, sends HOME goal, waits for `DONE_UNLOADING` (AT_HOME), then proceeds to undocking.
9. Arm gating — READY: at `DESTINATION_REACHED`, confirm `send_ready_goal()` fires when arm not already READY; node holds until `_handle_manipulator` clears `arm_ready_pending`; then advances to `START_HARVESTING`.
10. Arm gating — STOW after harvest: at `DONE_HARVESTING`, confirm load increments once, `send_stow_goal()` fires, node holds until `_handle_manipulator` clears `arm_stow_pending`; then advances to `JOB_DONE`.
11. Arm gating — STOW before undock: at `START_UNDOCKING` (from any undocking context), confirm `send_stow_goal()` fires when arm not already STOW; `_handle_manipulator` re-enters `_subtask_undocking()` once confirmed.
