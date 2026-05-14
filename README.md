# husky_operations_manager — image_detection_based_navigation

ROS2 package that manages the full operational lifecycle of a Clearpath Husky A300 robot performing **camera-guided lavender harvesting**. The robot navigates to the start of each crop row using Nav2, then drives forward using a `DriveClient` while a camera-based detector (`ImageDetectionPose`) identifies individual lavender bushes. Each detected bush triggers a simulated harvest cycle before driving continues to the next bush.

> **Branch:** `image_detection_based_navigation`
> This branch replaces fixed per-bush waypoint navigation with continuous forward driving + real-time image detection.

## Table of Contents

- [How It Differs from the Main Branch](#how-it-differs-from-the-main-branch)
- [Workflow Overview](#workflow-overview)
- [State Machine](#state-machine)
- [Node: HuskyOperationsManager (test_lavernder_ops.py)](#node-huskyoperationsmanager)
  - [Published Topics](#published-topics)
  - [Subscribed Topics](#subscribed-topics)
  - [Parameters](#parameters)
- [Action Clients](#action-clients)
- [DriveClient](#driveclient)
- [Supporting Modules](#supporting-modules)
- [Package Structure](#package-structure)
- [Launch Files](#launch-files)
- [Unit Test Nodes](#unit-test-nodes)
- [Building](#building)
- [Running](#running)
- [Known Limitations](#known-limitations)

---

## How It Differs from the Main Branch

| Aspect | `main` branch | `image_detection_based_navigation` |
|--------|--------------|-------------------------------------|
| Bush navigation | Fixed Nav2 waypoint per bush | Nav2 to row start → DriveClient forward scan |
| Bush detection | Counted from waypoints | `ImageDetectionPose` topic from camera |
| Arm management | `ManipulatorTaskActionClient` with STOW/READY gates | No arm client — harvest simulated with 5 s timer |
| Action clients | 5 (nav, dock, undock, manipulator, reverse drive) | 4 (nav, dock, undock, reverse drive) |
| New subscription | — | `{ns}/manipulators/arm_0_detection/image_annotated/detection_pose` |
| Row parameters | Task-level waypoints | `row_{i}_side_a_start/end`, `row_{i}_side_b_start/end` per row |

---

## Workflow Overview

```
Node starts
  └─► DockingParamFetcher (async, 0.5s poll, 30s timeout)
        └─► _on_docking_config_ready
              └─► Init NavigationActionClient, DockingActionClient,
                  UndockingActionClient, ReverseDriveClient
              └─► Start 1Hz timer + init-check timer

check_initial_position
  ├─► Robot NOT at dock → startup_undock_complete = True → Ready for tasks
  └─► Robot AT dock → startup undocking sequence → IDLE → Ready for tasks

─── Main 1Hz loop (timer_callback) ───────────────────────────────────────
  ERROR/ABNORMAL ──► _handle_error_recovery (cancel nav → IDLE)
  Startup not done ──► _handle_startup_undocking
  Normal ──► _handle_task_execution
    ├─► Low battery? → generate CHARGING_TASK
    ├─► Load = 100%? → generate UNLOADING_TASK
    └─► _handle_task_start → dispatch to subtask handler

─── MOVING subtask ────────────────────────────────────────────────────────
  JOB_START → START_MOVING
  START_MOVING → Nav2 send_goal (Starting Pose / row side start) → MOVING
  MOVING → (Nav2 active) → SUCCEEDED / ABORTED / CANCELED
  SUCCEEDED (within 0.25m) → DESTINATION_REACHED (Starting Pose)
                                └─► Start DriveClient (forward scan)
                                      └─► Lavender detected?
                                            YES → Stop Drive Client
                                                → Generate Harvest Task (JOB_DONE)
                                            NO  → continue driving
                                                  (no_detection_timeout → next nav goal)

─── HARVESTING subtask ────────────────────────────────────────────────────
  DESTINATION_REACHED → START_HARVESTING → HARVESTING (5s timer)
  → DONE_HARVESTING → increment load → JOB_DONE

─── DOCKING subtask ───────────────────────────────────────────────────────
  DESTINATION_REACHED → START_DOCKING (1s pause) → dock_robot → DOCKING
  → DONE_DOCKING  (retry up to docking.max_retries on ERROR)

─── CHARGING subtask ──────────────────────────────────────────────────────
  DONE_DOCKING → START_CHARGING → CHARGING (poll battery ≥ full_threshold)
  → DONE_CHARGING → store last_undocking_subtask → START_UNDOCKING

─── UNLOADING subtask ─────────────────────────────────────────────────────
  DONE_DOCKING → START_UNLOADING → UNLOADING (4s timer) → DONE_UNLOADING
  → reset load to 0 → store last_undocking_subtask → START_UNDOCKING

─── UNDOCKING subtask ─────────────────────────────────────────────────────
  START_UNDOCKING → send undock_robot goal → UNDOCKING
  → DONE_UNDOCKING → JOB_DONE
  ERROR → ReverseDriveClient fallback (closed-loop reverse to staging pose)
```

---

## State Machine

The `RobotStatusEnum` drives all control flow. Transitions are logged at INFO level via `_transition_status()`.

```
IDLE
 └─► JOB_START
      ├─► START_MOVING ──► MOVING ──► DESTINATION_REACHED
      │                                    └─► (DriveClient active)
      ├─► START_HARVESTING ──► HARVESTING ──► DONE_HARVESTING
      ├─► START_DOCKING ──► DOCKING ──► DONE_DOCKING
      ├─► START_LOADING ──► LOADING ──► DONE_LOADING
      ├─► START_UNDOCKING ──► UNDOCKING ──► DONE_UNDOCKING
      │         └─ (fallback) ReverseDriveClient ──► DONE_UNDOCKING
      ├─► START_UNLOADING ──► UNLOADING ──► DONE_UNLOADING
      ├─► START_CHARGING ──► CHARGING ──► DONE_CHARGING
      └─► JOB_DONE ──► IDLE (on next task)

Special: ERROR(94), PAUSED(95), MAINTENANCE(96), OFFLINE(97),
         EMERGENCY_STOP(98), ABNORMAL(99)
```

---

## Node: HuskyOperationsManager

**Source file:** `husky_operations_manager/unit_test/test_lavernder_ops.py`
**Executable:** `test_lavender_ops` *(see setup.py)*
**Node name:** `husky_operations_manager`
**Namespace:** configurable via launch argument (default `/a300_00036`)

> **Note:** This node lives in `unit_test/` because it is the integration test harness for the image-detection navigation workflow. It does **not** use a `ManipulatorTaskActionClient` — the arm harvest is simulated with a 5 s timer.

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `{ns}/status/robot` | `status_interfaces/RobotStatus` | Full robot status at 1 Hz |

### Subscribed Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `{ns}/platform/bms/state` | `sensor_msgs/BatteryState` | BEST_EFFORT | Battery level; drives low-battery checks |
| `{ns}/ground_truth/pose` | `geometry_msgs/PoseWithCovarianceStamped` | RELIABLE depth=10 | Pose used for dock proximity and nav-abort recovery |
| `{ns}/sensors/gps_0/imu` | `sensor_msgs/Imu` | BEST_EFFORT | IMU (stored; not used in state logic) |
| `{ns}/platform/emergency_stop` | `std_msgs/Bool` | BEST_EFFORT | E-stop → sets `EMERGENCY_STOP` online flag |
| `{ns}/status/task` | `status_interfaces/Task` | RELIABLE depth=10 | Task + subtask list from JobPublisher |
| `{ns}/manipulators/arm_0_detection/image_annotated/detection_pose` | `status_interfaces/ImageDetectionPose` | RELIABLE depth=10 | **Lavender bush detection** — triggers DriveClient stop and harvest task generation |

### Parameters

All declared with defaults; override via YAML at launch.

**General**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rows` | `1` | Number of crop rows to traverse |
| `no_detection_timeout` | `10.0 s` | Seconds without a detection before treating the row end as reached |
| `harvest_duration` | `5.0 s` | Simulated harvest time per bush |
| `unload_duration` | `5.0 s` | Simulated unloading duration at dock |
| `loading.increment` | `20.0 %` | Load added per harvest cycle |

**Navigation**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `navigation.max_retries` | `3` | Max Nav2 goal retry attempts |
| `navigation.retry_delay` | `5.0 s` | Delay between retries |

**Docking**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `docking.max_retries` | `2` | Max dock_robot retry attempts |
| `docking.retry_delay` | `3.0 s` | Delay between docking retries |
| `docking.threshold` | `0.25 m` | Dock proximity threshold for startup check |

**Battery**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `battery.low_threshold` | `50.0 %` | Triggers CHARGING_TASK generation |
| `battery.full_threshold` | `99.0 %` | Ends CHARGING subtask |

**Timing**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timing.timer_period` | `1.0 s` | Main control loop period |
| `timing.initial_position_check_delay` | `2.0 s` | Delay before startup dock proximity check |

**DriveClient**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drive.v_linear` | `0.2 m/s` | Forward drive speed during row scan |
| `drive.v_angular` | `0.5 rad/s` | Max heading correction during scan |
| `drive.tf_base_frame` | `arm_0_base_link` | Base frame for alignment check |
| `drive.tf_detection_frame` | `arm_0_detections` | Detection child frame (lavender bush centroid) |
| `drive.tolerance` | `0.05 m` | Lateral alignment tolerance |
| `drive.timeout` | `30.0 s` | DriveClient hard timeout |
| `drive.tf_polling_rate` | `10.0 Hz` | TF lookup rate |

**Row Waypoints** (declared for each `i` in `range(num_rows)`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `row_{i}_side_a_start` | `[0.0, 0.0, 0.0]` | `[x, y, yaw]` — Nav2 goal at start of side A |
| `row_{i}_side_a_end` | `[0.0, 5.0, 0.0]` | End of side A (transition point) |
| `row_{i}_side_b_start` | `[0.5, 5.0, 3.14]` | Start of side B (other side of row) |
| `row_{i}_side_b_end` | `[0.5, 0.0, 3.14]` | End of side B → move to next row |

---

## Action Clients

### NavigationActionClient (`action_clients/navigation.py`)

Wraps Nav2 `navigate_to_pose`. Used for:
- Navigating to the **Starting Pose** (row side start) before DriveClient scan
- Retrying on ABORTED/ERROR (up to `navigation.max_retries`)
- Detecting physical arrival within 0.25 m as a nav-abort recovery path

### DockingActionClient (`action_clients/docking.py`)

Wraps `opennav_docking/dock_robot`. Retries up to `docking.max_retries` times with a 1 s pre-pause to allow the robot to fully stop before the goal is sent.

### UndockingActionClient (`action_clients/undocking.py`)

Wraps `opennav_docking/undock_robot`. On ERROR falls back to `ReverseDriveClient` TF closed-loop reverse drive.

### ReverseDriveClient (`action_clients/reverse_drive_client.py`)

Fallback undocking path. Drives the robot in reverse toward the staging pose using continuous TF feedback when the `undock_robot` action fails. Configured via `config/config.yaml` (`reverse_navigation_node` section).

---

## DriveClient

**Source:** `action_clients/drive_client_new.py`

The `DriveClient` is the core of the image-detection navigation approach. It continuously publishes forward velocity to `{ns}/cmd_vel` while polling TF for the `arm_0_detections` frame relative to `arm_0_base_link`. When the detection frame falls within `drive.tolerance` of lateral centre, the robot is considered aligned.

```
DriveClient states (DriveStatus):
  IDLE → FORWARD → DONE | CANCELED | ERROR
```

**Interaction with detection:**
The node subscribes to `ImageDetectionPose` independently. When a valid detection arrives (`_detection_callback`), the DriveClient is stopped, `current_status` is not directly changed by the callback — instead a Harvest Task is generated externally by the JobPublisher pipeline, which triggers `_subtask_harvesting` on the next timer tick.

**Alignment detection flow:**
1. `DriveClient.forward()` called at `DESTINATION_REACHED`
2. TF is polled at `drive.tf_polling_rate` Hz
3. Lateral offset from `arm_0_base_link` to `arm_0_detections` measured
4. If offset ≤ `drive.tolerance` → `DriveStatus.DONE` (aligned with bush)
5. If timeout exceeded → `DriveStatus.ERROR`

---

## Supporting Modules

### `docking_param_fetcher.py` — `DockingParamFetcher`

Async pipeline that queries the `docking_server` ROS2 parameter server: `IDLE → LISTING → FETCHING → RESOLVING → COMPUTING → DONE`. On ERROR the node retries every 0.5 s within a 30 s window, then shuts down cleanly.

### `enum.py`

All enumerations used across the package:
- `RobotStatusEnum` — 27 operational states
- `TaskEnum` / `SubTaskEnum` — task and subtask type codes
- `NavigationStatus` — IDLE through ERROR
- `DriveStatus` — IDLE / FORWARD / REVERSE / DONE / CANCELED / ERROR
- `ReverseDriveStatus` — IDLE / REVERSING / DONE / CANCELED / ERROR
- `DockingParamFetcherStatus` — fetch pipeline stages
- `OnlineFlagEnum` — OFFLINE / ONLINE / EMERGENCY_STOP / ABNORMAL

### `dataclass.py`

Shared data containers including `DriveConfig` (holds all `drive.*` parameters), `DockingConfig`, `DockInstanceConfig`, `DockPluginConfig`.

---

## Package Structure

```
husky_operations_manager/
├── husky_operations_manager/
│   ├── husky_operations_manager.py          # Production node (main branch — arm management)
│   ├── docking_param_fetcher.py
│   ├── enum.py
│   ├── dataclass.py
│   ├── lavender_detection_test.py
│   └── action_clients/
│       ├── navigation.py
│       ├── docking.py
│       ├── undocking.py
│       ├── reverse_drive_client.py
│       ├── drive_client.py                  # Original DriveClient
│       └── drive_client_new.py              # ★ TF alignment DriveClient (this branch)
│   └── unit_test/
│       ├── test_lavernder_ops.py            # ★ Main image-detection node (this branch)
│       ├── test_lavender_harvest.py         # ★ Standalone row-harvesting integration test
│       ├── test_drive_client_new.py         # ★ DriveClient_new unit test
│       ├── test_husky_ops_navigation.py     # Navigation-only harness (main branch)
│       ├── test_navigation_client.py
│       ├── test_docking_client.py
│       ├── test_undocking_client.py
│       ├── test_harvest_client.py
│       ├── test_drive_client.py
│       ├── test_docking_param_fetcher.py
│       └── test_reverse_navigation.py
├── config/
│   ├── config.yaml
│   ├── drive_client.yaml
│   └── reverse_navigation.yaml
├── data/
│   └── best_model.pth                       # Lavender detection ML model (Git LFS)
├── launch/
│   ├── husky_operations_manager_launch.py
│   ├── test_husky_ops.launch.py
│   ├── test_lavender_detection.launch.py
│   ├── test_lavender_harvest.launch.py      # ★ Row-harvesting integration test launch
│   ├── test_docking_param_fetcher.launch.py
│   └── test_reverse_navigation.launch.py
├── package.xml
└── setup.py
```

★ = new or changed in this branch

---

## Launch Files

### Row-harvest integration test

```bash
ros2 launch husky_operations_manager test_lavender_harvest.launch.py \
  namespace:=/a300_00036
```

Loads `config/test_lavender_harvest.yaml` which must define `num_rows`, all `row_*` waypoints, and `drive.*` parameters.

### Production node (navigation + docking, no arm)

```bash
ros2 launch husky_operations_manager test_husky_ops.launch.py \
  namespace:=/a300_00036
```

### Detection-only test

```bash
ros2 launch husky_operations_manager test_lavender_detection.launch.py
```

### Reverse navigation standalone

```bash
ros2 launch husky_operations_manager test_reverse_navigation.launch.py
```

---

## Unit Test Nodes

| Executable | Source | Purpose |
|-----------|--------|---------|
| `test_lavender_harvest` | `test_lavender_harvest.py` | ★ Full row-traversal test — Nav2 + DriveClient + detection per side |
| `test_drive_client_new` | `test_drive_client_new.py` | ★ DriveClient_new alignment logic |
| `test_harvest_ops` | `test_husky_ops_navigation.py` | Nav + dock + charge pipeline (no arm, 5 s timer) |
| `test_navigation_client` | `test_navigation_client.py` | Nav2 goal send/cancel/retry |
| `test_docking_client` | `test_docking_client.py` | `dock_robot` action |
| `test_undocking_client` | `test_undocking_client.py` | `undock_robot` action |
| `test_harvest_client` | `test_harvest_client.py` | `ManipulatorTask` action (arm) |
| `test_drive_client` | `test_drive_client.py` | Original DriveClient |
| `docking_param_fetcher` | `test_docking_param_fetcher.py` | `DockingParamFetcher` standalone |
| `reverse_navigation_node` | `test_reverse_navigation.py` | TF closed-loop reverse drive |
| `test_lavender_detection` | `lavender_detection_test.py` | ML model inference check |

### Standalone row-traversal test (`test_lavender_harvest.py`)

This node uses an internal `HarvestPhase` enum and manages its own row iteration without a JobPublisher:

```
INIT → NAVIGATE_TO_ROW_START → DRIVE_ROW → NAVIGATE_TO_ROW_END
     → NAVIGATE_TO_NEXT_SIDE → DRIVE_ROW (other side)
     → (load ≥ 100%) → NAVIGATE_TO_UNLOAD_DOCK → DOCKING_FOR_UNLOAD
                      → UNLOADING → UNDOCKING_AFTER_UNLOAD
                      → NAVIGATE_RETURN → next row
     → COMPLETE
```

Safety: e-stop monitoring, low-battery abort, nav retry, DriveClient alignment timeout (logs warning and proceeds).

---

## Building

```bash
cd ~/colcon_ws
source /opt/ros/humble/setup.bash

# Build only this package
colcon build --packages-select husky_operations_manager --symlink-install

source install/setup.bash
```

`status_interfaces` must be built first (provides `ImageDetectionPose`, `Task`, `SubTask`, `RobotStatus`, etc.).

---

## Running

```bash
source ~/colcon_ws/install/setup.bash

# Image-detection row harvesting
ros2 launch husky_operations_manager test_lavender_harvest.launch.py \
  namespace:=/a300_00036

# Monitor state
ros2 topic echo /a300_00036/status/robot

# Check parameters
ros2 param list /a300_00036/test_lavender_harvest
ros2 param get /a300_00036/test_lavender_harvest num_rows
```

**Prerequisites before starting:**

- `docking_server` (opennav_docking) available within 30 s
- Nav2 stack with `navigate_to_pose` action server
- Camera pipeline publishing `ImageDetectionPose` on `{ns}/manipulators/arm_0_detection/image_annotated/detection_pose`
- Battery, pose, and e-stop topics publishing
- TF frames `arm_0_base_link` and `arm_0_detections` being broadcast

---

## Known Limitations

1. **Simulated harvest** — `_subtask_harvesting` uses a hardcoded 5 s timer (`harvest_duration` parameter). Physical arm integration requires replacing this with a `ManipulatorTaskActionClient` goal.

2. **Detection callback decoupled from state machine** — `_detection_callback` stores `ImageDetectionPose` but does not directly stop the DriveClient or generate the harvest task within the node. The JobPublisher is expected to issue a HARVESTING subtask in response to a detection event. If tight feedback-loop control is needed, the callback needs to directly call `drive_client.cancel()`.

3. **IMU topic typo** — `_init_subscriptions` subscribes to `{ns}/ssensors/gps_0/imu` (double `s`). Should be `{ns}/sensors/gps_0/imu`. The IMU is not used in state logic so this has no runtime effect.

4. **Blocking `time.sleep()` calls** — `check_initial_position`, `_subtask_docking`, `_handle_error_recovery`, and the navigation retry paths use `time.sleep()`, which blocks the single-threaded executor and delays all subscription callbacks during sleep.

5. **Single-threaded executor** — `rclpy.spin()` is used. Combined with blocking sleeps, callbacks for detection and battery can be delayed during retry waits.

6. **Single dock** — `active_dock` and `active_plugin` are always taken from index 0. Multi-dock support requires per-task dock lookup.

7. **`test_lavernder_ops.py` filename typo** — the primary node file has a typo in its name (`lavernder` instead of `lavender`). This is a cosmetic issue only — the entry point in `setup.py` is consistent with the misspelled filename.
