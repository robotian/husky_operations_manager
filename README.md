# husky_operations_manager

ROS2 package that manages the full operational lifecycle of a Clearpath Husky A300 robot performing lavender harvesting operations. The node drives a 27-state state machine that coordinates navigation, docking, undocking, arm control, loading, unloading, and charging — dispatching work to five underlying action clients and recovering from failures automatically.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [State Machine](#state-machine)
- [Node: HuskyOperationsManager](#node-huskyoperationsmanager)
  - [Published Topics](#published-topics)
  - [Subscribed Topics](#subscribed-topics)
  - [Parameters](#parameters)
- [Action Clients](#action-clients)
- [Supporting Modules](#supporting-modules)
- [Package Structure](#package-structure)
- [Configuration Files](#configuration-files)
- [Launch Files](#launch-files)
- [Unit Test Nodes](#unit-test-nodes)
- [Building](#building)
- [Running](#running)
- [Known Limitations](#known-limitations)

---

## Overview

The `HuskyOperationsManager` node receives `Task` messages from an external `JobPublisher` and executes them as a sequence of subtasks. Three task types are supported:

| Task | Subtask sequence |
|------|-----------------|
| `HARVESTING_TASK` | MOVING → HARVESTING (arm) → LOADING |
| `CHARGING_TASK` | DOCKING → CHARGING → UNDOCKING |
| `UNLOADING_TASK` | DOCKING → UNLOADING → UNDOCKING |

On startup the node fetches live docking configuration from the `docking_server` ROS2 parameter server via `DockingParamFetcher`. Until that fetch completes (up to 30 s) all action clients are held un-initialised. Once ready, the node checks whether the robot is already at the dock and, if so, executes a startup undocking sequence before accepting tasks.

---

## Architecture

```
JobPublisher ──────────── /status/task ──────────────────────────────┐
                                                                      ▼
docking_server ─── DockingParamFetcher ──► HuskyOperationsManager (1Hz state machine)
                                                      │
                    ┌─────────────────────────────────┼──────────────────────────────────┐
                    ▼                    ▼             ▼              ▼                   ▼
         NavigationActionClient  DockingActionClient  UndockingActionClient  ManipulatorTaskActionClient  ReverseDriveClient
                    │                    │             │              │                   │
               Nav2 BT               opennav_docking  opennav_docking  ManipulatorTask    reverse_navigation_node
               navigate_to_pose       dock_robot      undock_robot     action server      (TF closed-loop fallback)
```

**Startup sequence:**

1. `DockingParamFetcher` polls `docking_server` every 0.5 s (30 s timeout)
2. On `DONE` → `_on_docking_config_ready()` instantiates all five action clients
3. `check_initial_position` fires after `initial_position_check_delay` seconds
4. If robot is within `docking.threshold` of the dock → execute startup undocking
5. Node enters normal task processing loop

---

## State Machine

The `RobotStatusEnum` drives all control flow. Transitions happen via `_transition_status()`.

```
IDLE
 └─► JOB_START
      ├─► START_MOVING ──► MOVING ──► DESTINATION_REACHED
      │
      ├─► START_HARVESTING ──► HARVESTING ──► DONE_HARVESTING
      │
      ├─► START_DOCKING ──► DOCKING ──► DONE_DOCKING
      │
      ├─► START_LOADING ──► LOADING ──► DONE_LOADING
      │
      ├─► START_UNDOCKING ──► UNDOCKING ──► DONE_UNDOCKING
      │        └─ (fallback) ──► ReverseDriveClient ──► DONE_UNDOCKING
      │
      ├─► START_UNLOADING ──► UNLOADING ──► DONE_UNLOADING
      │
      ├─► START_CHARGING ──► CHARGING ──► DONE_CHARGING
      │
      └─► JOB_DONE ──► IDLE (on next task arrival)

Special states: ERROR (94), PAUSED (95), MAINTENANCE (96),
                OFFLINE (97), EMERGENCY_STOP (98), ABNORMAL (99)
```

**Arm safety gates** prevent motion in unsafe configurations:

- `START_UNDOCKING` is held until `last_confirmed_arm_command == ArmCommand.GO_STOW`
- `DESTINATION_REACHED` (pre-harvest) is held until `last_confirmed_arm_command == ArmCommand.GO_READY`

---

## Node: HuskyOperationsManager

**Executable:** `husky_operations_manager`  
**Node name:** `husky_operations_manager`  
**Namespace:** configurable via launch argument (default `/a300_00036`)

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `{ns}/status/robot` | `status_interfaces/RobotStatus` | Full robot status at 1 Hz (state, battery, location, load) |

### Subscribed Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `{ns}/platform/bms/state` | `sensor_msgs/BatteryState` | BEST_EFFORT | Battery percentage; drives low/full threshold checks |
| `{ns}/ground_truth/pose` | `geometry_msgs/PoseWithCovarianceStamped` | RELIABLE depth=10 | Ground-truth pose; used for dock distance check |
| `{ns}/sensors/gps_0/imu` | `sensor_msgs/Imu` | BEST_EFFORT | IMU data (stored; not currently used in state logic) |
| `{ns}/platform/emergency_stop` | `std_msgs/Bool` | BEST_EFFORT | E-stop state → sets `EMERGENCY_STOP` online flag |
| `{ns}/status/task` | `status_interfaces/Task` | RELIABLE depth=10 | Task + subtask list from JobPublisher |

### Parameters

All parameters are declared with defaults and overridden by `config/config.yaml` at launch.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `navigation.max_retries` | `3` | Max navigation goal retry attempts |
| `navigation.retry_delay` | `5.0 s` | Delay between navigation retries |
| `docking.max_retries` | `2` | Max docking goal retry attempts |
| `docking.retry_delay` | `3.0 s` | Delay between docking retries |
| `docking.threshold` | `0.25 m` | Distance to dock to consider "already docked" |
| `battery.low_threshold` | `50.0 %` | Battery level triggering CHARGING_TASK |
| `battery.full_threshold` | `99.0 %` | Battery level that ends a CHARGING subtask |
| `loading.increment` | `20.0 %` | Load added per harvest cycle |
| `timing.timer_period` | `1.0 s` | Main control loop period |
| `timing.initial_position_check_delay` | `2.0 s` | Delay before dock proximity check at startup |
| `server.action_server_timeout` | `5.0 s` | Timeout waiting for action servers to become available |

---

## Action Clients

### NavigationActionClient (`action_clients/navigation.py`)

Wraps Nav2 `navigate_to_pose` action. Sends `WayPoint` goals and monitors `NavigationStatus`. On `ABORTED` or `FAILED`, increments `navigation_retry_count` up to `navigation_max_retries` before transitioning to `ERROR`.

### DockingActionClient (`action_clients/docking.py`)

Wraps `opennav_docking/dock_robot` action. Uses `active_dock` and `active_plugin` from `DockingConfig`. Retries up to `docking_max_retries` times. On persistent failure transitions to `ERROR`.

### UndockingActionClient (`action_clients/undocking.py`)

Wraps `opennav_docking/undock_robot` action. Sends an `UndockGoal` carrying the dock type. On failure the node falls back to `ReverseDriveClient` for closed-loop TF-based reverse drive to the staging pose.

### ManipulatorTaskActionClient (`action_clients/manipulator.py`)

Wraps `status_interfaces/ManipulatorTask` action server. Supported `ArmCommand` values:

| Command | Meaning |
|---------|---------|
| `GO_STOW` | Safe travel configuration — required before undocking |
| `GO_READY` | Pre-harvest reach configuration — required before harvesting |
| `GO_DROP` | Drop harvested material |
| `START_HARVEST` | Execute a single harvest stroke |
| `MOVE_EEF` | Move end-effector to a target `PoseStamped` |

Feedback `ManipulatorStatus` values map to `RobotStatusEnum` entries for live status publishing.

### ReverseDriveClient (`action_clients/reverse_drive_client.py`)

TF closed-loop fallback undocking. When `undock_robot` action fails, drives the robot in reverse toward the staging pose using continuous TF feedback. Configured via `config/config.yaml` (`reverse_navigation_node` section) and `config/reverse_navigation.yaml`. Parameters include `linear_speed: 0.15 m/s`, `staging_x_offset: -1.5 m`, and `timeout: 30 s`.

---

## Supporting Modules

### `docking_param_fetcher.py` — `DockingParamFetcher`

Asynchronously queries the `docking_server` ROS2 parameter server to build a complete `DockingConfig` object before any action client is created. The fetch pipeline progresses through states: `IDLE → LISTING → FETCHING → RESOLVING → COMPUTING → DONE`.

If the server is unavailable, it returns `ERROR` and the node retries every 0.5 s within a 30 s window, after which the node shuts down cleanly.

### `enum.py`

Defines all enumerations used across the package:

- `RobotStatusEnum` — 27 operational states plus ERROR/ABNORMAL variants
- `TaskEnum` / `SubTaskEnum` — task and subtask type codes
- `NavigationStatus` — IDLE through ERROR (9 values)
- `ManipulatorStatus` — IDLE/PLANNING/MOVING/MOVING_COMPLETE/FAILED
- `ReverseDriveStatus` — IDLE/REVERSING/DONE/CANCELED/ERROR
- `DockingParamFetcherStatus` — fetch pipeline stages
- `OnlineFlagEnum` — OFFLINE/ONLINE/EMERGENCY_STOP/ABNORMAL
- `DriveStatus` — drive client states

### `dataclass.py`

Pure-data containers shared across modules:

- `DockingConfig` — complete docking server configuration including controller tuning
- `DockInstanceConfig` — per-dock pose and metadata
- `DockPluginConfig` — per-plugin detection and control parameters
- `ManipulatorTaskFeedback` — arm task feedback snapshot
- `WPFStatus` — waypoint follower status
- `DriveConfig` — drive client configuration

---

## Package Structure

```
husky_operations_manager/
├── husky_operations_manager/
│   ├── husky_operations_manager.py      # Main node (production)
│   ├── docking_param_fetcher.py         # Async docking config fetcher
│   ├── enum.py                          # All enumerations
│   ├── dataclass.py                     # Shared data containers
│   ├── lavender_detection_test.py       # Lavender detection test node
│   └── action_clients/
│       ├── navigation.py                # Nav2 navigate_to_pose client
│       ├── docking.py                   # opennav dock_robot client
│       ├── undocking.py                 # opennav undock_robot client
│       ├── manipulator.py               # ManipulatorTask action client
│       ├── reverse_drive_client.py      # TF closed-loop fallback undock
│       └── drive_client.py              # Generic drive client
│   └── unit_test/
│       ├── test_husky_ops_navigation.py # Navigation-only integration harness
│       ├── test_navigation_client.py
│       ├── test_docking_client.py
│       ├── test_undocking_client.py
│       ├── test_harvest_client.py
│       ├── test_drive_client.py
│       ├── test_docking_param_fetcher.py
│       └── test_reverse_navigation.py
├── config/
│   ├── config.yaml                      # Main node parameters
│   ├── drive_client.yaml                # DriveClient parameters
│   └── reverse_navigation.yaml         # ReverseNavigationNode parameters
├── data/
│   └── best_model.pth                   # Lavender detection ML model (Git LFS)
├── launch/
│   ├── husky_operations_manager_launch.py  # Production launch
│   ├── test_husky_ops.launch.py            # Navigation test harness launch
│   ├── test_docking_param_fetcher.launch.py
│   ├── test_lavender_detection.launch.py
│   └── test_reverse_navigation.launch.py
├── package.xml
└── setup.py
```

---

## Configuration Files

### `config/config.yaml`

Main parameters for `HuskyOperationsManager` and `reverse_navigation_node`. The file uses wildcard namespace matching (`/*/husky_operations_manager`) so it applies regardless of robot namespace.

```yaml
/*/husky_operations_manager:
  ros__parameters:
    navigation:
      max_retries: 3
      retry_delay: 5.0
    docking:
      max_retries: 2
      retry_delay: 3.0
      threshold: 0.25
    battery:
      low_threshold: 50.0
      full_threshold: 99.0
    loading:
      increment: 20.0
    timing:
      timer_period: 1.0
      initial_position_check_delay: 2.0
    server:
      action_server_timeout: 5.0

/*/reverse_navigation_node:
  ros__parameters:
    goal_x: 0.5
    goal_y: 1.9
    goal_yaw: 1.571
    staging_x_offset: -1.5
    linear_speed: 0.15
    angular_speed: 0.30
    linear_tolerance: 0.05
    angular_tolerance: 0.1
    timeout: 30.0
    control_frequency: 20.0
```

### `config/drive_client.yaml`

Parameters for the `DriveClient` used in detection-aligned approach:

```yaml
ros__parameters:
  base_frame: "base_link"
  tf_base_frame: "arm_0_base_link"
  tf_detection_frame: "arm_0_detections"
  v_linear: 0.5
  v_angular: 0.5
  tf_polling_rate: 10.0
  timeout: 30.0
  tolerance: 0.05
```

---

## Launch Files

### Production launch

```bash
ros2 launch husky_operations_manager husky_operations_manager_launch.py \
  namespace:=/a300_00036
```

Launch arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `namespace` | `/a300_00036` | Robot namespace |
| `config_file` | `config/config.yaml` | Path to parameter file |

TF remappings (`/tf → tf`, `/tf_static → tf_static`) are applied automatically so the node operates correctly under a robot namespace.

### Test harness launch

```bash
# Navigation + docking/undocking pipeline (no arm)
ros2 launch husky_operations_manager test_husky_ops.launch.py

# Lavender detection test
ros2 launch husky_operations_manager test_lavender_detection.launch.py

# Reverse navigation only
ros2 launch husky_operations_manager test_reverse_navigation.launch.py
```

---

## Unit Test Nodes

Individual action-client test nodes can be run directly for integration verification:

| Executable | Tests |
|-----------|-------|
| `test_harvest_ops` | Full navigation + docking + charging pipeline (simulated 5 s harvest) |
| `test_navigation_client` | Nav2 goal send/cancel/retry |
| `test_docking_client` | `dock_robot` action |
| `test_undocking_client` | `undock_robot` action |
| `test_harvest_client` | `ManipulatorTask` action |
| `test_drive_client` | Drive client alignment |
| `docking_param_fetcher` | DockingParamFetcher standalone |
| `reverse_navigation_node` | TF closed-loop reverse drive |
| `test_lavender_detection` | Lavender detection model inference |

> **Note:** `test_harvest_ops` (`test_husky_ops_navigation.py`) uses a 5-second timer to simulate the arm harvest sequence. It does **not** send goals to the `ManipulatorTask` server and has no arm STOW/READY gates — it is a navigation-only integration harness.

---

## Building

```bash
cd ~/colcon_ws

# Source ROS2 underlay
source /opt/ros/humble/setup.bash

# Install dependencies
rosdep install --from-paths src --ignore-src -y

# Build
colcon build --packages-select husky_operations_manager --symlink-install

# Source overlay
source install/setup.bash
```

The package requires `status_interfaces` (custom messages/actions) to be built first. Build it as part of the workspace or ensure it is installed in the underlay.

---

## Running

```bash
# Source the workspace
source ~/colcon_ws/install/setup.bash

# Launch production node
ros2 launch husky_operations_manager husky_operations_manager_launch.py namespace:=/a300_00036

# Monitor robot state
ros2 topic echo /a300_00036/status/robot

# Check parameters at runtime
ros2 param list /a300_00036/husky_operations_manager
ros2 param get /a300_00036/husky_operations_manager navigation.max_retries
```

**Prerequisites:** The following must be running before the node starts:

- `docking_server` (opennav_docking) — required within 30 s of startup
- Nav2 stack with `navigate_to_pose` action server
- `ManipulatorTask` action server
- Battery, pose, IMU, and e-stop topics publishing

---

## Known Limitations

1. **Single dock support** — `active_dock` and `active_plugin` are always taken from index 0 of the `docking_server` lists. Multi-dock / multi-robot assignment requires per-task dock lookup (marked `TODO` in `_on_docking_config_ready`).

2. **Blocking `time.sleep()` calls** — Several retry paths (`_handle_navigation_retry`, `_subtask_docking`, `_handle_error_recovery`, `check_initial_position`) use `time.sleep()` which blocks the single-threaded executor. These should be replaced with non-blocking timer-based delays.

3. **Single-threaded executor** — The node uses `rclpy.spin()`. The retry delays block all subscription callbacks during sleep. Upgrading to `MultiThreadedExecutor` requires converting blocking sleeps to async timer patterns.

4. **Load status initialisation** — `current_load_status` is seeded from the first `Task` message's `crop_load` field. If the task topic is not publishing when the node starts, load begins at 0.
