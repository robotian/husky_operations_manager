import time
from typing import Callable

from rclpy.node import Node
from rclpy.impl.rcutils_logger import RcutilsLogger
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.parameter import parameter_value_to_python

from husky_operations_manager.dataclass import DockingConfig, DockPluginConfig, DockInstanceConfig
from husky_operations_manager.enum import DockingParamFetcherStatus


TOP_LEVEL_PARAM_MAP: dict[str, str] = {
    "base_frame":                   "base_frame",
    "fixed_frame":                  "fixed_frame",
    "controller_frequency":         "controller_frequency",
    "initial_perception_timeout":   "initial_perception_timeout",
    "wait_charge_timeout":          "wait_charge_timeout",
    "dock_approach_timeout":        "dock_approach_timeout",
    "undock_linear_tolerance":      "undock_linear_tolerance",
    "undock_angular_tolerance":     "undock_angular_tolerance",
    "max_retries":                  "max_retries",
    "dock_backwards":               "dock_backwards",
    "dock_prestaging_tolerance":    "dock_prestaging_tolerance",
}

CONTROLLER_PARAM_MAP: dict[str, str] = {
    "controller.k_phi":                                 "controller_k_phi",
    "controller.k_delta":                               "controller_k_delta",
    "controller.v_linear_min":                          "controller_v_linear_min",
    "controller.v_linear_max":                          "controller_v_linear_max",
    "controller.v_angular_max":                         "controller_v_angular_max",
    "controller.slowdown_radius":                       "controller_slowdown_radius",
    "controller.use_collision_detection":               "controller_use_collision_detection",
    "controller.costmap_topic":                         "controller_costmap_topic",
    "controller.footprint_topic":                       "controller_footprint_topic",
    "controller.transform_tolerance":                   "controller_transform_tolerance",
    "controller.projection_time":                       "controller_projection_time",
    "controller.simulation_time_step":                  "controller_simulation_time_step",
    "controller.dock_collision_threshold":              "controller_dock_collision_threshold",
    "controller.rotate_to_heading_angular_vel":         "controller_rotate_to_heading_angular_vel",
    "controller.rotate_to_heading_max_angular_accel":   "controller_rotate_to_heading_max_angular_accel",
}

PLUGIN_PARAM_MAP: dict[str, str] = {
    "plugin":                               "plugin",
    "docking_threshold":                    "docking_threshold",
    "staging_x_offset":                     "staging_x_offset",
    "staging_yaw_offset":                   "staging_yaw_offset",
    "use_external_detection_pose":          "use_external_detection_pose",
    "external_detection_timeout":           "external_detection_timeout",
    "external_detection_translation_x":     "external_detection_translation_x",
    "external_detection_translation_y":     "external_detection_translation_y",
    "external_detection_rotation_roll":     "external_detection_rotation_roll",
    "external_detection_rotation_pitch":    "external_detection_rotation_pitch",
    "external_detection_rotation_yaw":      "external_detection_rotation_yaw",
    "filter_coef":                          "filter_coef",
    "detector_service_name":                "detector_service_name",
    "detector_service_timeout":             "detector_service_timeout",
    "subscribe_toggle":                     "subscribe_toggle",
    "use_battery_status":                   "use_battery_status",
    "use_stall_detection":                  "use_stall_detection",
    "stall_velocity_threshold":             "stall_velocity_threshold",
    "stall_effort_threshold":               "stall_effort_threshold",
    "charging_threshold":                   "charging_threshold",
    "rotate_to_dock":                       "rotate_to_dock",
    "dock_direction":                       "dock_direction",
}

DOCK_INSTANCE_PARAM_MAP: dict[str, str] = {
    "type":  "type",
    "frame": "frame",
    "pose":  "pose",
    "id":    "id",
}


class DockingParamFetcher:
    """Fetches docking_server params and builds a DockingConfig via callback."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self.logger = RcutilsLogger(self.__class__.__name__)
        self.namespace = self.node.get_namespace().rstrip('/')

        self._status = DockingParamFetcherStatus.IDLE
        self._raw_params: dict = {}
        self._param_names: list[str] = []
        self._fetch_index: int = 0
        self._on_complete: Callable[[DockingConfig], None] | None = None
        self._config: DockingConfig | None = None
        self._dock_plugin_names: list[str] = []
        self._dock_instance_names: list[str] = []

        # Timing
        self._t_fetch_start: float = 0.0
        self._t_list_done: float = 0.0
        self._t_get_done: float = 0.0
        self._t_build_done: float = 0.0

        self._list_client = self.node.create_client(
            ListParameters,
            f'{self.namespace}/docking_server/list_parameters'
        )
        self._get_client = self.node.create_client(
            GetParameters,
            f'{self.namespace}/docking_server/get_parameters'
        )

        self.logger.info(f"DockingParamFetcher init | {self.namespace}/docking_server")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch(self, on_complete: Callable[[DockingConfig], None]) -> None:
        """Start async fetch. Delivers DockingConfig via on_complete callback."""
        self._t_fetch_start = time.perf_counter()

        if self._status not in (DockingParamFetcherStatus.IDLE, DockingParamFetcherStatus.ERROR):
            self.logger.warning(f"fetch() ignored — already {self._status.name}")
            return
        
        self._on_complete = on_complete
        self._raw_params.clear()
        self._param_names = []
        self._fetch_index = 0
        self._config = None
        self._dock_plugin_names = []
        self._dock_instance_names = []
        self._list_parameters()

    def fetch(self) -> None:
        """Start async fetch. Delivers DockingConfig via on_complete callback."""
        self._t_fetch_start = time.perf_counter()
        
        if self._status not in (DockingParamFetcherStatus.IDLE, DockingParamFetcherStatus.ERROR):
            self.logger.warning(f"fetch() ignored — already {self._status.name}")
            return

        self._raw_params.clear()
        self._param_names = []
        self._fetch_index = 0
        self._config = None
        self._dock_plugin_names = []
        self._dock_instance_names = []
        self._list_parameters()

    def get_config(self) -> DockingConfig | None:
        return self._config

    def get_status(self) -> DockingParamFetcherStatus:
        return self._status

    def reset(self) -> None:
        """Reset fetcher to IDLE so fetch() can be safely called again on retry."""
        self._status = DockingParamFetcherStatus.IDLE
        self._raw_params.clear()
        self._param_names = []
        self._fetch_index = 0
        self._config = None
        self._dock_plugin_names = []
        self._dock_instance_names = []
        self.logger.info("DockingParamFetcher reset to IDLE")

    # ------------------------------------------------------------------
    # Step 1 — Parameter List
    # ------------------------------------------------------------------

    def _list_parameters(self) -> None:
        if not self._list_client.wait_for_service(timeout_sec=3.0):
            self.logger.error("list_parameters service unavailable")
            self._status = DockingParamFetcherStatus.ERROR
            return

        self._status = DockingParamFetcherStatus.LISTING
        req = ListParameters.Request()
        req.depth = 0
        self._list_client.call_async(req).add_done_callback(self._list_cb)

    def _list_cb(self, future) -> None:
        try:
            response: ListParameters.Response = future.result()
        except Exception as e:
            self.logger.error(f"ListParameters failed: {e}")
            self._status = DockingParamFetcherStatus.ERROR
            return

        if response is None or len(response.result.names) == 0:
            self.logger.error("No parameters on docking_server")
            self._status = DockingParamFetcherStatus.ERROR
            return

        self._t_list_done = time.perf_counter()
        self._param_names = sorted(response.result.names)
        self.logger.info(f"Found {len(self._param_names)} params — fetching")
        self._status = DockingParamFetcherStatus.FETCHING
        self._fetch_index = 0
        self._fetch_next()

    # ------------------------------------------------------------------
    # Step 2 — Fetch one Paramater at a time
    # ------------------------------------------------------------------

    def _fetch_next(self) -> None:
        if self._fetch_index >= len(self._param_names):
            self._t_get_done = time.perf_counter()
            self.logger.info(
                f"FETCHING done in {self._t_get_done - self._t_list_done:.3f}s "
                f"— {len(self._raw_params)} params fetched "
                f"({len(self._param_names) - len(self._raw_params)} skipped NOT_SET)"
            )
            self._resolve_sections()
            return

        if not self._get_client.wait_for_service(timeout_sec=3.0):
            self.logger.error("get_parameters service unavailable")
            self._status = DockingParamFetcherStatus.ERROR
            return

        req = GetParameters.Request()
        req.names = [self._param_names[self._fetch_index]]
        self._get_client.call_async(req).add_done_callback(
            lambda f: self._get_cb(f, self._param_names[self._fetch_index])
        )

    def _get_cb(self, future, name: str) -> None:
        try:
            response: GetParameters.Response = future.result()
        except Exception as e:
            self.logger.error(f"[{name}] failed: {e}")
            self._advance()
            return

        if response is None or len(response.values) == 0:
            self._advance()
            return

        pv = response.values[0]
        if pv.type != 0:
            self._raw_params[name] = parameter_value_to_python(pv)

        self._advance()

    def _advance(self) -> None:
        self._fetch_index += 1
        self._fetch_next()

    # ------------------------------------------------------------------
    # Step 3 — Resolve dynamic section names
    # ------------------------------------------------------------------

    def _resolve_sections(self) -> None:
        self._status = DockingParamFetcherStatus.RESOLVING

        dock_plugins = self._raw_params.get("dock_plugins", [])
        docks        = self._raw_params.get("docks", [])

        if not dock_plugins or not docks:
            self.logger.error("dock_plugins or docks list is empty")
            self._status = DockingParamFetcherStatus.ERROR
            return

        self._dock_plugin_names   = list(dock_plugins)
        self._dock_instance_names = list(docks)

        self.logger.info(
            f"Plugins: {self._dock_plugin_names} | "
            f"Docks: {self._dock_instance_names}"
        )
        self._build_config()

    # ------------------------------------------------------------------
    # Step 4 — Build DockingConfig
    # ------------------------------------------------------------------

    def _build_config(self) -> None:
        self._status = DockingParamFetcherStatus.COMPUTING

        # Build plugin configs — missing fields set to None
        plugin_configs: dict[str, DockPluginConfig] = {}
        for name in self._dock_plugin_names:
            plugin_configs[name] = self._build_plugin_config(name)

        # Build dock instance configs — missing fields set to None
        dock_configs: dict[str, DockInstanceConfig] = {}
        for name in self._dock_instance_names:
            cfg = self._build_dock_instance_config(name)
            if cfg is None:
                self._status = DockingParamFetcherStatus.ERROR
                return
            dock_configs[name] = cfg

        # Assemble kwargs — missing params map to None via .get()
        kwargs: dict = {
            field: self._raw_params.get(raw)
            for raw, field in {**TOP_LEVEL_PARAM_MAP, **CONTROLLER_PARAM_MAP}.items()
        }

        kwargs.update({
            "dock_plugins":   self._raw_params.get("dock_plugins"),
            "docks":          self._raw_params.get("docks"),
            "plugin_configs": plugin_configs,
            "dock_configs":   dock_configs,
        })

        self._config = DockingConfig(**kwargs)
        self._status = DockingParamFetcherStatus.DONE
        self.logger.info("DockingConfig ready")

        if self._on_complete:
            self._on_complete(self._config)

        self._t_build_done = time.perf_counter()
        total = self._t_build_done - self._t_fetch_start
 
        self.logger.info(
            f"DockingConfig ready | "
            f"total={total:.3f}s  "
            f"list={self._t_list_done - self._t_fetch_start:.3f}s  "
            f"fetch={self._t_get_done - self._t_list_done:.3f}s  "
            f"build={self._t_build_done - self._t_get_done:.3f}s"
        )


    # ------------------------------------------------------------------
    # Sub-config builders
    # ------------------------------------------------------------------

    def _build_plugin_config(self, plugin_name: str) -> DockPluginConfig:
        prefix = f"{plugin_name}."
        raw = {k[len(prefix):]: v for k, v in self._raw_params.items() if k.startswith(prefix)}

        # Missing fields default to None via .get()
        kwargs = {"plugin_name": plugin_name}
        kwargs.update({field: raw.get(key) for key, field in PLUGIN_PARAM_MAP.items()})

        return DockPluginConfig(**kwargs)

    def _build_dock_instance_config(self, dock_name: str) -> DockInstanceConfig | None:
        prefix = f"{dock_name}."
        raw = {k[len(prefix):]: v for k, v in self._raw_params.items() if k.startswith(prefix)}

        # pose is the only field we must parse — fail if missing or malformed
        pose = raw.get("pose")
        if not pose or len(pose) < 3:
            self.logger.error(f"Dock '{dock_name}' pose missing or incomplete: {pose}")
            return None

        kwargs = {
            "instance_name": dock_name,
            "dock_x":        float(pose[0]),
            "dock_y":        float(pose[1]),
            "dock_theta":    float(pose[2]),
        }
        kwargs.update({field: raw.get(key) for key, field in DOCK_INSTANCE_PARAM_MAP.items()})

        return DockInstanceConfig(**kwargs)