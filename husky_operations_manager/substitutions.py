"""Custom launch substitutions for husky_operations_manager."""

import tempfile

from launch.substitution import Substitution
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

import yaml


class NamespacedYaml(Substitution):
    """Rewrite a parameter YAML file's top-level key to include a ROS namespace.

    ROS2 matches YAML keys against the node's fully-qualified name, which
    includes the namespace (e.g. ``a300_00036/my_node``).  When a node runs
    under a namespace pushed with ``PushRosNamespace``, a YAML file whose
    top-level key is just ``my_node`` is silently skipped and all parameters
    fall back to declared defaults.

    This substitution reads the source YAML, prepends ``<namespace>/`` to
    every top-level key, writes the result to a temp file, and returns the
    temp-file path — which is then passed to ``parameters=[...]`` in the Node
    action exactly as a normal YAML file path.

    Usage::

        from husky_operations_manager.launch_utils import NamespacedYaml

        parameters=[
            NamespacedYaml(
                source_file=config_path,          # str or Substitution
                namespace=LaunchConfiguration('namespace'),
            )
        ]
    """

    def __init__(self, *, source_file, namespace):
        """Store source file path and namespace as substitution lists."""
        super().__init__()
        self._source_file = normalize_to_list_of_substitutions(source_file)
        self._namespace = normalize_to_list_of_substitutions(namespace)
        self._tmp_files: list[str] = []

    def describe(self):
        """Return a human-readable description of this substitution."""
        return f'NamespacedYaml({self._source_file})'

    def perform(self, context) -> str:
        """Rewrite YAML keys with the resolved namespace and return the temp file path."""
        source_file = perform_substitutions(context, self._source_file)
        namespace = perform_substitutions(context, self._namespace).strip('/')

        with open(source_file) as f:
            data = yaml.safe_load(f)

        rewritten = {}
        for node_name, value in data.items():
            key = f'{namespace}/{node_name}' if namespace else node_name
            rewritten[key] = value

        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, prefix='ros2_ns_params_'
        )
        yaml.dump(rewritten, tmp)
        tmp.close()
        self._tmp_files.append(tmp.name)
        return tmp.name
