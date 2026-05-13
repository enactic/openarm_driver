# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config for OpenArm Driver."""

from pathlib import Path

import numpy as np
import yaml


class Config:
    """Configuration loader for OpenArm driver."""

    def __init__(self, config_path: str | None = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to config YAML file. If None, uses default config.yaml
                        in the package directory.

        """
        if config_path is None:
            # Use default config in the package directory
            package_dir = Path(__file__).parent
            config_path = package_dir / "openarm_cell.yaml"

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def get_joint_limits(self, arm_side: str) -> np.ndarray:
        """Get joint limits for specified arm."""
        limits = self._config["joint_limits"][arm_side]
        return np.array(limits)

    def get_joint_offsets(self, arm_side: str) -> np.ndarray:
        """Get joint offsets for specified arm."""
        offsets = self._config["joint_offsets"][arm_side]
        return np.array(offsets)

    def get_joint_delta_position_limits(self) -> dict:
        """Get joint delta position limits."""
        return self._config["joint_delta_position_limits"]

    def get_can_interface(self, arm_side: str) -> str:
        """Get can interface string."""
        return self._config["can_interface"][arm_side]

    def get_motor_types(self) -> list[str]:
        """Get motor type strings."""
        return self._config["motor_config"]["types"]

    def get_send_ids(self) -> list[int]:
        """Get motor send IDs."""
        return self._config["motor_config"]["send_ids"]

    def get_recv_ids(self) -> list[int]:
        """Get motor receive IDs."""
        return self._config["motor_config"]["recv_ids"]

    def get_gripper_posforce(self) -> bool:
        """Get gripper position-force control mode."""
        return self._config.get("gripper_posforce", True)

    def get_gripper_posforce_limits(self) -> np.ndarray:
        """Get gripper pos force mode limits."""
        return np.array(self._config.get("gripper_posforce_limits", [50.0, 1.0]))

    def get_default_kps(self) -> np.ndarray:
        """Get default proportional gains."""
        return np.array(self._config["control_gains"]["kps"])

    def get_default_kds(self) -> np.ndarray:
        """Get default derivative gains."""
        return np.array(self._config["control_gains"]["kds"])

    def get_start_config(self) -> dict:
        """Get start config."""
        return self._config["start"]

    def get_stop_config(self) -> dict:
        """Get stop config."""
        return self._config["stop"]


# Global default config instance
_default_config: Config | None = None


def get_default_config() -> Config:
    """Get the default global config instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_default_config(config: Config):
    """Set the default global config instance."""
    global _default_config
    _default_config = config
