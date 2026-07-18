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

"""Safety checkers for OpenArm robot control."""

import numpy as np
from numpy.typing import ArrayLike

from .base_safety import Checker, CheckResult


DEFAULT_COMMAND_DT_S = 1.0 / 250.0
MAX_COMMAND_DT_S = 0.1


class JointPosChecker(Checker):
    """Check that joint positions are within limits."""

    def __init__(self, joint_limits: ArrayLike):
        """Initialize joint position checker.

        Args:
            joint_limits: (8, 2) array of [min, max] limits.

        """
        self.joint_limits = np.asarray(joint_limits, dtype=float)

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        positions = np.asarray(joint_positions, dtype=float)
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]

        position_limited = np.clip(positions, low, high)
        violations = (positions < low) | (positions > high)

        if np.any(violations):
            violated_joints = np.where(violations)[0].tolist()
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=position_limited,
                message=f"Joint positions over limits at joints: {violated_joints}",
                check_type="joint_limits",
                details={"violated_joints": violated_joints},
            )

        return CheckResult(
            is_safe=True,
            message="All joint positions within limits.",
            check_type="joint_limits",
        )


class JointDeltaPosChecker(Checker):
    """Check that a single joint command does not contain an abnormal jump."""

    def __init__(self, delta_limits: ArrayLike):
        """Initialize delta position checker.

        Args:
            delta_limits: Maximum allowed change per step for each joint.

        """
        self.delta_limits = np.asarray(delta_limits, dtype=float)

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        driver = kwargs.get("driver")
        if driver is None or not hasattr(driver, "last_command"):
            return CheckResult(
                is_safe=True,
                message="No previous command to compare against.",
                check_type="joint_delta",
            )

        positions = np.asarray(joint_positions, dtype=float)
        delta = positions - driver.last_command

        for i, d in enumerate(delta):
            if abs(d) > self.delta_limits[i]:
                return CheckResult(
                    is_safe=False,
                    force_stop=True,
                    message=(
                        f"Joint {i} delta {d:.4f} exceeds limit {self.delta_limits[i]:.4f}"
                    ),
                    check_type="joint_delta",
                    details={
                        "joint": i,
                        "delta": float(d),
                        "limit": float(self.delta_limits[i]),
                    },
                )

        return CheckResult(
            is_safe=True,
            message="All joint deltas within limits.",
            check_type="joint_delta",
        )


class JointVelocityChecker(Checker):
    """Limit per-joint command velocity using elapsed command time."""

    def __init__(
        self,
        velocity_limits: ArrayLike,
        default_dt_s: float = DEFAULT_COMMAND_DT_S,
        max_dt_s: float = MAX_COMMAND_DT_S,
    ):
        """Initialize joint velocity checker.

        Args:
            velocity_limits: Maximum command velocity in rad/s for each joint.
            default_dt_s: Fallback command period when timing is unavailable.
            max_dt_s: Maximum elapsed time credited to one command.

        """
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        self.default_dt_s = float(default_dt_s)
        self.max_dt_s = float(max_dt_s)
        if np.any(self.velocity_limits <= 0.0):
            raise ValueError("Joint velocity limits must be positive.")
        if self.default_dt_s <= 0.0:
            raise ValueError("Default command period must be positive.")
        if self.max_dt_s < self.default_dt_s:
            raise ValueError(
                "Maximum command period must be at least the default period."
            )

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Clamp a command to the motion allowed since the last command."""
        driver = kwargs.get("driver")
        if driver is None or not hasattr(driver, "last_command"):
            return CheckResult(
                is_safe=True,
                message="No previous command to compare against.",
                check_type="joint_velocity",
            )

        positions = np.asarray(joint_positions, dtype=float)
        previous = np.asarray(driver.last_command, dtype=float)
        if (
            positions.shape != previous.shape
            or positions.shape != self.velocity_limits.shape
        ):
            raise ValueError(
                "Joint positions, previous command, and velocity limits "
                "must have the same shape."
            )

        command_time_s = kwargs.get("command_time_s")
        previous_time_s = getattr(driver, "last_command_time_s", None)
        if command_time_s is None or previous_time_s is None:
            dt_s = self.default_dt_s
        else:
            elapsed_s = float(command_time_s) - float(previous_time_s)
            if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
                dt_s = self.default_dt_s
            else:
                dt_s = min(elapsed_s, self.max_dt_s)

        delta = positions - previous
        max_delta = self.velocity_limits * dt_s
        velocity_limited = previous + np.clip(delta, -max_delta, max_delta)
        violations = np.abs(delta) > max_delta
        if np.any(violations):
            violated_joints = np.where(violations)[0].tolist()
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=velocity_limited,
                message=f"Joint velocity limited at joints: {violated_joints}",
                check_type="joint_velocity",
                details={
                    "violated_joints": violated_joints,
                    "dt_s": dt_s,
                },
            )

        return CheckResult(
            is_safe=True,
            message="All joint command velocities within limits.",
            check_type="joint_velocity",
        )
