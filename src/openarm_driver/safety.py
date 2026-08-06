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
    """Check that joint position changes don't exceed velocity limits."""

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

    def __init__(self, velocity_limits: ArrayLike):
        """Initialize with maximum command velocities in rad/s."""
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        if (
            self.velocity_limits.ndim != 1
            or not np.all(np.isfinite(self.velocity_limits))
            or np.any(self.velocity_limits <= 0.0)
        ):
            raise ValueError("Joint velocity limits must be finite and positive.")

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Clamp a command to the motion allowed since the last command."""
        dt_s = float(kwargs["dt_s"])
        if not np.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("Command period must be finite and non-negative.")

        positions = np.asarray(joint_positions, dtype=float)
        previous = np.asarray(kwargs["driver"].last_command, dtype=float)
        if not (positions.shape == previous.shape == self.velocity_limits.shape):
            raise ValueError(
                "Joint positions, previous command, and velocity limits "
                "must have the same shape."
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(previous)):
            raise ValueError("Joint positions must contain only finite values.")

        delta = positions - previous
        max_delta = self.velocity_limits * dt_s
        limited_positions = previous + np.clip(delta, -max_delta, max_delta)
        violations = np.abs(delta) > max_delta
        if np.any(violations):
            violated_joints = np.where(violations)[0].tolist()
            return CheckResult(
                is_safe=False,
                force_stop=False,
                fixed_joint_positions=limited_positions,
                message=f"Joint velocity limited at joints: {violated_joints}",
                check_type="joint_velocity",
                details={"violated_joints": violated_joints, "dt_s": dt_s},
            )

        return CheckResult(
            is_safe=True,
            message="All joint command velocities within limits.",
            check_type="joint_velocity",
        )
