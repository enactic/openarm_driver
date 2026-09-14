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

"""Driver for OpenArm."""

import logging
import time
from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike
import openarm_can as oa

from .config import Config, get_default_config
from .base_safety import Checker, CompositeChecker
from .safety import (
    JointPosChecker,
    JointDeltaPosChecker,
    JointVelocityChecker,
)


logger = logging.getLogger(__name__)

MAX_COMMAND_DT_S = 0.1

# How long an axis may go without a reply before it is reported as silent.
# Deliberately generous: this only decides when a line is logged, never how the
# arm is driven, so it should not fire on an ordinary single missed frame.
AXIS_STALE_TIMEOUT_S = 0.2

# Shortest gap between two health checks. set_latest_state runs every
# control cycle, and reading the diagnostics costs about as much as the rest
# of that function; at this interval the cost is negligible and the delay is
# still far below AXIS_STALE_TIMEOUT_S, which is what decides when anything
# is reported.
HEALTH_CHECK_INTERVAL_S = 0.02

# Shortest gap between two reports of the same bus counter. Counters that
# advance once per failed frame would otherwise produce a line per frame for
# as long as the fault lasts.
FAULT_LOG_COOLDOWN_S = 5.0

# openarm_can.BusStatus fields that are ErrorCounter (count + seconds_ago()).
# Listed explicitly so a name that disappears upstream shows up as "stops being
# logged" during review rather than as a surprise at runtime.
_BUS_COUNTER_NAMES = (
    "bus_off",
    "error_passive",
    "error_warning",
    "tx_overflow",
    "rx_overflow",
    "ack_error",
    "tx_timeout",
    "restarted",
    "write_net_down",
    "write_no_buffer",
    "write_other",
)


class _IncrementReporter:
    """Reports growth of monotonically increasing counters, keyed by name.

    A counter is reported the first time it moves, and at most once per
    `cooldown_s` after that. Write failures advance once per failed frame, so
    reporting every increment means a line per frame for as long as the fault
    lasts. Suppressed increments are accumulated into the next report rather
    than dropped, so the totals stay honest.
    """

    def __init__(self, cooldown_s: float):
        self._cooldown_s = cooldown_s
        self._reported: dict = {}
        self._reported_at: dict = {}

    def take(self, key, count: int, now: float) -> int | None:
        """Growth since the last report, or None if there is nothing to say."""
        reported = self._reported.get(key, 0)
        if count <= reported:
            return None
        at = self._reported_at.get(key)
        if at is not None and now - at < self._cooldown_s:
            return None
        self._reported[key] = count
        self._reported_at[key] = now
        return count - reported


def _create_default_checker(arm_side: str, config: Config) -> CompositeChecker:
    """Create basic checker with joint limits."""
    joint_limits = config.get_joint_limits(arm_side)
    delta_limits = config.get_joint_delta_position_limits()
    checkers = [
        JointPosChecker(joint_limits),
        JointDeltaPosChecker(delta_limits),
    ]
    velocity_limits = config.get_joint_velocity_limits()
    if velocity_limits is not None:
        checkers.append(JointVelocityChecker(velocity_limits))
    return CompositeChecker(checkers)


class SingleArmDriver:
    """Driver for single arm."""

    def __init__(
        self,
        arm_side: str,
        config: Config | None = None,
        kps: ArrayLike | None = None,
        kds: ArrayLike | None = None,
        safety_checker: Checker | None = None,
        can_interface: str | None = None,
    ):
        """Initialize single arm driver.

        Args:
            arm_side: "left_arm" or "right_arm".
            config: Driver configuration. Uses default if None.
            kps: Proportional gains. Uses config defaults if None.
            kds: Derivative gains. Uses config defaults if None.
            safety_checker: Safety checker to use. If None, creates a basic
                           checker with joint limits.
            can_interface: SocketCAN interface to use, overriding the config.
                           Interface names are a property of the machine
                           rather than of the arm, so a host that renames
                           them can point at one without editing the config.

        """
        self.config = config if config is not None else get_default_config()
        self.arm_side = arm_side
        self.can_interface = (
            self.config.get_can_interface(self.arm_side)
            if can_interface is None
            else can_interface
        )
        self.openarm = oa.OpenArm(self.can_interface, True)
        self.latest_state = None
        self.started = False

        # Load joint offsets from config
        self.joint_offsets = self.config.get_joint_offsets(self.arm_side)

        # Load motor configuration from config
        motor_type_strs = self.config.get_motor_types()
        motor_types = [getattr(oa.MotorType, mt) for mt in motor_type_strs]
        send_ids = self.config.get_send_ids()
        recv_ids = self.config.get_recv_ids()
        self.gripper_posforce = self.config.get_gripper_posforce()
        self.gripper_posforce_limits = self.config.get_gripper_posforce_limits()

        # Initialize motors
        if self.gripper_posforce:
            self.num_mit_motors = 7
            self.openarm.init_arm_motors(motor_types[:-1], send_ids[:-1], recv_ids[:-1])
            self.openarm.init_gripper_motor(
                motor_types[-1], send_ids[-1], recv_ids[-1], oa.ControlMode.POS_FORCE
            )
        else:
            self.num_mit_motors = 8
            self.openarm.init_arm_motors(motor_types, send_ids, recv_ids)

        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)

        # Use provided gains or defaults from config
        self.kps = self.config.get_default_kps() if kps is None else np.array(kps)
        self.kds = self.config.get_default_kds() if kds is None else np.array(kds)

        # If no checker is provided, use the default safety checks.
        self.safety_checker = (
            _create_default_checker(arm_side, self.config)
            if safety_checker is None
            else safety_checker
        )

        # Bus and per-axis diagnostics are newer than the oldest openarm_can
        # this package accepts, so they are used only when present. Checked
        # once here rather than per cycle: without them there is nothing to
        # report, and the control loop should not pay for finding that out
        # several hundred times a second.
        self._health_reporting = hasattr(self.openarm, "get_bus_status") and hasattr(
            oa, "motor_error_to_string"
        )
        if self._health_reporting:
            # get_arm()/get_gripper() return references to the same underlying
            # collections for the life of self.openarm, so these are resolved
            # once. Axes carry an index into _health_collections rather than
            # the collection itself, so that the motor list of each can be
            # fetched once per cycle instead of once per axis. The per-axis
            # history below is indexed the same way as _health_axes.
            self._health_collections = [self.openarm.get_arm()]
            self._health_axes = [
                (f"arm[{i}]", 0, i) for i in range(self.num_mit_motors)
            ]
            if self.gripper_posforce:
                self._health_collections.append(self.openarm.get_gripper())
                self._health_axes.append(("gripper", 1, 0))
        else:
            logger.info(
                "%s: bus and per-axis diagnostics unavailable; the installed "
                "openarm_can does not expose them",
                self.arm_side,
            )
            self._health_collections = []
            self._health_axes = []
        self._axis_was_stale = [False] * len(self._health_axes)
        self._axis_had_error = [False] * len(self._health_axes)
        # Delivery counts as of the previous report, and when the current run
        # of unanswered commands started. Silence is judged from these rather
        # than from the time of the last reply, so that an axis nothing has
        # asked anything of is not mistaken for one that stopped answering.
        self._axis_last_sent = [0] * len(self._health_axes)
        self._axis_last_recv = [0] * len(self._health_axes)
        self._axis_unanswered_since: list[float | None] = [None] * len(
            self._health_axes
        )
        self._bus_counter_log = _IncrementReporter(FAULT_LOG_COOLDOWN_S)
        self._unmatched_log = _IncrementReporter(FAULT_LOG_COOLDOWN_S)
        self._link_was_down = False
        self._health_checked_at = 0.0

        # iterate until commutation is stable
        for _ in range(20):
            time.sleep(0.01)
            self.last_command = self.fetch_position(refresh=True)
        # Monotonic time drives rate limits; wall time correlates external events.
        self.last_command_time_s = time.monotonic()
        self.last_command_dispatch_timestamp_ns: int | None = None

    def start(self):
        """Start the arm."""
        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.openarm.enable_all()
        self.set_latest_state(timeout_us=500)
        self.openarm.refresh_all()
        self.set_latest_state(timeout_us=500)
        # Do not expose command metadata from a previous enable session.
        self.last_command_dispatch_timestamp_ns = None
        self._on_start()
        self.started = True

    def stop(self):
        """Stop the arm."""
        self._on_stop()
        self.openarm.disable_all()
        self.set_latest_state(timeout_us=1000)
        time.sleep(1)
        self.started = False

    def get_health(self) -> tuple[list[str], dict]:
        """Return the current per-axis status and bus state.

        For callers outside this class (e.g. a dora node publishing it
        alongside qpos) that want the current picture rather than a log line.
        Axis silence is read from the same tracking `_log_health_if_changed`
        already maintains rather than recomputed here, so this never
        disagrees with what was just logged -- and a caller does not need to
        know about openarm_can's API to get it, only this method's.

        Returns:
            (motor_status, bus): `motor_status` has one entry per axis, in
            qpos order (joints, then gripper) -- the motor's own status name,
            or "SILENT" if it has stopped answering. `bus` has `carrier`
            (bool) plus a count for each fault class in `_BUS_COUNTER_NAMES`.
            ([], {}) if the installed openarm_can doesn't expose these.

        """
        if not self._health_reporting:
            return [], {}
        motor_status = []
        for idx, (_, ci, i) in enumerate(self._health_axes):
            if self._axis_was_stale[idx]:
                motor_status.append("SILENT")
                continue
            motor = self._health_collections[ci].get_motors()[i]
            motor_status.append(oa.motor_error_to_string(motor.get_error_code()))

        bus = self.openarm.get_bus_status()
        bus_state = {"carrier": not self._link_was_down}
        for name in _BUS_COUNTER_NAMES:
            bus_state[name] = getattr(bus, name).count
        return motor_status, bus_state

    # Detection only. These report what openarm_can already knows about the
    # bus and each motor; whether a given fault should stop the arm depends on
    # context this class cannot see, so that decision is left to the caller.
    #
    # Logged on change rather than every call, because this runs from
    # set_latest_state, which a control loop reaches every cycle. A standing
    # fault would otherwise repeat one line hundreds of times a second.

    def _log_health_if_changed(self):
        if not self._health_reporting:
            return
        # One clock reading for the whole report, so the bus and the axes are
        # judged against the same instant, and so the throttle below costs a
        # single call rather than a second one.
        now = time.monotonic()
        if now - self._health_checked_at < HEALTH_CHECK_INTERVAL_S:
            return
        self._health_checked_at = now
        try:
            self._log_bus_health(now)
            self._log_axis_health(now)
        except Exception:
            # Reporting must never be the reason set_latest_state stops
            # returning motor state. These are attribute reads and
            # comparisons on library objects, so a failure means the shape of
            # what openarm_can returns is not what is expected here -- which
            # will be just as true next cycle. Give up rather than raise the
            # same traceback at loop rate, and say so once, since silently
            # reporting nothing is the failure this whole path exists to
            # prevent.
            self._health_reporting = False
            logger.warning(
                "%s: bus and per-axis diagnostics disabled after an unexpected failure",
                self.arm_side,
                exc_info=True,
            )

    def _log_bus_health(self, now):
        bus = self.openarm.get_bus_status()
        for name in _BUS_COUNTER_NAMES:
            counter = getattr(bus, name)
            delta = self._bus_counter_log.take(name, counter.count, now)
            if delta is not None:
                logger.warning(
                    "%s: bus %s %s +%d (total %d, latest %.2fs ago)",
                    self.arm_side,
                    self.can_interface,
                    name,
                    delta,
                    counter.count,
                    counter.seconds_ago(),
                )

        link_running = self.openarm.is_link_running()
        if not link_running and not self._link_was_down:
            logger.warning(
                "%s: interface %s lost carrier (bus-off with no auto-restart, "
                "or unplugged)",
                self.arm_side,
                self.can_interface,
            )
        elif link_running and self._link_was_down:
            logger.info(
                "%s: interface %s carrier restored",
                self.arm_side,
                self.can_interface,
            )
        self._link_was_down = not link_running

        for can_id, count in self.openarm.get_unmatched_frames().items():
            delta = self._unmatched_log.take(can_id, count, now)
            if delta is not None:
                logger.warning(
                    "%s: unmatched reply on id 0x%02X +%d (a motor's master id, "
                    "RID 7, is likely misconfigured)",
                    self.arm_side,
                    can_id,
                    delta,
                )

    def _log_axis_health(self, now):
        # get_motors() copies and builds the whole list, and is the only
        # by-collection motor accessor openarm_can exposes to Python, so it is
        # called once per collection rather than once per axis.
        motors_per_collection = [c.get_motors() for c in self._health_collections]
        for idx, (label, ci, i) in enumerate(self._health_axes):
            collection = self._health_collections[ci]
            link = collection.get_link_stats(i)

            # An unplugged motor raises no bus error at all: CAN acknowledges
            # a frame if any node hears it, so a missing axis is only visible
            # as replies that stop arriving.
            #
            # Judged on "asked and not answered" rather than on time since the
            # last reply. MotorLinkStats.is_stale() reports an axis with no
            # replies yet as stale, which every axis is for the first cycle,
            # and it cannot tell an axis that went quiet from one this loop
            # stopped addressing.
            sent, recv = link.commands_sent, link.responses
            answered = recv > self._axis_last_recv[idx]
            asked = sent > self._axis_last_sent[idx]
            self._axis_last_sent[idx] = sent
            self._axis_last_recv[idx] = recv

            if answered:
                self._axis_unanswered_since[idx] = None
                if self._axis_was_stale[idx]:
                    logger.info("%s: %s responding again", self.arm_side, label)
                    self._axis_was_stale[idx] = False
            elif asked:
                started = self._axis_unanswered_since[idx]
                if started is None:
                    self._axis_unanswered_since[idx] = now
                elif (
                    not self._axis_was_stale[idx]
                    and now - started > AXIS_STALE_TIMEOUT_S
                ):
                    logger.warning(
                        "%s: %s went silent (no reply for %.2fs, %d/%d commands "
                        "answered overall)",
                        self.arm_side,
                        label,
                        now - started,
                        recv,
                        sent,
                    )
                    self._axis_was_stale[idx] = True
            # Neither asked nor answered: nothing was addressed to this axis
            # since the last report, so there is nothing to conclude.

            motor = motors_per_collection[ci][i]
            has_error = motor.has_error()
            if has_error and not self._axis_had_error[idx]:
                logger.warning(
                    "%s: %s reports %s",
                    self.arm_side,
                    label,
                    oa.motor_error_to_string(motor.get_error_code()),
                )
            elif self._axis_had_error[idx] and not has_error:
                logger.info("%s: %s error cleared", self.arm_side, label)
            self._axis_had_error[idx] = has_error

    def set_latest_state(self, timeout_us=300):
        """Update the state."""
        self.openarm.recv_all(timeout_us)
        self._log_health_if_changed()
        motor_values = (
            (
                m.get_position(),
                m.get_velocity(),
                m.get_torque(),
                m.get_state_tmos(),
                m.get_state_trotor(),
            )
            for m in self._iter_motors()
        )
        qpos, qvel, qtau, tmos, trotor = zip(*motor_values)
        self.latest_state = {
            "qpos": np.array(qpos, dtype=float) - self.joint_offsets,
            "qvel": np.array(qvel, dtype=float),
            "qtorque": np.array(qtau, dtype=float),
            "tmos": np.array(tmos, dtype=int),
            "trotor": np.array(trotor, dtype=int),
        }

    def fetch_state(self, refresh=True) -> dict[str, np.ndarray]:
        """Fetch the state."""
        if refresh:
            self.openarm.refresh_all()
        # TODO: maybe ?
        self.set_latest_state(timeout_us=300)
        return self.latest_state

    def fetch_position(self, refresh=True) -> np.ndarray:
        """Fetch the position."""
        return self.fetch_state(refresh=refresh)["qpos"]

    def fetch_velocity(self, refresh=True) -> np.ndarray:
        """Fetch the velocity."""
        return self.fetch_state(refresh=refresh)["qvel"]

    def fetch_torque(self, refresh=True) -> np.ndarray:
        """Fetch the torque."""
        return self.fetch_state(refresh=refresh)["qtorque"]

    def fetch_mos_temperature(self, refresh=True) -> np.ndarray:
        """Fetch the MOS temperature for each motor."""
        return self.fetch_state(refresh=refresh)["tmos"]

    def fetch_rotor_temperature(self, refresh=True) -> np.ndarray:
        """Fetch the rotor temperature for each motor."""
        return self.fetch_state(refresh=refresh)["trotor"]

    def send_position(self, position: ArrayLike) -> None:
        """Move the arm by dispatching a checked position target."""
        command_time_s = time.monotonic()
        elapsed_s = max(command_time_s - self.last_command_time_s, 0.0)
        dt_s = min(elapsed_s, MAX_COMMAND_DT_S)
        checked_result = self.safety_checker.check(
            position,
            driver=self,
            dt_s=dt_s,
        )
        if not checked_result.is_safe:
            if checked_result.force_stop:
                raise RuntimeError(checked_result.message)
            if checked_result.fixed_joint_positions is not None:
                position = checked_result.fixed_joint_positions

        target_pos = np.array(position, dtype=float)
        mit_params = [
            oa.MITParam(
                self.kps[i],
                self.kds[i],
                target_pos[i] + self.joint_offsets[i],
                0,
                0,
            )
            for i in range(self.num_mit_motors)
        ]
        dispatch_timestamp_ns = time.time_ns()
        self.openarm.get_arm().mit_control_all(mit_params)
        if self.gripper_posforce:
            # TODO: Now We should multiply 10 to convert Nm to pu?
            self.openarm.get_gripper().set_position(
                target_pos[-1] + self.joint_offsets[-1],
                speed_rad_s=self.gripper_posforce_limits[0],
                torque_pu=self.gripper_posforce_limits[1] / 4.5,
            )

        self.last_command = target_pos
        self.last_command_time_s = command_time_s
        self.last_command_dispatch_timestamp_ns = dispatch_timestamp_ns
        self.set_latest_state(timeout_us=300)

    def smooth_move(
        self,
        position: ArrayLike,
        hz: float,
        duration: float,
    ):
        """Move the arm smoothly by interpolating the trajectory to the final position."""
        num_steps = int(hz * duration)
        if num_steps <= 0:
            raise ValueError(
                f"smooth_move step calc error: hz {hz}, duration: {duration}"
            )
        for smoothed_position in self._interpolate(
            np.array([np.array(self.last_command), np.array(position)]),
            num_steps=num_steps,
        ):
            self.send_position(smoothed_position)
            time.sleep(1.0 / hz)

    def move_to_start_position(self):
        """Move to start position."""
        start_config = self.config.get_start_config()
        if start_config["moves"]:
            for move in start_config["moves"]:
                self.smooth_move(
                    move["position"][self.arm_side],
                    hz=move["hz"],
                    duration=move["duration"],
                )

    def move_to_stop_position(self):
        """Move to end position."""
        end_config = self.config.get_stop_config()
        if end_config["moves"]:
            for move in end_config["moves"]:
                self.smooth_move(
                    move["position"][self.arm_side],
                    hz=move["hz"],
                    duration=move["duration"],
                )

    def _interpolate(
        self, positions: np.ndarray, num_steps: int
    ) -> Iterator[np.ndarray]:
        for a0, a1 in zip(positions[:-1], positions[1:]):
            yield from np.linspace(a0, a1, num_steps)

    def _iter_motors(self) -> Iterator:
        yield from self.openarm.get_arm().get_motors()
        if self.gripper_posforce:
            yield self.openarm.get_gripper().get_motors()[0]

    def _on_start(self):
        self.move_to_start_position()

    def _on_stop(self):
        self.move_to_stop_position()
