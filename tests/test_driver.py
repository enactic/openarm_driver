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

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from openarm_driver import driver as driver_module
from openarm_driver.base_safety import CheckResult
from openarm_driver.config import get_default_config
from openarm_driver.driver import SingleArmDriver
from openarm_driver.safety import JointPosChecker, JointVelocityChecker


class MotorStub:
    def __init__(self):
        self.position = 0.5
        self.velocity = 0.0
        self.torque = 0.0
        self.tmos = 25
        self.trotor = 30

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_torque(self):
        return self.torque

    def get_state_tmos(self):
        return self.tmos

    def get_state_trotor(self):
        return self.trotor


class CanMock:
    def __init__(self, *args, **kwargs):
        self.motors = []

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def init_arm_motors(self, motor_types, *args):
        self.motors = [MotorStub() for _ in range(len(motor_types))]

    def get_motors(self):
        return self.motors

    def mit_control_all(self, mit_params):
        for motor, mit_param in zip(self.motors, mit_params):
            motor.position = mit_param.q


@pytest.fixture
def can_mock(monkeypatch):
    monkeypatch.setattr("openarm_driver.driver.oa.OpenArm", CanMock)


@pytest.fixture
def config_mock_hard_delta_limit(monkeypatch):
    monkeypatch.setattr(
        "openarm_driver.config.Config.get_joint_delta_position_limits",
        lambda self: [0.5] * 8,
    )


def test_start_clears_command_dispatch_timestamp(can_mock):
    driver = SingleArmDriver("right_arm")
    assert driver.last_command_dispatch_timestamp_ns is None
    driver.last_command_dispatch_timestamp_ns = 123
    driver._on_start = lambda: None

    driver.start()

    assert driver.last_command_dispatch_timestamp_ns is None
    assert driver.started


def test_stop(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.stop()


def test_fetch_position(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.fetch_position(refresh=True)
    driver.fetch_position(refresh=False)


def test_fetch_velocity(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.fetch_velocity(refresh=True)
    driver.fetch_velocity(refresh=False)


def test_fetch_torque(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.fetch_torque(refresh=True)
    driver.fetch_torque(refresh=False)


def test_fetch_mos_temperature(can_mock):
    driver = SingleArmDriver("right_arm")
    temps = driver.fetch_mos_temperature(refresh=True)
    assert temps.tolist() == [25] * 8
    driver.fetch_mos_temperature(refresh=False)


def test_fetch_rotor_temperature(can_mock):
    driver = SingleArmDriver("right_arm")
    temps = driver.fetch_rotor_temperature(refresh=True)
    assert temps.tolist() == [30] * 8
    driver.fetch_rotor_temperature(refresh=False)


def test_fetch_state(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.fetch_state(refresh=True)
    driver.fetch_state(refresh=False)


def test_send_position(can_mock, monkeypatch):
    monkeypatch.setattr("openarm_driver.driver.time.monotonic", lambda: 1.01)
    monkeypatch.setattr("openarm_driver.driver.time.time_ns", lambda: 123)
    driver = SingleArmDriver("right_arm")
    driver.last_command = np.zeros(8)
    driver.last_command_time_s = 1.0
    requested = np.full(8, 0.01)
    expected = requested.copy()

    driver.send_position(requested)
    requested.fill(1.0)

    np.testing.assert_allclose(driver.last_command, expected)
    assert driver.last_command_dispatch_timestamp_ns == 123
    np.testing.assert_allclose(
        driver.latest_state["qpos"][: driver.num_mit_motors],
        expected[: driver.num_mit_motors],
    )


def test_send_position_does_not_update_state_on_dispatch_error(can_mock, monkeypatch):
    driver = SingleArmDriver("right_arm")
    driver.last_command = np.zeros(8)
    driver.last_command_time_s = 1.0
    driver.last_command_dispatch_timestamp_ns = 123
    monkeypatch.setattr("openarm_driver.driver.time.monotonic", lambda: 1.01)
    monkeypatch.setattr("openarm_driver.driver.time.time_ns", lambda: 456)

    def fail_dispatch(_):
        raise RuntimeError("dispatch failed")

    driver.openarm.mit_control_all = fail_dispatch

    with pytest.raises(RuntimeError, match="dispatch failed"):
        driver.send_position(np.full(8, 0.01))

    np.testing.assert_allclose(driver.last_command, np.zeros(8))
    assert driver.last_command_time_s == 1.0
    assert driver.last_command_dispatch_timestamp_ns == 123


def test_smooth_move(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.smooth_move([0.0] * 8, 50.0, 1.0)


def test_pos_limit(can_mock):
    config = get_default_config()
    checker = JointPosChecker(config.get_joint_limits("right_arm"))
    driver = SingleArmDriver("right_arm", safety_checker=checker)
    upper_limits = config.get_joint_limits("right_arm")[:, 1]
    driver.send_position(upper_limits + 1.0)
    np.testing.assert_allclose(driver.last_command, upper_limits)


def test_delta_pos_limit(can_mock, config_mock_hard_delta_limit):
    driver = SingleArmDriver("right_arm")
    with pytest.raises(RuntimeError):
        driver.send_position([3.0] * 8)


def test_velocity_limit():
    checker = JointVelocityChecker([1.0, 2.0])
    driver = SimpleNamespace(last_command=np.zeros(2))

    result = checker.check([1.0, -1.0], driver=driver, dt_s=0.1)

    assert not result.is_safe
    np.testing.assert_allclose(result.fixed_joint_positions, [0.1, -0.2])


def test_default_velocity_limit_updates_command(can_mock, monkeypatch):
    driver = SingleArmDriver("right_arm")
    driver.last_command = np.zeros(8)
    driver.last_command_time_s = 1.0
    monkeypatch.setattr(
        "openarm_driver.driver.time.monotonic",
        lambda: 1.01,
    )

    requested = np.full(8, 0.1)
    limits = np.asarray(driver.config.get_joint_velocity_limits())
    expected = np.clip(requested, -limits * 0.01, limits * 0.01)

    driver.send_position(requested)

    np.testing.assert_allclose(driver.last_command, expected)
    np.testing.assert_allclose(
        driver.latest_state["qpos"][: driver.num_mit_motors],
        expected[: driver.num_mit_motors],
    )
    assert driver.last_command_time_s == pytest.approx(1.01)


@pytest.mark.parametrize("dt_s", [np.nan, np.inf, -0.1])
def test_velocity_limit_rejects_invalid_dt(dt_s):
    checker = JointVelocityChecker([1.0])
    driver = SimpleNamespace(last_command=np.zeros(1))

    with pytest.raises(ValueError, match="period"):
        checker.check([1.0], driver=driver, dt_s=dt_s)


def test_driver_caps_command_dt(can_mock, monkeypatch):
    monkeypatch.setattr("openarm_driver.driver.time.monotonic", lambda: 2.0)

    class RecordingChecker:
        def check(self, joint_positions, **kwargs):
            self.dt_s = kwargs["dt_s"]
            return CheckResult(is_safe=True)

    checker = RecordingChecker()
    driver = SingleArmDriver("right_arm", safety_checker=checker)
    driver.last_command_time_s = 1.0
    driver.send_position(driver.last_command)

    assert checker.dt_s == pytest.approx(0.1)


# --- bus and per-axis health reporting -------------------------------------
#
# These drive the private reporting helpers directly. Going through
# SingleArmDriver.__init__ would need an openarm_can new enough to expose the
# diagnostics, and would pay the commutation-settling loop for every case.


class CounterFake:
    def __init__(self, count=0, ago=1.0):
        self.count = count
        self._ago = ago

    def seconds_ago(self):
        return self._ago


class BusFake:
    def __init__(self):
        for name in driver_module._BUS_COUNTER_NAMES:
            setattr(self, name, CounterFake())


class LinkStatsFake:
    def __init__(self, responses=0, commands_sent=0):
        self.responses = responses
        self.commands_sent = commands_sent

    def ask(self, answered=True):
        """Record one command, and its reply unless the axis is silent."""
        self.commands_sent += 1
        if answered:
            self.responses += 1


class MotorFake:
    def __init__(self, has_error=False, code=0):
        self.error = has_error
        self.code = code

    def has_error(self):
        return self.error

    def get_error_code(self):
        return self.code


class CollectionFake:
    def __init__(self, link, motor):
        self.link = link
        self.motor = motor

    def get_link_stats(self, i):
        return self.link

    def get_motors(self):
        return [self.motor]


class OpenArmFake:
    def __init__(self, bus, link_running=True, unmatched=None):
        self.bus = bus
        self.link_running = link_running
        self.unmatched = unmatched if unmatched is not None else {}

    def get_bus_status(self):
        return self.bus

    def is_link_running(self):
        return self.link_running

    def get_unmatched_frames(self):
        return self.unmatched


def make_health_driver(openarm, collections=(), axes=()):
    driver = SingleArmDriver.__new__(SingleArmDriver)
    driver.arm_side = "right_arm"
    driver.can_interface = "can0"
    driver.openarm = openarm
    driver._health_reporting = True
    driver._health_collections = list(collections)
    driver._health_axes = list(axes)
    driver._axis_was_stale = [False] * len(driver._health_axes)
    driver._axis_had_error = [False] * len(driver._health_axes)
    driver._axis_last_sent = [0] * len(driver._health_axes)
    driver._axis_last_recv = [0] * len(driver._health_axes)
    driver._axis_unanswered_since = [None] * len(driver._health_axes)
    driver._bus_counter_log = driver_module._IncrementReporter(
        driver_module.FAULT_LOG_COOLDOWN_S
    )
    driver._unmatched_log = driver_module._IncrementReporter(
        driver_module.FAULT_LOG_COOLDOWN_S
    )
    driver._link_was_down = False
    driver._health_checked_at = 0.0
    return driver


def test_bus_counter_logs_once_per_increment(caplog):
    bus = BusFake()
    driver = make_health_driver(OpenArmFake(bus))

    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(0.0)
    assert caplog.records == []

    bus.bus_off.count = 1
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(1.0)
    assert any("bus_off" in r.getMessage() for r in caplog.records)

    # A latched counter that has not moved must not be reported again; at
    # control-loop rates that would be one line per cycle forever. Well past
    # the cooldown, so silence here is about the count, not the rate limit.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(1.0 + driver_module.FAULT_LOG_COOLDOWN_S * 2)
    assert caplog.records == []


def test_carrier_loss_and_recovery_each_log_once(caplog):
    openarm = OpenArmFake(BusFake(), link_running=False)
    driver = make_health_driver(openarm)

    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_bus_health(0.0)
    assert any("lost carrier" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_bus_health(1.0)
    assert caplog.records == []

    openarm.link_running = True
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_bus_health(2.0)
    assert any("carrier restored" in r.getMessage() for r in caplog.records)


def test_unmatched_frame_reports_only_new_ones(caplog):
    unmatched = {0x00: 3}
    driver = make_health_driver(OpenArmFake(BusFake(), unmatched=unmatched))

    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(0.0)
    assert any("0x00" in r.getMessage() for r in caplog.records)

    # Past the cooldown, so silence means the count did not move.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(driver_module.FAULT_LOG_COOLDOWN_S * 2)
    assert caplog.records == []


def test_axis_silence_reported_only_after_the_timeout(caplog, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(driver_module.time, "monotonic", lambda: clock[0])
    link = LinkStatsFake()
    collection = CollectionFake(link, MotorFake())
    driver = make_health_driver(
        OpenArmFake(BusFake()), collections=[collection], axes=[("arm[0]", 0, 0)]
    )

    # Answered: healthy, nothing to say.
    link.ask(answered=True)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert caplog.records == []

    # First unanswered command only starts the clock. Reporting here is what
    # made every axis look silent on the very first cycle.
    link.ask(answered=False)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert caplog.records == []

    # Still inside the timeout.
    clock[0] += driver_module.AXIS_STALE_TIMEOUT_S / 2
    link.ask(answered=False)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert caplog.records == []

    clock[0] += driver_module.AXIS_STALE_TIMEOUT_S
    link.ask(answered=False)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert any("went silent" in r.getMessage() for r in caplog.records)

    caplog.clear()
    clock[0] += 1.0
    link.ask(answered=False)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert caplog.records == []

    link.ask(answered=True)
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(clock[0])
    assert any("responding again" in r.getMessage() for r in caplog.records)


def test_axis_not_addressed_is_not_called_silent(caplog, monkeypatch):
    # Polling without sending anything -- fetch_state(refresh=False) in a
    # loop -- must not be read as the motor having gone quiet.
    clock = [100.0]
    monkeypatch.setattr(driver_module.time, "monotonic", lambda: clock[0])
    link = LinkStatsFake()
    collection = CollectionFake(link, MotorFake())
    driver = make_health_driver(
        OpenArmFake(BusFake()), collections=[collection], axes=[("arm[0]", 0, 0)]
    )
    link.ask(answered=True)
    driver._log_axis_health(clock[0])

    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        for _ in range(5):
            clock[0] += driver_module.AXIS_STALE_TIMEOUT_S
            driver._log_axis_health(clock[0])

    assert caplog.records == []


def test_motor_fault_named_in_log_and_cleared(caplog, monkeypatch):
    monkeypatch.setattr(
        driver_module.oa,
        "motor_error_to_string",
        lambda code: "OVERCURRENT" if code == 0xA else "UNKNOWN",
        raising=False,
    )
    motor = MotorFake(has_error=False)
    collection = CollectionFake(LinkStatsFake(), motor)
    driver = make_health_driver(
        OpenArmFake(BusFake()), collections=[collection], axes=[("arm[0]", 0, 0)]
    )

    motor.error = True
    motor.code = 0xA
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(0.0)
    assert any("OVERCURRENT" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(1.0)
    assert caplog.records == []

    motor.error = False
    with caplog.at_level(logging.INFO, logger="openarm_driver.driver"):
        driver._log_axis_health(2.0)
    assert any("error cleared" in r.getMessage() for r in caplog.records)


def test_reporting_failure_does_not_propagate(caplog):
    # CanMock answers every attribute access with itself, so the counter
    # comparisons raise. Reporting is diagnostic, and must not take
    # set_latest_state down with it.
    driver = make_health_driver(CanMock())

    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_health_if_changed()

    assert any("diagnostics disabled" in r.getMessage() for r in caplog.records)


def test_reporting_gives_up_after_a_failure(caplog):
    # The failure is structural, so retrying it every cycle would repeat the
    # same traceback at loop rate without ever succeeding.
    driver = make_health_driver(CanMock())

    driver._log_health_if_changed()
    assert not driver._health_reporting

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="openarm_driver.driver"):
        driver._log_health_if_changed()
    assert caplog.records == []


def test_reporting_skipped_when_openarm_can_lacks_it():
    driver = make_health_driver(OpenArmFake(BusFake()))
    driver._health_reporting = False
    driver.openarm = None  # would raise if it were consulted

    driver._log_health_if_changed()


def test_state_update_reports_health(can_mock, monkeypatch):
    driver = SingleArmDriver("right_arm")
    calls = []
    monkeypatch.setattr(driver, "_log_health_if_changed", lambda: calls.append(True))

    driver.set_latest_state()

    assert calls, "set_latest_state must report health"


def test_can_interface_defaults_to_config(can_mock):
    driver = SingleArmDriver("right_arm")
    assert driver.can_interface == get_default_config().get_can_interface("right_arm")


def test_can_interface_can_be_overridden(can_mock):
    driver = SingleArmDriver("right_arm", can_interface="can0")
    assert driver.can_interface == "can0"


def test_repeating_counter_is_reported_once_then_summarised(caplog, monkeypatch):
    # write_other advances once per failed frame, so an interface that is down
    # moves it several times per cycle. Reporting each increment buried the
    # rest of the log under one line per frame.
    clock = [100.0]
    monkeypatch.setattr(driver_module.time, "monotonic", lambda: clock[0])
    bus = BusFake()
    driver = make_health_driver(OpenArmFake(bus))

    bus.write_other.count = 8
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(clock[0])
    assert sum("write_other" in r.getMessage() for r in caplog.records) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        for _ in range(50):
            clock[0] += 0.01
            bus.write_other.count += 8
            driver._log_bus_health(clock[0])
    assert caplog.records == []

    # Suppressed increments accumulate rather than being lost.
    clock[0] += driver_module.FAULT_LOG_COOLDOWN_S
    bus.write_other.count += 8
    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        driver._log_bus_health(clock[0])
    assert any("+408" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]
