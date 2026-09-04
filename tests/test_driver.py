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


def test_start_syncs_command_to_measured_position(can_mock):
    driver = SingleArmDriver("right_arm")
    driver._on_start = lambda: None

    driver.start()

    np.testing.assert_allclose(driver.last_command, driver.latest_state["qpos"])
    assert driver.started


def test_stop_skips_motion_after_safety_stop(can_mock, monkeypatch):
    driver = SingleArmDriver("right_arm")
    disable_calls = []
    driver.started = True
    driver._safety_stop_reason = "test safety stop"
    driver._on_stop = lambda: pytest.fail("stop trajectory should be skipped")
    driver.openarm.disable_all = lambda: disable_calls.append(True)
    monkeypatch.setattr("openarm_driver.driver.time.sleep", lambda _: None)

    driver.stop()

    assert not driver.started
    assert disable_calls == [True]


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
    command_times = iter([1.0, 1.01])
    monkeypatch.setattr(
        "openarm_driver.driver.time.monotonic",
        lambda: next(command_times),
    )
    driver = SingleArmDriver("right_arm")
    driver.last_command = np.zeros(8)
    requested = np.full(8, 0.01)

    driver.send_position(requested)

    np.testing.assert_allclose(driver.last_command, requested)
    np.testing.assert_allclose(
        driver.latest_state["qpos"][: driver.num_mit_motors],
        requested[: driver.num_mit_motors],
    )


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


def test_delta_pos_limit(can_mock, config_mock_hard_delta_limit, caplog, monkeypatch):
    driver = SingleArmDriver("right_arm")
    previous = driver.last_command.copy()
    command_times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(
        "openarm_driver.driver.time.monotonic",
        lambda: next(command_times),
    )

    with caplog.at_level(logging.WARNING, logger="openarm_driver.driver"):
        results = [driver.send_position([3.0] * 8) for _ in range(3)]

    assert results == [False, False, False]
    assert driver._safety_stop_reason is not None
    np.testing.assert_allclose(driver.last_command, previous)
    assert sum("Safety stop" in record.message for record in caplog.records) == 2


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
    command_times = iter([1.0, 2.0])
    monkeypatch.setattr(
        "openarm_driver.driver.time.monotonic",
        lambda: next(command_times),
    )

    class RecordingChecker:
        def check(self, joint_positions, **kwargs):
            self.dt_s = kwargs["dt_s"]
            return CheckResult(is_safe=True)

    checker = RecordingChecker()
    driver = SingleArmDriver("right_arm", safety_checker=checker)
    driver.send_position(driver.last_command)

    assert checker.dt_s == pytest.approx(0.04)
