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

import pytest

from openarm_driver.driver import SingleArmDriver


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


def test_start(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.start()


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


def test_send_position(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.send_position([0.0] * 8)


def test_smooth_move(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.smooth_move([0.0] * 8, 50.0, 1.0)


def test_pos_limit(can_mock):
    driver = SingleArmDriver("right_arm")
    driver.send_position([1.0] * 8)
    driver.send_position([2.0] * 8)
    driver.send_position([3.0] * 8)
    assert all(pos < 3.0 for pos in driver.fetch_position())


def test_delta_pos_limit(can_mock, config_mock_hard_delta_limit):
    driver = SingleArmDriver("right_arm")
    with pytest.raises(RuntimeError):
        driver.send_position([3.0] * 8)
