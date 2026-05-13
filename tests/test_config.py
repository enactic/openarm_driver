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
import numpy as np
import numpy.testing as npt

import openarm_driver.config
from openarm_driver.config import (
    get_default_config,
    set_default_config,
)


@pytest.fixture
def reset_default_config():
    default_config = openarm_driver.config._default_config
    yield
    openarm_driver.config._default_config = default_config


def test_get_default_config():
    config = get_default_config()
    assert config is not None


def test_set_default_config(reset_default_config):
    set_default_config("dummy config")
    config = get_default_config()
    assert config == "dummy config"


def test_get_joint_limits():
    config = get_default_config()
    right_limit = config.get_joint_limits("right_arm")
    left_limit = config.get_joint_limits("left_arm")
    npt.assert_allclose(
        right_limit,
        np.array(
            [
                [-1.3, 2.0],
                [-0.174533, 2.0],
                [-1.570796, 1.570796],
                [0.0, 2.443461],
                [-1.570796, 1.570796],
                [-0.785398, 0.785398],
                [-1.570796, 1.570796],
                [-1.047198, 0.4],
            ],
        ),
    )
    npt.assert_allclose(
        left_limit,
        np.array(
            [
                [-2.0, 1.396263],
                [-2.0, 0.174533],
                [-1.570796, 1.570796],
                [0.0, 2.443461],
                [-1.570796, 1.570796],
                [-0.785398, 0.785398],
                [-1.570796, 1.570796],
                [-0.4, 1.047198],
            ],
        ),
    )


def test_get_joint_offsets():
    config = get_default_config()
    right_offset = config.get_joint_offsets("right_arm")
    left_offset = config.get_joint_offsets("left_arm")
    npt.assert_allclose(
        right_offset,
        np.array([0.0, -0.506145, 1.570796, -1.745329, 0.0, 0.331612, -1.570796, 0.0]),
    )
    npt.assert_allclose(
        left_offset,
        np.array([0.0, 0.506145, -1.570796, -1.745329, 0.0, -0.331612, 1.570796, 0.0]),
    )


def test_get_joint_delta_position_limits():
    config = get_default_config()
    delta_limit = config.get_joint_delta_position_limits()
    assert delta_limit == [1.8, 1.8, 3.3, 2.3, 3.5, 3.5, 3.5, 3.5]


def test_get_can_interface():
    config = get_default_config()
    right_can = config.get_can_interface("right_arm")
    left_can = config.get_can_interface("left_arm")
    assert (right_can, left_can) == ("can0", "can1")


def test_get_motor_types():
    config = get_default_config()
    motor_types = config.get_motor_types()
    assert motor_types == [
        "DM8009",
        "DM8009",
        "DM4340",
        "DM4340",
        "DM4310",
        "DM4310",
        "DM4310",
        "DM4310",
    ]


def test_get_send_ids():
    config = get_default_config()
    send_ids = config.get_send_ids()
    assert send_ids == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]


def test_get_recv_ids():
    config = get_default_config()
    recv_ids = config.get_recv_ids()
    assert recv_ids == [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]


def test_get_gripper_posforce():
    config = get_default_config()
    posforce = config.get_gripper_posforce()
    assert posforce


def test_get_gripper_posforce_limits():
    config = get_default_config()
    posforce_limits = config.get_gripper_posforce_limits()
    npt.assert_allclose(posforce_limits, np.array([50.0, 1.0]))


def test_get_default_kps():
    config = get_default_config()
    kps = config.get_default_kps()
    npt.assert_allclose(kps, np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0, 10.0]))


def test_get_default_kds():
    config = get_default_config()
    kds = config.get_default_kds()
    npt.assert_allclose(kds, np.array([2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5, 0.2]))


def test_get_start_config():
    config = get_default_config()
    start_config = config.get_start_config()
    assert start_config["moves"][0]["name"] == "initial"


def test_get_stop_config():
    config = get_default_config()
    stop_config = config.get_stop_config()
    assert stop_config["moves"][0]["name"] == "initial"
