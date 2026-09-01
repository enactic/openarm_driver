#!/usr/bin/env python3
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

# Bring one arm up, report what came back, and put it down again. No motion is
# commanded, so this is a way to tell a wiring or configuration problem from a
# control problem before running anything that moves.
#
#     uv run python samples/start_stop.py right_arm
#     uv run python samples/start_stop.py left_arm openarm_pedestal

import argparse
import logging
import time

import openarm_can as oa

from openarm_driver import Config, SingleArmDriver

# Motor register holding the control mode.
CTRL_MODE_RID = 10

# The driver reports a gripper whose control mode never took through logging.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def print_params(driver, rid):
    # Reading a register needs PARAM callbacks, so this cannot run inside a
    # control loop. Only the gripper's mode is written at startup; the arm
    # relies on whatever its motors hold in flash, which is why reading it back
    # is the only way to see what the motors are actually in.
    # The USB adapter's round trip is about 2 ms while set_latest_state waits
    # 300 us, so replies land in the socket buffer a cycle late and are still
    # sitting there now. Reading them in PARAM mode would parse state frames as
    # parameters, which both fails and eats the reads meant for the replies.
    driver.openarm.set_callback_mode_all(oa.CallbackMode.IGNORE)
    for _ in range(5):
        driver.openarm.recv_all(2000)

    driver.openarm.set_callback_mode_all(oa.CallbackMode.PARAM)
    try:
        driver.openarm.query_param_all(rid)
        # One recv_all stops as soon as the queue drains, which happens before
        # every motor has answered.
        for _ in range(5):
            time.sleep(0.01)
            driver.openarm.recv_all(5000)
        name = next(
            (
                n
                for n in dir(oa.MotorVariable)
                if not n.startswith("_") and getattr(oa.MotorVariable, n).value == rid
            ),
            "?",
        )
        print(f"    RID {rid} ({name}) per axis:")
        for i, motor in enumerate(driver._iter_motors()):
            print(
                f"      motor[{i}] 0x{motor.get_send_can_id():02X}  {motor.get_param(rid)}"
            )
    finally:
        driver.openarm.set_callback_mode_all(oa.CallbackMode.IGNORE)
        driver.openarm.recv_all(1000)
        driver.openarm.set_callback_mode_all(oa.CallbackMode.STATE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arm_side", nargs="?", default="right_arm", choices=["right_arm", "left_arm"]
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="bundled config name or path (default: openarm_cell)",
    )
    parser.add_argument(
        "--rid",
        type=int,
        default=CTRL_MODE_RID,
        help=f"motor register to read back per axis (default: {CTRL_MODE_RID}, CTRL_MODE)",
    )
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    print(f">>> creating driver for {args.arm_side}")
    driver = SingleArmDriver(args.arm_side, config=config)
    print(f"    interface: {driver.can_interface}")

    print(">>> start")
    driver.start()

    if driver.gripper_posforce:
        ok = driver.gripper_mode_verified
        print(f"    gripper control mode verified: {ok}")
        if not ok:
            print(
                "    -> the gripper will ignore position commands; see the warning above"
            )

    state = driver.fetch_state()
    print("    qpos:", state["qpos"])
    print("    qvel:", state["qvel"])
    print("    qtau:", state["qtorque"])

    # A motor that never answered leaves its state at zero, which is
    # indistinguishable from a real reading of zero. Its reported status code
    # says which one it is: a live motor reports ENABLED, a silent one reports
    # nothing and keeps the 0 it was constructed with.
    for i, motor in enumerate(driver._iter_motors()):
        code = motor.get_error_code() if hasattr(motor, "get_error_code") else None
        code_str = f"0x{code:X}" if code is not None else "n/a"
        print(
            f"    motor[{i}] enabled={motor.is_enabled()} status={code_str} "
            f"tmos={motor.get_state_tmos()}C trotor={motor.get_state_trotor()}C"
        )

    print_params(driver, args.rid)

    time.sleep(0.5)

    print(">>> stop")
    driver.stop()
    print(">>> done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
