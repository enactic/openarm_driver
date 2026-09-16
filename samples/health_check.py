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

"""Watch what the driver can tell you about the bus and each axis.

Polls the arm and prints what came back, so that pulling a connector or
removing a terminator can be watched as it happens. The driver logs bus and
per-axis faults as they change, and those lines appear interleaved with the
table below.

    uv run samples/health_check.py right_arm -i can0

Torque stays off unless --start is passed: polling the motors works whether or
not they are armed, so the arm cannot move on its own while this runs.

Nothing here decides that a fault means "stop" -- the driver reports, and what
to do about it belongs to whoever is driving the arm.
"""

import argparse
import logging
import time

import openarm_can as oa

from openarm_driver.driver import SingleArmDriver

BUS_COUNTERS = (
    "bus_off",
    "error_passive",
    "error_warning",
    "ack_error",
    "tx_overflow",
    "rx_overflow",
    "tx_timeout",
    "restarted",
    "write_net_down",
    "write_no_buffer",
    "write_other",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_side", nargs="?", default="right_arm")
    parser.add_argument(
        "-i",
        "--can-interface",
        default=None,
        help="SocketCAN interface, overriding the config",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=30.0, help="seconds to watch"
    )
    parser.add_argument(
        "-t", "--interval", type=float, default=0.5, help="seconds between rows"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="arm the motors and run the configured start/stop moves. THE ARM "
        "WILL MOVE. Without this the motors are only polled.",
    )
    return parser.parse_args()


def iter_axes(driver):
    """Yield (label, collection, index) for every axis, gripper included."""
    yield from (
        (f"arm[{i}]", driver.openarm.get_arm(), i) for i in range(driver.num_mit_motors)
    )
    if driver.gripper_posforce:
        yield "gripper", driver.openarm.get_gripper(), 0


def print_axes(driver, diagnostics):
    header = f"  {'axis':<10}{'pos(rad)':>10}{'MOS':>5}{'Rtr':>5}"
    if diagnostics:
        header += f"  {'status':<20}{'recv/sent':>12}{'miss':>8}"
    print(header)

    motors_per_collection = {}
    for label, collection, i in iter_axes(driver):
        motors = motors_per_collection.setdefault(
            id(collection), collection.get_motors()
        )
        motor = motors[i]
        row = (
            f"  {label:<10}{motor.get_position():>10.4f}"
            f"{motor.get_state_tmos():>5}{motor.get_state_trotor():>5}"
        )
        if diagnostics:
            link = collection.get_link_stats(i)
            note = ""
            if not link.ever_responded():
                note = "  <-- never answered"
            elif motor.has_error():
                note = "  <-- FAULT"
            row += (
                f"  {oa.motor_error_to_string(motor.get_error_code()):<20}"
                f"{link.responses:>6}/{link.commands_sent:<5}"
                f"{link.miss_rate() * 100:>7.1f}%{note}"
            )
        print(row)


def print_bus(driver):
    bus = driver.openarm.get_bus_status()
    carrier = "carrier" if driver.openarm.is_link_running() else "NO CARRIER"
    print(
        f"  bus [{carrier}] healthy={bus.healthy()} "
        f"writes_ok={bus.writes_ok} error_frames={bus.error_frames}"
    )
    for name in BUS_COUNTERS:
        counter = getattr(bus, name)
        if counter:
            print(f"      {name}: x{counter.count} ({counter.seconds_ago():.2f}s ago)")
    if bus.tec or bus.rec:
        print(f"      TEC/REC {bus.tec}/{bus.rec}")
    for can_id, count in driver.openarm.get_unmatched_frames().items():
        print(f"      unmatched id 0x{can_id:02X} x{count} (check RID 7 master id)")


def main() -> int:
    args = parse_args()
    # The driver reports faults through logging, which is the point of this
    # sample, so a handler has to exist for them to go anywhere.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    print(f">>> opening {args.arm_side}")
    driver = SingleArmDriver(args.arm_side, can_interface=args.can_interface)
    print(f"    interface  : {driver.can_interface}")
    if not driver._health_reporting:
        print(
            "    diagnostics: UNAVAILABLE -- the installed openarm_can does not\n"
            "                 expose them, so only positions are shown below"
        )

    if args.start:
        print(">>> start: arming motors and running the configured moves")
        driver.start()

    try:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            driver.fetch_state(refresh=True)
            print(f"\n=== {driver.can_interface} ===")
            print_axes(driver, driver._health_reporting)
            if driver._health_reporting:
                print_bus(driver)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n>>> interrupted")
    finally:
        # Only disarm what this script armed. Sending a disable regardless
        # would drop an arm that something else is holding up.
        if args.start:
            print(">>> stop")
            driver.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
