# OpenArm Driver

A Python library for controlling [OpenArm](https://github.com/enactic/openarm/), using [OpenArm CAN](https://github.com/enactic/openarm_can/).

## Quick start

TODO

## Install

```bash
pip install openarm-driver
```

## Sample usage

```python
import openarm_driver

arm = openarm_driver.SingleArmDriver("right_arm")
# You can also use your own config file as well.
# config = openarm_driver.Config("/path/to/config.yaml")
# arm = openarm_driver.SingleArmDriver("right_arm", config)

try:
    arm.start()
    while True:
        cur_position = arm.fetch_position()
        # Some process to calculate the next steps.
        next_positions = inference(cur_position)
        for next_postion in next_positions:
            arm.smooth_move(next_postion, hz=50, duration=1)
            # you can use simple command as well (Please be careful not to move the arm too much).
            # arm.send_position(next_postion)
finally:
    arm.stop()
```

## Config

Please refer to the [default configuration](src/openarm_driver/configs/openarm_cell.yaml).

### Bundled configurations

The package bundles several configurations. Pass a bundled name to `Config()`
to select one, or pass a path to use your own file:

```python
import openarm_driver

openarm_driver.available_configs()
# ['openarm_cell', 'openarm_cell_higher_pd', 'openarm_pedestal']

config = openarm_driver.Config("openarm_pedestal")
arm = openarm_driver.SingleArmDriver("right_arm", config)

# Or make it the default for every driver created afterwards.
openarm_driver.set_default_config(config)
```

| Name | Description |
| --- | --- |
| `openarm_cell` | Default. OpenArm mounted on the cell frame. |
| `openarm_cell_higher_pd` | Same as `openarm_cell` with higher PD gains. |
| `openarm_pedestal` | OpenArm mounted on the pedestal (zero joint offsets). |

The default safety checks run in this order:

1. `JointPosChecker` clips commands to joint position limits.
2. `JointDeltaPosChecker` rejects excessive single-command jumps.
3. `JointVelocityChecker` limits the remaining command using the elapsed time.

`joint_velocity_limits` is specified in rad/s. `send_position()` measures the
elapsed command time automatically, so callers do not need to provide the node
control frequency. Custom configurations may omit this field to disable command
velocity limiting.

Elapsed command time is capped at 40 ms. This bounds the position increment
allowed by the velocity limiter after a scheduling pause or command gap.
At 250 Hz, a normal 4 ms interval still uses 4 ms in the calculation.

## Safety stops and recovery

`send_position()` returns `True` after dispatching the checked target, including
any safety clamping. A force-stop safety rejection returns `False` and latches
the reason in the read-only `arm.safety_stop_reason` property. The rejected target
is never dispatched. Further position commands return `False`, with warnings
at most once every two seconds while commands are attempted. This replaces the
previous `RuntimeError` for force-stop safety rejections.

The latch stops new position commands and leaves the last dispatched target in
place. It does not automatically disable motors or periodically resend that
target; actual holding behavior depends on the motor and communication state.
An explicit `stop()` skips the return trajectory after a latch, logs the reason,
and calls `disable_all()`. If a safety rejection interrupts a normal stop
trajectory, `stop()` still proceeds to disable the motors.

After inspecting and resolving the cause, recover with `stop()` followed by
`start()`. Starting clears the latch and synchronizes the command baseline to
the position read from the motors before sending the startup trajectory.
`safety_stop_reason` is `None` when no safety stop is latched. The existing
`get_health()` return structure remains `(motor_status, bus)`.

`smooth_move()`, `move_to_start_position()`, `move_to_stop_position()`, and
`start()` return `False` when their trajectory is rejected and `True` when it
finishes dispatching. Rejection stops the remaining trajectory steps. A failed
start leaves `started=False`, retains the reason, and blocks subsequent position
commands until recovery. Motors may still be enabled until `stop()` is called.
Successful dispatch does not verify physical arrival at the target.

Existing callers can continue to ignore these return values. Custom trajectory
hooks returning `None` remain supported; an explicit `False` or a latched safety
stop prevents successful startup. Configuration errors and CAN exceptions retain
their exception behavior. This change requires no additional node inputs or
metadata. It adds no fresh-feedback startup gate: cached feedback can still be
stale when motors are not responding. Health diagnostics remain observational.

## Development

### Test

```bash
uv sync
uv run pytest
```

### Release

```bash
git clone git@github.com:enactic/openarm_driver.git
cd openarm_driver
dev/release.sh ${VERSION} # e.g. dev/release.sh 1.0.0
```

## Related links

- 📚 Read the [documentation](https://docs.openarm.dev/software/can/)
- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License

Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
