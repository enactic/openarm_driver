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
