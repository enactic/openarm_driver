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

Please refer to [src/openarm_driver/config.yaml](src/openarm_driver/config.yaml), the default configuration.

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
