from __future__ import annotations

import re

from formula_foundry.openems.toolchain import DEFAULT_OPENEMS_TOOLCHAIN_PATH, load_openems_toolchain


def test_openems_toolchain_pins_container_digest() -> None:
    assert DEFAULT_OPENEMS_TOOLCHAIN_PATH.exists()
    toolchain = load_openems_toolchain()
    assert toolchain.version
    assert "@sha256:" in toolchain.docker_image
    assert re.search(r"@sha256:[0-9a-f]{64}$", toolchain.docker_image)
