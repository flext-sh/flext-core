"""Constants of the flext_core beartype.claw runtime type-checking bootstrap.

The bootstrap's mode is an operational knob owned by ``FlextSettings``
(``beartype_mode``); this module owns only the fixed claw policy.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from flext_core import t


class FlextConstantsBeartype:
    """beartype.claw policy constants published under ``c``."""

    BEARTYPE_CLAW_SKIP_PACKAGES: ClassVar[t.VariadicTuple[str]] = (
        "flext_core._models.context",
        "flext_core._typings",
        "flext_core._utilities.logging_config",
        "flext_core._utilities.parser",
        "flext_core._utilities.reliability",
        "flext_core.loggings",
        "flext_core.runtime",
    )
    """Package paths skipped by the flext_core beartype bootstrap."""


__all__: list[str] = ["FlextConstantsBeartype"]
