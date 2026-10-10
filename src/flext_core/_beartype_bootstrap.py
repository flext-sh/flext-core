"""Package-local beartype.claw bootstrap for flext_core imports.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib import import_module
from typing import ClassVar

from beartype.claw import beartype_this_package

from flext_core import c
from flext_core._typings.base import FlextTypingBase


class FlextCoreBeartypeBootstrap:
    """Build and activate the flext_core beartype import hook."""

    _activated: ClassVar[bool] = False
    _activating: ClassVar[bool] = False

    @classmethod
    def activate_package_beartype(cls) -> None:
        """Install beartype.claw for flext_core exactly once."""
        if cls._activated or cls._activating:
            return
        cls._activating = True
        try:
            if c.BEARTYPE_MODE is c.EnforcementMode.OFF:
                return
            conf_module = import_module("flext_core._utilities.beartype_conf")
            beartype_this_package(
                conf=conf_module.FlextUtilitiesBeartypeConf.build_beartype_conf(),
            )
            cls._activated = True
        finally:
            cls._activating = False


__all__: FlextTypingBase.MutableSequenceOf[str] = ["FlextCoreBeartypeBootstrap"]
