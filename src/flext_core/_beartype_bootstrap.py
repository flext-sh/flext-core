"""Package-local beartype.claw bootstrap for flext_core imports.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib import import_module
from typing import ClassVar

from flext_core._typings import FlextTypingBase


class FlextCoreBeartypeBootstrap:
    """Build and activate the flext_core beartype import hook."""

    _activated: ClassVar[bool] = False
    _activating: ClassVar[bool] = False

    @classmethod
    def activate_package_beartype(cls) -> None:
        """Resolve the flext_core beartype.claw decision exactly once per process.

        The mode is the operational ``FlextSettings.beartype_mode``, read once
        from the layer-zero settings singleton; the claw is imported and
        installed only when that mode enables it, so a disabled mode costs
        nothing on later lazy loads.
        """
        if cls._activated or cls._activating:
            return
        cls._activating = True
        try:
            # Layer-zero modules only: the settings and conf modules are
            # imported directly (never through a facade) because this runs
            # while flext_core's own lazy exports are still being resolved.
            settings = import_module("flext_core._settings").FlextSettings
            mode = settings.fetch_global().beartype_mode
            if mode != "off":
                claw = import_module("beartype.claw")
                conf_module = import_module("flext_core._utilities.beartype_conf")
                claw.beartype_this_package(
                    conf=conf_module.FlextUtilitiesBeartypeConf.build_beartype_conf(
                        mode
                    ),
                )
            cls._activated = True
        finally:
            cls._activating = False


__all__: FlextTypingBase.MutableSequenceOf[str] = ["FlextCoreBeartypeBootstrap"]
