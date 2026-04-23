"""Package-local beartype.claw bootstrap for flext_core imports."""

from __future__ import annotations

from importlib import import_module
from typing import ClassVar

from beartype import BeartypeConf, BeartypeStrategy
from beartype.claw import beartype_this_package


class FlextCoreBeartypeBootstrap:
    """Build and activate the flext_core beartype import hook."""

    _activated: ClassVar[bool] = False

    @classmethod
    def _enforcement_constants(cls) -> type:
        """Load enforcement constants lazily to avoid package-init cycles."""
        module = import_module("flext_core._constants.enforcement")
        return module.FlextConstantsEnforcement

    @classmethod
    def build_beartype_conf(cls) -> BeartypeConf:
        """Return the package beartype configuration for the current mode."""
        FlextConstantsEnforcement = cls._enforcement_constants()
        mode = FlextConstantsEnforcement.BEARTYPE_MODE
        if mode is FlextConstantsEnforcement.EnforcementMode.OFF:
            return BeartypeConf(strategy=BeartypeStrategy.O0)
        return BeartypeConf(
            violation_type=UserWarning
            if mode is FlextConstantsEnforcement.EnforcementMode.WARN
            else TypeError,
            strategy=BeartypeStrategy.O1,
            claw_skip_package_names=FlextConstantsEnforcement.BEARTYPE_CLAW_SKIP_PACKAGES,
        )

    @classmethod
    def activate_package_beartype(cls) -> None:
        """Install beartype.claw for flext_core exactly once."""
        FlextConstantsEnforcement = cls._enforcement_constants()
        if (
            cls._activated
            or FlextConstantsEnforcement.BEARTYPE_MODE
            is FlextConstantsEnforcement.EnforcementMode.OFF
        ):
            return
        beartype_this_package(conf=cls.build_beartype_conf())
        cls._activated = True


__all__: list[str] = ["FlextCoreBeartypeBootstrap"]
