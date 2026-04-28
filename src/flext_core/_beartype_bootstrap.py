"""Package-local beartype.claw bootstrap for flext_core imports."""

from __future__ import annotations

from importlib import import_module
from typing import ClassVar

from beartype.claw import beartype_this_package

from flext_core._typings.base import FlextTypingBase as t


class FlextCoreBeartypeBootstrap:
    """Build and activate the flext_core beartype import hook."""

    _activated: ClassVar[bool] = False

    @classmethod
    def _enforcement_constants(cls) -> type:
        """Load enforcement constants lazily to avoid package-init cycles."""
        module = import_module("flext_core._constants.enforcement")
        constants_cls: type = module.FlextConstantsEnforcement
        return constants_cls

    @classmethod
    def activate_package_beartype(cls) -> None:
        """Install beartype.claw for flext_core exactly once."""
        enforcement_constants = cls._enforcement_constants()
        if (
            cls._activated
            or enforcement_constants.BEARTYPE_MODE
            is enforcement_constants.EnforcementMode.OFF
        ):
            return
        conf_module = import_module("flext_core._utilities.beartype_conf")
        beartype_this_package(
            conf=conf_module.FlextUtilitiesBeartypeConf.build_beartype_conf()
        )
        cls._activated = True


__all__: t.MutableSequenceOf[str] = ["FlextCoreBeartypeBootstrap"]
