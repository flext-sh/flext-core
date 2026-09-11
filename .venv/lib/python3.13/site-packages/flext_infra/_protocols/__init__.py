# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra. Protocols package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .base import FlextInfraProtocolsBase
    from .check import FlextInfraProtocolsCheck
    from .deps import FlextInfraProtocolsDeps
    from .docs import FlextInfraProtocolsDocs
    from .rope import FlextInfraProtocolsRope
    from .rope_runtime import FlextInfraProtocolsRopeRuntime
__all__: tuple[str, ...] = (
    "FlextInfraProtocolsBase",
    "FlextInfraProtocolsCheck",
    "FlextInfraProtocolsDeps",
    "FlextInfraProtocolsDocs",
    "FlextInfraProtocolsRope",
    "FlextInfraProtocolsRopeRuntime",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("FlextInfraProtocolsBase",),
            ".check": ("FlextInfraProtocolsCheck",),
            ".deps": ("FlextInfraProtocolsDeps",),
            ".docs": ("FlextInfraProtocolsDocs",),
            ".rope": ("FlextInfraProtocolsRope",),
            ".rope_runtime": ("FlextInfraProtocolsRopeRuntime",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
