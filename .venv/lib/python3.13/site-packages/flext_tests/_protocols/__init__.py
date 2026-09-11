# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Protocols package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .enforcement import FlextTestsEnforcementProtocolsMixin
    from .valuefactory import FlextTestsValueFactoryProtocolsMixin
    from .workspace_cleanup import FlextTestsWorkspaceCleanupProtocolsMixin
__all__: tuple[str, ...] = (
    "FlextTestsEnforcementProtocolsMixin",
    "FlextTestsValueFactoryProtocolsMixin",
    "FlextTestsWorkspaceCleanupProtocolsMixin",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".enforcement": ("FlextTestsEnforcementProtocolsMixin",),
            ".valuefactory": ("FlextTestsValueFactoryProtocolsMixin",),
            ".workspace_cleanup": ("FlextTestsWorkspaceCleanupProtocolsMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
