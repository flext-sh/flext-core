# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.docs package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .auditor import FlextInfraDocAuditor
    from .auditor_mixin import FlextInfraDocAuditorMixin
    from .base import FlextInfraDocServiceBase
    from .builder import FlextInfraDocBuilder
    from .fixer import FlextInfraDocFixer
    from .generator import FlextInfraDocGenerator
    from .server import FlextInfraDocServer
    from .validator import FlextInfraDocValidator
__all__: tuple[str, ...] = (
    "FlextInfraDocAuditor",
    "FlextInfraDocAuditorMixin",
    "FlextInfraDocBuilder",
    "FlextInfraDocFixer",
    "FlextInfraDocGenerator",
    "FlextInfraDocServer",
    "FlextInfraDocServiceBase",
    "FlextInfraDocValidator",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".auditor": ("FlextInfraDocAuditor",),
            ".auditor_mixin": ("FlextInfraDocAuditorMixin",),
            ".base": ("FlextInfraDocServiceBase",),
            ".builder": ("FlextInfraDocBuilder",),
            ".fixer": ("FlextInfraDocFixer",),
            ".generator": ("FlextInfraDocGenerator",),
            ".server": ("FlextInfraDocServer",),
            ".validator": ("FlextInfraDocValidator",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
