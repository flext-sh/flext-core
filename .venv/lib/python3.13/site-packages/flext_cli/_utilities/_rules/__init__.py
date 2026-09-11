# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Utilities. Rules package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._loaders import FlextCliUtilitiesRulesLoadersMixin
    from ._matchers import FlextCliUtilitiesRulesMatchersMixin
__all__: tuple[str, ...] = (
    "FlextCliUtilitiesRulesLoadersMixin",
    "FlextCliUtilitiesRulesMatchersMixin",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._loaders": ("FlextCliUtilitiesRulesLoadersMixin",),
            "._matchers": ("FlextCliUtilitiesRulesMatchersMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
