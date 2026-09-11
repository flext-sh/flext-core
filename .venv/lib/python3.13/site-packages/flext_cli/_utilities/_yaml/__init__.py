# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Utilities. Yaml package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._convert import FlextCliUtilitiesYamlConvertMixin
    from ._editing import FlextCliUtilitiesYamlEditingMixin
    from ._engine import FlextCliUtilitiesYamlEngineMixin
__all__: tuple[str, ...] = (
    "FlextCliUtilitiesYamlConvertMixin",
    "FlextCliUtilitiesYamlEditingMixin",
    "FlextCliUtilitiesYamlEngineMixin",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._convert": ("FlextCliUtilitiesYamlConvertMixin",),
            "._editing": ("FlextCliUtilitiesYamlEditingMixin",),
            "._engine": ("FlextCliUtilitiesYamlEngineMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
