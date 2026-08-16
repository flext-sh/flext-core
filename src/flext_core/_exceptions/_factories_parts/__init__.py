# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Factories Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextexceptionsfactories_part_04 import FlextExceptionsFactories
__all__: tuple[str, ...] = ("FlextExceptionsFactories",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                ".flextexceptionsfactories_part_04": ("FlextExceptionsFactories",)
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
