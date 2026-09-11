# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Constants package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .data_cases import FlextTestsConstantsDataCases
    from .docker import FlextTestsConstantsDocker
    from .files import FlextTestsConstantsFiles
    from .make import FlextTestsConstantsMake
    from .matcher import FlextTestsConstantsMatcher
    from .validator import FlextTestsConstantsValidator
__all__: tuple[str, ...] = (
    "FlextTestsConstantsDataCases",
    "FlextTestsConstantsDocker",
    "FlextTestsConstantsFiles",
    "FlextTestsConstantsMake",
    "FlextTestsConstantsMatcher",
    "FlextTestsConstantsValidator",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".data_cases": ("FlextTestsConstantsDataCases",),
            ".docker": ("FlextTestsConstantsDocker",),
            ".files": ("FlextTestsConstantsFiles",),
            ".make": ("FlextTestsConstantsMake",),
            ".matcher": ("FlextTestsConstantsMatcher",),
            ".validator": ("FlextTestsConstantsValidator",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
