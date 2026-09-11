# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Constants package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .base import FlextCliConstantsBase
    from .config import FlextCliConstantsConfig
    from .docx import FlextCliConstantsDocx
    from .enums import FlextCliConstantsEnums
    from .errors import FlextCliConstantsErrors
    from .exceptions import (
        CliDefinitionError,
        CliValidationError,
        FlextCliConstantsExceptions,
    )
    from .files import FlextCliConstantsFiles
    from .output import FlextCliConstantsOutput
    from .pptx import FlextCliConstantsPptx
    from .settings import FlextCliConstantsSettings
    from .xlsx import FlextCliConstantsXlsx
    from .xlsx_future_functions import FlextCliConstantsXlsxFutureFunctions
__all__: tuple[str, ...] = (
    "CliDefinitionError",
    "CliValidationError",
    "FlextCliConstantsBase",
    "FlextCliConstantsConfig",
    "FlextCliConstantsDocx",
    "FlextCliConstantsEnums",
    "FlextCliConstantsErrors",
    "FlextCliConstantsExceptions",
    "FlextCliConstantsFiles",
    "FlextCliConstantsOutput",
    "FlextCliConstantsPptx",
    "FlextCliConstantsSettings",
    "FlextCliConstantsXlsx",
    "FlextCliConstantsXlsxFutureFunctions",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("FlextCliConstantsBase",),
            ".config": ("FlextCliConstantsConfig",),
            ".docx": ("FlextCliConstantsDocx",),
            ".enums": ("FlextCliConstantsEnums",),
            ".errors": ("FlextCliConstantsErrors",),
            ".exceptions": (
                "CliDefinitionError",
                "CliValidationError",
                "FlextCliConstantsExceptions",
            ),
            ".files": ("FlextCliConstantsFiles",),
            ".output": ("FlextCliConstantsOutput",),
            ".pptx": ("FlextCliConstantsPptx",),
            ".settings": ("FlextCliConstantsSettings",),
            ".xlsx": ("FlextCliConstantsXlsx",),
            ".xlsx_future_functions": ("FlextCliConstantsXlsxFutureFunctions",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
