# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Errors Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextconstantserrors_part_01 import FlextConstantsErrorsMessages
    from .flextconstantserrors_part_02 import FlextConstantsErrorsRuntimeExceptions
    from .flextconstantserrors_part_03 import FlextConstantsErrorsValidationExceptions
    from .flextconstantserrors_part_04 import FlextConstantsErrorsDomainParser
    from .flextconstantserrors_part_05 import FlextConstantsErrorsRuntimeSettings
__all__: tuple[str, ...] = (
    "FlextConstantsErrorsDomainParser", "FlextConstantsErrorsMessages", "FlextConstantsErrorsRuntimeExceptions", "FlextConstantsErrorsRuntimeSettings",
    "FlextConstantsErrorsValidationExceptions",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextconstantserrors_part_01": ("FlextConstantsErrorsMessages",),
            ".flextconstantserrors_part_02": ("FlextConstantsErrorsRuntimeExceptions",),
            ".flextconstantserrors_part_03": (
                "FlextConstantsErrorsValidationExceptions",
            ),
            ".flextconstantserrors_part_04": ("FlextConstantsErrorsDomainParser",),
            ".flextconstantserrors_part_05": ("FlextConstantsErrorsRuntimeSettings",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
