# AUTO-GENERATED FILE — Regenerate with: make gen
"""Errors Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantserrors_part_01": ("FlextConstantsErrorsMessages",),
        ".flextconstantserrors_part_02": ("FlextConstantsErrorsRuntimeExceptions",),
        ".flextconstantserrors_part_03": ("FlextConstantsErrorsValidationExceptions",),
        ".flextconstantserrors_part_04": ("FlextConstantsErrorsDomainParser",),
        ".flextconstantserrors_part_05": ("FlextConstantsErrorsRuntimeSettings",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
