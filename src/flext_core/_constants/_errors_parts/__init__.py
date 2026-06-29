# AUTO-GENERATED FILE — Regenerate with: make gen
"""Errors Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._constants._errors_parts.flextconstantserrors_part_01 import (
        FlextConstantsErrorsMessages as FlextConstantsErrorsMessages,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_02 import (
        FlextConstantsErrorsRuntimeExceptions as FlextConstantsErrorsRuntimeExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_03 import (
        FlextConstantsErrorsValidationExceptions as FlextConstantsErrorsValidationExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_04 import (
        FlextConstantsErrorsDomainParser as FlextConstantsErrorsDomainParser,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_05 import (
        FlextConstantsErrorsRuntimeSettings as FlextConstantsErrorsRuntimeSettings,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextconstantserrors_part_01": ("FlextConstantsErrorsMessages",),
        ".flextconstantserrors_part_02": ("FlextConstantsErrorsRuntimeExceptions",),
        ".flextconstantserrors_part_03": ("FlextConstantsErrorsValidationExceptions",),
        ".flextconstantserrors_part_04": ("FlextConstantsErrorsDomainParser",),
        ".flextconstantserrors_part_05": ("FlextConstantsErrorsRuntimeSettings",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
