# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Errors Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants._errors_parts.flextconstantserrors_part_01 import (
        FlextConstantsErrorsMessages,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_02 import (
        FlextConstantsErrorsRuntimeExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_03 import (
        FlextConstantsErrorsValidationExceptions,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_04 import (
        FlextConstantsErrorsDomainParser,
    )
    from flext_core._constants._errors_parts.flextconstantserrors_part_05 import (
        FlextConstantsErrorsRuntimeSettings,
    )


__all__: tuple[str, ...] = (
    "FlextConstantsErrorsDomainParser",
    "FlextConstantsErrorsMessages",
    "FlextConstantsErrorsRuntimeExceptions",
    "FlextConstantsErrorsRuntimeSettings",
    "FlextConstantsErrorsValidationExceptions",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextConstantsErrorsDomainParser": ".flextconstantserrors_part_04",
        "FlextConstantsErrorsMessages": ".flextconstantserrors_part_01",
        "FlextConstantsErrorsRuntimeExceptions": ".flextconstantserrors_part_02",
        "FlextConstantsErrorsRuntimeSettings": ".flextconstantserrors_part_05",
        "FlextConstantsErrorsValidationExceptions": ".flextconstantserrors_part_03",
    }),
    public_exports=__all__,
)[0]
