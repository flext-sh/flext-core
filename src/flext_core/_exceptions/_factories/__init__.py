# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Factories package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flext_core
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._factories.fail import FlextExceptionsFactoriesFail


__all__: tuple[str, ...] = ("FlextExceptionsFactoriesFail", "flext_core")

install_lazy_exports(
    __name__,
    globals(),
    {"FlextExceptionsFactoriesFail": ".fail"},
    public_exports=__all__,
)
