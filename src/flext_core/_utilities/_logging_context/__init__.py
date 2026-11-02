# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging_context.bound_logger import (
        FlextUtilitiesLoggingBoundLogger,
    )


__all__: tuple[str, ...] = ("FlextUtilitiesLoggingBoundLogger",)

install_lazy_exports(
    __name__,
    globals(),
    {"FlextUtilitiesLoggingBoundLogger": ".bound_logger"},
    public_exports=__all__,
)
