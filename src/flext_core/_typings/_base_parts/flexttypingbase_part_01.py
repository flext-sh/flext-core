"""External type-family composition for the canonical typing base.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings import FlextTypesAnnotateds, FlextTypesPydantic


class FlextTypingBase(FlextTypesPydantic, FlextTypesAnnotateds):
    """Compose the foundational type families before local alias declarations."""


__all__: list[str] = ["FlextTypingBase"]
