"""FlextConstantsPagination - page size constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsPagination:
    """SSOT for pagination-related constants."""

    DEFAULT_PAGE_SIZE: Final[int] = 10
    MAX_PAGE_SIZE: Final[int] = 1000
    MIN_PAGE_SIZE: Final[int] = 1
