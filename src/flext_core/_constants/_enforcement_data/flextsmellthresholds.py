"""Smell threshold model for the enforcement smell audit.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel


class FlextSmellThresholds(BaseModel):
    """Complexity thresholds for the smell audit."""

    params: int
    returns: int
    nesting: int
    fn_cx: int
    file_cx: int


__all__: list[str] = ["FlextSmellThresholds"]
