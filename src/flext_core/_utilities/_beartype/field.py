"""Extracted facade class for field.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._field import FlextUtilitiesBeartypeFieldUnionScan


class FlextBeartypeField(
    FlextUtilitiesBeartypeFieldUnionScan,
):
    """Facade for field."""


__all__: list[str] = ["FlextBeartypeField"]
