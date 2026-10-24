"""Exception base facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._exceptions._base.flextexceptionsbase_part_03 import (
    FlextExceptionsBasePart03,
)


class FlextExceptionsBase(FlextExceptionsBasePart03):
    """Public exception base namespace."""


__all__: list[str] = ["FlextExceptionsBase"]
