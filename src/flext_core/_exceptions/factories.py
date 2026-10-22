"""Fail DSL factory facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._exceptions._factories.flextexceptionsfactories_part_04 import (
    FlextExceptionsFactoriesPart04,
)


class FlextExceptionsFactories(FlextExceptionsFactoriesPart04):
    """Public failure factory namespace."""


__all__: list[str] = ["FlextExceptionsFactories"]
