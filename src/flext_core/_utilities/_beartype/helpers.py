"""Facade for FlextUtilitiesBeartypeHelpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
    FlextUtilitiesBeartypeHelpersPart03,
)


class FlextUtilitiesBeartypeHelpers(FlextUtilitiesBeartypeHelpersPart03):
    """Compose beartype helpers through their internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
