"""Facade for FlextUtilitiesMapperAccess.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._mapper_access_parts.mapper_access_part_02 import (
    FlextUtilitiesMapperAccessPart02,
)


class FlextUtilitiesMapperAccess(FlextUtilitiesMapperAccessPart02):
    """Compose mapper access through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesMapperAccess"]
