"""Facade for FlextUtilitiesBeartypeClassVisitor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._class_visitor.visitor import (
    FlextUtilitiesBeartypeClassVisitorPart03,
)


class FlextUtilitiesBeartypeClassVisitor(FlextUtilitiesBeartypeClassVisitorPart03):
    """Compose class governance through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
