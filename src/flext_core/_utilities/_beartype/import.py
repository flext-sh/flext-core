"""Extracted facade class for import.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._import import (
    FlextUtilitiesBeartypeImportBlacklistVisitor,
)


class FlextBeartypeImport(
    FlextUtilitiesBeartypeImportBlacklistVisitor,
):
    """Facade for import."""


__all__: list[str] = ["FlextBeartypeImport"]
