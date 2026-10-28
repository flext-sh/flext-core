"""Extracted facade class for library.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._library import (
    FlextUtilitiesBeartypeLibraryVisitor,
)


class FlextBeartypeLibrary(
    FlextUtilitiesBeartypeLibraryVisitor,
):
    """Facade for library."""


__all__: list[str] = ["FlextBeartypeLibrary"]
