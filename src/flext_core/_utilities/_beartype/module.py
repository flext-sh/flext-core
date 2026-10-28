"""Extracted facade class for module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._module import (
    FlextUtilitiesBeartypeModuleSourceDeferred,
)


class FlextBeartypeModule(
    FlextUtilitiesBeartypeModuleSourceDeferred,
):
    """Facade for module."""


__all__: list[str] = ["FlextBeartypeModule"]
