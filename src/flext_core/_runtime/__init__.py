"""Composed runtime facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._runtime._container import FlextRuntimeContainer
from flext_core._runtime._dependency import FlextRuntimeDependencyIntegration


class FlextRuntime(
    FlextRuntimeContainer,
    FlextRuntimeDependencyIntegration,
):
    """Expose runtime normalization, DI, and validation helpers."""


__all__: list[str] = ["FlextRuntime"]
