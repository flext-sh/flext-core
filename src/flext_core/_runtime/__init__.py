"""Runtime facade implementation building blocks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._runtime._container import FlextRuntimeContainer
from flext_core._runtime._dependency import FlextRuntimeDependencyIntegration

__all__: list[str] = [
    "FlextRuntimeContainer",
    "FlextRuntimeDependencyIntegration",
]
