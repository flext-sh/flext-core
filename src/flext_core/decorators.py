"""Public decorators facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._decorators import _runtime


class FlextDecoratorsRuntime(_runtime.FlextDecoratorsRuntime):
    """Public decorator facade composed through the runtime implementation MRO."""


d = FlextDecoratorsRuntime

__all__: list[str] = ["FlextDecoratorsRuntime", "d"]
