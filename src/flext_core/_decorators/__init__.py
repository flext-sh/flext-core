"""Composed decorator facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._decorators._runtime import FlextDecoratorsRuntime


class FlextDecorators(FlextDecoratorsRuntime):
    """Automation decorators for infrastructure concerns."""


__all__: list[str] = ["FlextDecorators"]
