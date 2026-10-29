"""Public decorators facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._decorators import FlextDecoratorsRuntime

d = FlextDecoratorsRuntime

__all__: list[str] = ["FlextDecoratorsRuntime", "d"]
