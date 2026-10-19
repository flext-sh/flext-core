"""Public API facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.services.dispatcher import FlextServiceDispatcher
from flext_core.services.registry import FlextServiceRegistry


class FlextCore(
    FlextServiceRegistry,
    FlextServiceDispatcher,
):
    """Public flext-core facade — pure cooperative MRO."""


core: FlextCore = FlextCore.fetch_global()

__all__: list[str] = ["FlextCore", "core"]
