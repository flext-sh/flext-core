"""CQRS handler foundation facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._handlers_parts import FlextHandlers

h = FlextHandlers

__all__: list[str] = ["FlextHandlers", "h"]
