"""Structured logging namespace.

The composed class lives inside its private family package
(``flext_core._utilities``) so canonical facade files import it from a
private-family origin (ENFORCE-046); this module re-exports the single
canonical binding.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities import FlextUtilitiesLogging

__all__: list[str] = ["FlextUtilitiesLogging"]
