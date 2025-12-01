"""Utility façade for validation, parsing, and reliability helpers.

Re-export from internal _utilities module for backward compatibility.

**CRITICAL ARCHITECTURE**: FlextUtilities is a THIN FACADE - pure delegation
to _utilities classes. No other module can import from _utilities directly.
All external code MUST use FlextUtilities as the single access point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities import FlextUtilities

__all__ = ["FlextUtilities"]
