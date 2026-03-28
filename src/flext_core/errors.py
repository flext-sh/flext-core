"""Structured error handling for Result types.

Provides FlextErrorDomain enum and FlextError model for categorized error handling
with proper error routing and metadata support across the FLEXT ecosystem.

The canonical definitions now live in the namespace facades:
- ``c.ErrorDomain`` (FlextConstantsErrors.ErrorDomain)
- ``m.Error`` (FlextModelsErrors.Error)

This module re-exports them for backward compatibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants.errors import FlextConstantsErrors
from flext_core._models.errors import FlextModelsErrors

FlextErrorDomain = FlextConstantsErrors.ErrorDomain
FlextError = FlextModelsErrors.Error

__all__ = ["FlextError", "FlextErrorDomain"]
