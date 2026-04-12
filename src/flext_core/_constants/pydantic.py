"""Pydantic v2 configuration and exception types exported via FlextConstants.

This module provides public aliases for pydantic v2 config and error classes
that are used across the flext ecosystem. All projects consuming these
must reference flext_core.c.* instead of directly from pydantic.

Architecture: Abstraction boundary - constants layer
Boundary: flext-core is sole owner of pydantic v2 integration. All other
projects receive pydantic config/exceptions ONLY through public facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import ConfigDict, SecretBytes, SecretStr, ValidationError


class FlextConstantsPydantic:
    """Public configuration and exception types from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use these via c.* instead: c.ConfigDict, c.ValidationError, c.SecretBytes, c.SecretStr

    Available constants/exceptions (accessible as c.NAME):
        ConfigDict: TypedDict for model configuration (e.g., extra="ignore")
        ValidationError: Exception raised when model validation fails
        SecretStr: String type that masks value in repr/logging
        SecretBytes: Bytes type that masks value in repr/logging
    """

    # Public Pydantic v2 config/exceptions available via c.*
    ConfigDict = ConfigDict
    ValidationError = ValidationError
    SecretStr = SecretStr
    SecretBytes = SecretBytes
