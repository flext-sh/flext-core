"""Decorator configuration models for FLEXT decorators.

This module contains configuration models for decorators that require
structured validation and serialization. Simple decorators (inject, log_operation,
track_performance, railway, combined) do not need models and use built-in types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from flext_core._models.base import FlextModelFoundation
from flext_core.constants import c


class FlextModelsDecorators:
    """Decorator configuration model container class.

    This class acts as a namespace container for decorator configuration models.
    All nested classes are accessed via FlextModels.Decorator.* in the main models.py.
    """

    class TimeoutConfig(FlextModelFoundation.ArbitraryTypesModel):
        """Timeout decorator configuration with validation.

        Validates timeout duration and optional error code for timeout handling.
        Used by @timeout decorator to enforce operation time limits.
        """

        model_config = ConfigDict(
            frozen=True,
            extra="forbid",
            validate_assignment=True,
        )

        timeout_seconds: float = Field(
            gt=0,
            description="Timeout duration in seconds (must be positive)",
        )
        error_code: str | None = Field(
            default=None,
            description="Optional error code to use when timeout occurs",
        )
