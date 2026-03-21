"""Decorator configuration models for FLEXT decorators.

This module contains configuration models for decorators that require
structured validation and serialization. Simple decorators (inject, log_operation,
railway, combined) do not need models and use built-in types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from flext_core import FlextModelFoundation, t


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

        model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)
        timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(description="Timeout duration in seconds (must be positive)"),
        ]
        error_code: Annotated[
            str | None,
            Field(
                default=None,
                description="Optional error code to use when timeout occurs",
            ),
        ] = None
