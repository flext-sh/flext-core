"""FlextSettingsDispatcher — dispatcher / executor tuning fields.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import c, t


class FlextSettingsDispatcher:
    """Dispatcher and executor settings fields."""

    dispatcher_enable_logging: Annotated[
        bool,
        Field(
            description="Enable dispatcher logging",
        ),
    ] = c.ASYNC_ENABLED
    dispatcher_auto_context: Annotated[
        bool,
        Field(
            description="Auto context in dispatcher",
        ),
    ] = c.ASYNC_ENABLED
    dispatcher_timeout_seconds: Annotated[
        t.PositiveTimeout,
        Field(
            description="Dispatcher timeout",
        ),
    ] = c.DEFAULT_TIMEOUT_SECONDS
    dispatcher_enable_metrics: Annotated[
        bool,
        Field(
            description="Enable dispatcher metrics",
        ),
    ] = c.ASYNC_ENABLED
    executor_workers: Annotated[
        t.WorkerCount, Field(description="Executor workers")
    ] = c.DEFAULT_MAX_WORKERS


__all__: list[str] = ["FlextSettingsDispatcher"]
