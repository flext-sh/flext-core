"""FlextSettingsInfrastructure — resilience, rate-limiting, batching fields.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import FlextConstants as c, FlextTypes as t


class FlextSettingsInfrastructure:
    """Infrastructure resilience and resource-limit settings."""

    circuit_breaker_threshold: Annotated[
        t.PositiveInt,
        Field(
            description="Circuit breaker threshold",
        ),
    ] = c.BACKUP_COUNT
    rate_limit_max_requests: Annotated[
        t.PositiveInt,
        Field(
            description="Rate limit max requests",
        ),
    ] = c.HTTP_STATUS_MIN
    rate_limit_window_seconds: Annotated[
        t.PositiveInt,
        Field(
            description="Rate limit window",
        ),
    ] = c.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    retry_delay: Annotated[
        t.PositiveInt,
        Field(
            description="Retry delay",
        ),
    ] = c.DEFAULT_RETRY_DELAY_SECONDS
    max_retry_attempts: Annotated[
        t.RetryCount,
        Field(
            description="Max retry attempts",
        ),
    ] = c.MAX_RETRY_ATTEMPTS
    enable_timeout_executor: Annotated[
        bool, Field(description="Enable timeout executor")
    ] = True
    timeout_seconds: Annotated[
        t.PositiveTimeout, Field(description="Default timeout")
    ] = c.DEFAULT_TIMEOUT_SECONDS
    max_workers: Annotated[t.WorkerCount, Field(description="Max workers")] = (
        c.DEFAULT_MAX_WORKERS
    )
    max_batch_size: Annotated[t.BatchSize, Field(description="Max batch size")] = (
        c.MAX_ITEMS
    )
    api_key: Annotated[str | None, Field(description="API key")] = None
    exception_failure_level: Annotated[
        c.FailureLevel,
        Field(
            description="Exception failure level",
        ),
    ] = c.FAILURE_LEVEL_DEFAULT


__all__: list[str] = ["FlextSettingsInfrastructure"]
