"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, cast

from pydantic import Field, field_validator

from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class DomainServiceExecutionRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Domain service execution request with advanced validation."""

        service_name: str = Field(min_length=1, description="Service name")
        method_name: str = Field(min_length=1, description="Method to execute")
        parameters: dict[str, object] = Field(default_factory=dict)
        context: dict[str, object] = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            gt=0,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextConfig (Config has priority over Constants)",
        )
        execution: bool = False
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(cls, v: object) -> dict[str, object]:
            """Ensure context has required fields (using FlextUtilities.Generators)."""
            return FlextUtilities.Generators.ensure_trace_context(v)

        @field_validator("timeout_seconds", mode="after")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            """Validate timeout is reasonable (using FlextUtilities.Validation)."""
            max_timeout_seconds = FlextConstants.Performance.MAX_TIMEOUT_SECONDS
            result = FlextUtilities.Validation.validate_timeout(v, max_timeout_seconds)
            if result.is_failure:
                base_msg = "Timeout validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (invalid timeout value)"
                )
                raise ValueError(error_msg)
            return v

    class DomainServiceBatchRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: list[dict[str, object]] = Field(
            default_factory=list,
            min_length=1,
            max_length=FlextConstants.Performance.MAX_BATCH_OPERATIONS,
        )
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default=FlextConstants.Processing.MAX_BATCH_SIZE,
            description="Batch size from FlextConfig (Config has priority over Constants)",
        )
        timeout_per_operation: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            description="Timeout per operation from FlextConfig (Config has priority over Constants)",
        )

    class DomainServiceMetricsRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[FlextConstants.Cqrs.ServiceMetricTypeLiteral],
            Field(
                default_factory=lambda: [
                    "performance",
                    "errors",
                    "throughput",
                ],
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = FlextConstants.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.Aggregation.AVG,
        )
        group_by: list[str] = Field(default_factory=list)
        filters: dict[str, object] = Field(default_factory=dict)

    class DomainServiceResourceRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = "default_service"
        resource_type: str = Field(
            "default_resource",
            pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
        )
        resource_id: str | None = None
        resource_limit: int = Field(1000, gt=0)
        action: str = Field(default_factory=lambda: FlextConstants.Cqrs.Action.GET)
        data: dict[str, object] = Field(default_factory=dict)
        filters: dict[str, object] = Field(default_factory=dict)

    class OperationExecutionRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str = Field(
            max_length=FlextConstants.Performance.MAX_OPERATION_NAME_LENGTH,
            min_length=1,
            description="Operation name",
        )
        operation_callable: Callable[..., FlextTypes.ResultLike[object]]
        arguments: dict[str, object] = Field(default_factory=dict)
        keyword_arguments: dict[str, object] = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            gt=0,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextConfig (Config has priority over Constants)",
        )
        retry_config: dict[str, object] = Field(default_factory=dict)

        @field_validator("operation_callable", mode="after")
        @classmethod
        def validate_operation_callable(
            cls, v: object
        ) -> Callable[..., FlextTypes.ResultLike[object]]:
            """Validate operation is callable (using FlextUtilities.Validation)."""
            result = FlextUtilities.Validation.validate_callable(
                v,
                error_message="Operation must be callable",
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )
            if result.is_failure:
                raise TypeError(
                    f"Operation must be callable: {result.error}"
                    if result.error
                    else "Operation must be callable (validation failed)",
                )
            # Type-safe return: v is confirmed callable by validation
            return cast("Callable[..., FlextTypes.ResultLike[object]]", v)


__all__ = ["FlextModelsService"]
