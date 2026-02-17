"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.typings import t
from flext_core.utilities import u


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    # =========================================================================
    # SUPPORTING MODELS - Base classes for dynamic configuration
    # =========================================================================

    class TraceContext(FlextModelsBase.FrozenStrictModel):
        """Trace context for distributed tracing."""

        trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        parent_span_id: str | None = None

    class RetryConfiguration(FlextModelsBase.FrozenStrictModel):
        """Retry configuration for operations."""

        max_retries: int = Field(default=c.Reliability.DEFAULT_MAX_RETRIES, ge=0)
        initial_delay_seconds: float = Field(
            default=c.Reliability.DEFAULT_RETRY_DELAY_SECONDS, gt=0
        )
        max_delay_seconds: float = Field(default=c.Reliability.RETRY_BACKOFF_MAX, gt=0)
        exponential_base: float = Field(
            default=c.Reliability.RETRY_BACKOFF_BASE, ge=1.0
        )
        retry_on_timeout: bool = True

    class ServiceParameters(FlextModelsBase.DynamicConfigModel):
        """Dynamic parameters for service methods - allows extra fields."""

    class ServiceFilters(FlextModelsBase.DynamicConfigModel):
        """Filter criteria for queries - allows extra fields."""

    class ServiceData(FlextModelsBase.DynamicConfigModel):
        """Operation data model - allows extra fields."""

    class ServiceContext(FlextModelsBase.DynamicConfigModel):
        """Service execution context - allows extra fields."""

    # =========================================================================
    # REQUEST/RESPONSE MODELS
    # =========================================================================

    class DomainServiceExecutionRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service execution request with advanced validation."""

        service_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Service name",
        )
        method_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Method to execute",
        )
        parameters: FlextModelsService.ServiceParameters | None = None
        context: FlextModelsService.TraceContext | None = None
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextSettings (Config has priority over Constants)",
        )
        execution: bool = False
        enable_validation: bool = True

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.DomainServiceExecutionRequest:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            if self.context is None:
                self.context = FlextModelsService.TraceContext()
            return self

        @field_validator("timeout_seconds", mode="after")
        @classmethod
        def validate_timeout(cls, v: float) -> float:
            """Validate timeout is reasonable."""
            max_timeout_seconds = c.Performance.MAX_TIMEOUT_SECONDS
            if v <= 0:
                msg = "Timeout must be positive"
                raise ValueError(msg)
            if v > max_timeout_seconds:
                msg = f"Timeout {v}s exceeds maximum {max_timeout_seconds}s"
                raise ValueError(msg)
            return v

    class BatchOperation(FlextModelsBase.ArbitraryTypesModel):
        """Single operation in a batch."""

        operation_name: str = Field(min_length=1)
        parameters: FlextModelsService.ServiceParameters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.BatchOperation:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            return self

    class DomainServiceBatchRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: list[FlextModelsService.BatchOperation] = Field(
            default_factory=list,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            max_length=c.Performance.MAX_BATCH_OPERATIONS,
        )
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default=c.Performance.MAX_BATCH_SIZE,
            description="Batch size from FlextSettings (Config has priority over Constants)",
        )
        timeout_per_operation: float = Field(
            default=c.Defaults.TIMEOUT,
            description="Timeout per operation from FlextSettings",
        )

    class DomainServiceMetricsRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[c.Cqrs.ServiceMetricTypeLiteral],
            Field(
                default_factory=lambda: list(
                    c.Cqrs.DEFAULT_METRIC_CATEGORIES
                ),  # Constant reference, not class instance
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = c.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: str = Field(
            default=c.Cqrs.Aggregation.AVG,
        )
        group_by: list[str] = Field(default_factory=list)
        filters: FlextModelsService.ServiceFilters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.DomainServiceMetricsRequest:
            """Apply default values for optional nested classes."""
            if self.filters is None:
                self.filters = FlextModelsService.ServiceFilters()
            return self

    class DomainServiceResourceRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = c.Dispatcher.DEFAULT_SERVICE_NAME
        resource_type: str = Field(
            c.Dispatcher.DEFAULT_RESOURCE_TYPE,
            pattern=c.Platform.PATTERN_IDENTIFIER,
        )
        resource_id: str | None = None
        resource_limit: int = Field(c.Performance.MAX_BATCH_SIZE, gt=c.ZERO)
        action: str = Field(default=c.Cqrs.Action.GET)
        data: FlextModelsService.ServiceData | None = None
        filters: FlextModelsService.ServiceFilters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.DomainServiceResourceRequest:
            """Apply default values for optional nested classes."""
            if self.data is None:
                self.data = FlextModelsService.ServiceData()
            if self.filters is None:
                self.filters = FlextModelsService.ServiceFilters()
            return self

    class AclResponse(FlextModelsBase.ArbitraryTypesModel):
        """ACL (Access Control List) response model."""

        resource: str = Field(description="Resource identifier")
        user: str = Field(description="User identifier")
        action: str = Field(description="Requested action")
        allowed: bool = Field(description="Whether access is allowed")
        permissions: list[str] = Field(
            default_factory=list,
            description="Granted permissions",
        )
        denied_permissions: list[str] = Field(
            default_factory=list,
            description="Denied permissions",
        )
        context: FlextModelsService.ServiceContext | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.AclResponse:
            """Apply default values for optional nested classes."""
            if self.context is None:
                self.context = FlextModelsService.ServiceContext()
            return self

    class OperationExecutionRequest(FlextModelsBase.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str = Field(
            max_length=c.Performance.MAX_OPERATION_NAME_LENGTH,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Operation name",
        )
        operation_callable: Callable[
            [t.GeneralValueType],
            p.ResultLike[t.GeneralValueType],
        ] = Field(
            description="Callable operation returning result",
        )
        arguments: FlextModelsService.ServiceParameters | None = None
        keyword_arguments: FlextModelsService.ServiceParameters | None = None
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextSettings (Config has priority over Constants)",
        )
        retry_config: FlextModelsService.RetryConfiguration | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> FlextModelsService.OperationExecutionRequest:
            """Apply default values for optional nested classes."""
            if self.arguments is None:
                self.arguments = FlextModelsService.ServiceParameters()
            if self.keyword_arguments is None:
                self.keyword_arguments = FlextModelsService.ServiceParameters()
            if self.retry_config is None:
                self.retry_config = FlextModelsService.RetryConfiguration()
            return self

        @field_validator("operation_callable", mode="before")
        @classmethod
        def validate_operation_callable(
            cls,
            v: object,
        ) -> object:
            """Validate operation is callable."""
            validation = u.Validation.validate_callable(
                v,
                error_message=f"Operation callable must be callable, got {type(v).__name__}",
            )
            if validation.is_failure:
                msg = validation.error or "Operation callable must be callable"
                raise TypeError(msg)
            return v


RuntimeBootstrapOptions = p.RuntimeBootstrapOptions
