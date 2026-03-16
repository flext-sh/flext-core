"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from flext_core import c, p, t
from flext_core._models.base import FlextModelFoundation


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class ServiceRuntime(FlextModelFoundation.ArbitraryTypesModel):
        """Runtime triple (config, context, container) for services.

        Represents the core service runtime with configuration, context,
        and dependency injection container. CQRS components (dispatcher,
        registry) should be used directly - not through FlextService.
        """

        config: p.Config
        context: p.Context
        container: p.DI

    class TraceContext(FlextModelFoundation.FrozenStrictModel):
        """Trace context for distributed tracing."""

        trace_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Distributed trace identifier shared across related service calls.",
            title="Trace Id",
            examples=["c8f2d73e-9870-4cba-b873-5b4a3f7b95f4"],
        )
        span_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Span identifier for the current service operation within a trace.",
            title="Span Id",
            examples=["9fd8d2fd-a4bc-4b15-9e8a-47f6c7dd6a11"],
        )
        parent_span_id: str | None = None

    class ServiceRetryConfiguration(
        FlextModelFoundation.FrozenStrictModel,
        FlextModelFoundation.RetryConfigurationMixin,
    ):
        """Retry configuration for operations."""

        exponential_base: Annotated[
            float,
            Field(
                default=c.Reliability.RETRY_BACKOFF_BASE,
                ge=1.0,
                description="Exponential backoff base used to calculate retry delay growth.",
                title="Exponential Base",
                examples=[2.0],
            ),
        ] = c.Reliability.RETRY_BACKOFF_BASE
        retry_on_timeout: bool = True

    class ServiceParameters(FlextModelFoundation.DynamicConfigModel):
        """Dynamic parameters for service methods - allows extra fields."""

    class ServiceFilters(FlextModelFoundation.DynamicConfigModel):
        """Filter criteria for queries - allows extra fields."""

    class ServiceData(FlextModelFoundation.DynamicConfigModel):
        """Operation data model - allows extra fields."""

    class ServiceContext(FlextModelFoundation.DynamicConfigModel):
        """Service execution context - allows extra fields."""

    class DomainServiceExecutionRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service execution request with advanced validation."""

        service_name: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Service name",
            ),
        ]
        method_name: Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Method to execute",
            ),
        ]
        parameters: FlextModelsService.ServiceParameters | None = None
        context: FlextModelsService.TraceContext | None = None
        timeout_seconds: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                gt=c.ZERO,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
                description="Timeout from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.Defaults.TIMEOUT
        execution: bool = False
        enable_validation: bool = True

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            if self.context is None:
                self.context = FlextModelsService.TraceContext()
            return self

    class BatchOperation(FlextModelFoundation.ArbitraryTypesModel):
        """Single operation in a batch."""

        operation_name: Annotated[
            str,
            Field(
                min_length=1,
                description="Operation name executed as part of the batch request.",
                title="Operation Name",
                examples=["create_user", "sync_records"],
            ),
        ]
        parameters: FlextModelsService.ServiceParameters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            return self

    class DomainServiceBatchRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: Annotated[
            list[FlextModelsService.BatchOperation],
            Field(
                default_factory=list,
                min_length=c.Reliability.RETRY_COUNT_MIN,
                max_length=c.Performance.MAX_BATCH_OPERATIONS,
                description="Ordered batch operations to execute for the target service.",
                title="Batch Operations",
                examples=[
                    [{"operation_name": "validate"}, {"operation_name": "persist"}]
                ],
            ),
        ]
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: Annotated[
            int,
            Field(
                default=c.Performance.MAX_BATCH_SIZE,
                description="Batch size from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.Performance.MAX_BATCH_SIZE
        timeout_per_operation: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                description="Timeout per operation from FlextSettings",
            ),
        ] = c.Defaults.TIMEOUT

    class DomainServiceMetricsRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[Literal["performance", "errors", "throughput"]],
            Field(
                default_factory=lambda: list(c.Cqrs.DEFAULT_METRIC_CATEGORIES),
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = c.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: Annotated[
            str,
            Field(
                default=c.Cqrs.Aggregation.AVG,
                description="Aggregation strategy applied when summarizing metric values.",
                title="Aggregation",
                examples=["avg", "sum", "max"],
            ),
        ] = c.Cqrs.Aggregation.AVG
        group_by: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Metric dimensions used to group the resulting metric series.",
                title="Group By",
                examples=[["service_name", "handler_mode"]],
            ),
        ]
        filters: FlextModelsService.ServiceFilters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.filters is None:
                self.filters = FlextModelsService.ServiceFilters()
            return self

    class DomainServiceResourceRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = c.Dispatcher.DEFAULT_SERVICE_NAME
        resource_type: Annotated[
            str,
            Field(
                default=c.Dispatcher.DEFAULT_RESOURCE_TYPE,
                pattern=c.Platform.PATTERN_IDENTIFIER,
                description="Logical resource type targeted by the request, validated as an identifier.",
                title="Resource Type",
                examples=["user", "invoice", "job"],
            ),
        ] = c.Dispatcher.DEFAULT_RESOURCE_TYPE
        resource_id: str | None = None
        resource_limit: Annotated[
            int,
            Field(
                default=c.Performance.MAX_BATCH_SIZE,
                gt=c.ZERO,
                description="Maximum number of resources to retrieve or process in this request.",
                title="Resource Limit",
                examples=[100, 500],
            ),
        ] = c.Performance.MAX_BATCH_SIZE
        action: Annotated[
            str,
            Field(
                default=c.Cqrs.Action.GET,
                description="Requested operation to perform on the target resource type.",
                title="Action",
                examples=["get", "create", "update", "delete"],
            ),
        ] = c.Cqrs.Action.GET
        data: FlextModelsService.ServiceData | None = None
        filters: FlextModelsService.ServiceFilters | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.data is None:
                self.data = FlextModelsService.ServiceData()
            if self.filters is None:
                self.filters = FlextModelsService.ServiceFilters()
            return self

    class AclResponse(FlextModelFoundation.ArbitraryTypesModel):
        """ACL (Access Control List) response model."""

        resource: Annotated[str, Field(description="Resource identifier")]
        user: Annotated[str, Field(description="User identifier")]
        action: Annotated[str, Field(description="Requested action")]
        allowed: Annotated[bool, Field(description="Whether access is allowed")]
        permissions: Annotated[
            list[str],
            Field(default_factory=list, description="Granted permissions"),
        ]
        denied_permissions: Annotated[
            list[str],
            Field(default_factory=list, description="Denied permissions"),
        ]
        context: FlextModelsService.ServiceContext | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.context is None:
                self.context = FlextModelsService.ServiceContext()
            return self

    class OperationExecutionRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: Annotated[
            str,
            Field(
                max_length=c.Performance.MAX_OPERATION_NAME_LENGTH,
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Operation name",
            ),
        ]
        operation_callable: Annotated[
            Callable[
                [t.NormalizedValue | BaseModel],
                p.ResultLike[t.NormalizedValue | BaseModel],
            ],
            Field(description="Callable operation returning result"),
        ]
        arguments: FlextModelsService.ServiceParameters | None = None
        keyword_arguments: FlextModelsService.ServiceParameters | None = None
        timeout_seconds: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                gt=c.ZERO,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
                description="Timeout from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.Defaults.TIMEOUT
        retry_config: FlextModelsService.ServiceRetryConfiguration | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.arguments is None:
                self.arguments = FlextModelsService.ServiceParameters()
            if self.keyword_arguments is None:
                self.keyword_arguments = FlextModelsService.ServiceParameters()
            if self.retry_config is None:
                self.retry_config = FlextModelsService.ServiceRetryConfiguration()
            return self

    class RuntimeBootstrapOptions(FlextModelFoundation.ArbitraryTypesModel):
        """Options for runtime bootstrapping."""

        config_type: type[BaseSettings] | None = None
        config_overrides: Mapping[str, t.Scalar] | None = None
        context: p.Context | None = None
        subproject: str | None = None
        services: Mapping[str, t.RegisterableService] | None = None
        factories: Mapping[str, t.FactoryCallable] | None = None
        resources: Mapping[str, t.ResourceCallable] | None = None
        container_overrides: Mapping[str, t.Scalar] | None = None
        wire_modules: Sequence[ModuleType | str] | None = None
        wire_packages: Sequence[str] | None = None
        wire_classes: Sequence[type] | None = None


__all__ = ["FlextModelsService"]
