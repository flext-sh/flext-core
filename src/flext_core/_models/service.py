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
from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from flext_core import c, p, t
from flext_core._models.base import FlextModelFoundation


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class RuntimeBootstrapOptions(FlextModelFoundation.ArbitraryTypesModel):
        """Runtime bootstrap options for service initialization."""

        config_type: type[BaseSettings] | None = Field(
            default=None,
            description="Settings model class used to bootstrap runtime configuration.",
            title="Config Type",
            examples=["AppSettings"],
        )
        config_overrides: Mapping[str, t.Scalar] | None = Field(
            default=None,
            description="Configuration key overrides applied before runtime initialization.",
            title="Config Overrides",
            examples=[{"LOG_LEVEL": "DEBUG"}],
        )
        context: p.Context | None = Field(
            default=None,
            description="Initial context object injected into the service runtime scope.",
            title="Runtime Context",
        )
        subproject: str | None = Field(
            default=None,
            description="Subproject identifier used to scope runtime dependencies and settings.",
            title="Subproject",
            examples=["flext-core"],
        )
        services: Mapping[str, t.RegisterableService] | None = Field(
            default=None,
            description="Pre-registered service instances keyed by service name.",
            title="Services",
            examples=[{"logger": "service-instance"}],
        )
        factories: Mapping[str, t.FactoryCallable] | None = Field(
            default=None,
            description="Factory callables used to lazily create service instances.",
            title="Factories",
            examples=[{"db": "factory-callable"}],
        )
        resources: Mapping[str, t.ResourceCallable] | None = Field(
            default=None,
            description="Resource factory callables for lifecycle-managed dependencies.",
            title="Resources",
            examples=[{"redis": "resource-callable"}],
        )
        container_overrides: Mapping[str, t.Scalar] | None = Field(
            default=None,
            description="Dependency container configuration overrides applied at bootstrap.",
            title="Container Overrides",
            examples=[{"max_services": 256}],
        )
        wire_modules: Sequence[ModuleType] | None = Field(
            default=None,
            description="Python modules to wire for dependency injection.",
            title="Wire Modules",
        )
        wire_packages: Sequence[str] | None = Field(
            default=None,
            description="Package names to scan and wire for dependency injection.",
            title="Wire Packages",
            examples=[["app.api", "app.services"]],
        )
        wire_classes: Sequence[type] | None = Field(
            default=None,
            description="Concrete classes to wire explicitly in the dependency container.",
            title="Wire Classes",
            examples=[["UserService", "OrderService"]],
        )

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

    class RetryConfiguration(
        FlextModelFoundation.FrozenStrictModel,
        FlextModelFoundation.RetryConfigurationMixin,
    ):
        """Retry configuration for operations."""

        exponential_base: float = Field(
            default=c.Reliability.RETRY_BACKOFF_BASE,
            ge=1.0,
            description="Exponential backoff base used to calculate retry delay growth.",
            title="Exponential Base",
            examples=[2.0],
        )
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

        service_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN, description="Service name"
        )
        method_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN, description="Method to execute"
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
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            if self.context is None:
                self.context = FlextModelsService.TraceContext()
            return self

    class BatchOperation(FlextModelFoundation.ArbitraryTypesModel):
        """Single operation in a batch."""

        operation_name: str = Field(
            min_length=1,
            description="Operation name executed as part of the batch request.",
            title="Operation Name",
            examples=["create_user", "sync_records"],
        )
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
        operations: list[FlextModelsService.BatchOperation] = Field(
            default_factory=lambda: list[FlextModelsService.BatchOperation](),
            min_length=c.Reliability.RETRY_COUNT_MIN,
            max_length=c.Performance.MAX_BATCH_OPERATIONS,
            description="Ordered batch operations to execute for the target service.",
            title="Batch Operations",
            examples=[[{"operation_name": "validate"}, {"operation_name": "persist"}]],
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

    class DomainServiceMetricsRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[c.Cqrs.ServiceMetricTypeLiteral],
            Field(
                default_factory=lambda: list(c.Cqrs.DEFAULT_METRIC_CATEGORIES),
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = c.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: str = Field(
            default=c.Cqrs.Aggregation.AVG,
            description="Aggregation strategy applied when summarizing metric values.",
            title="Aggregation",
            examples=["avg", "sum", "max"],
        )
        group_by: list[str] = Field(
            default_factory=list,
            description="Metric dimensions used to group the resulting metric series.",
            title="Group By",
            examples=[["service_name", "handler_mode"]],
        )
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
        resource_type: str = Field(
            c.Dispatcher.DEFAULT_RESOURCE_TYPE,
            pattern=c.Platform.PATTERN_IDENTIFIER,
            description="Logical resource type targeted by the request, validated as an identifier.",
            title="Resource Type",
            examples=["user", "invoice", "job"],
        )
        resource_id: str | None = None
        resource_limit: int = Field(
            c.Performance.MAX_BATCH_SIZE,
            gt=c.ZERO,
            description="Maximum number of resources to retrieve or process in this request.",
            title="Resource Limit",
            examples=[100, 500],
        )
        action: str = Field(
            default=c.Cqrs.Action.GET,
            description="Requested operation to perform on the target resource type.",
            title="Action",
            examples=["get", "create", "update", "delete"],
        )
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

        resource: str = Field(description="Resource identifier")
        user: str = Field(description="User identifier")
        action: str = Field(description="Requested action")
        allowed: bool = Field(description="Whether access is allowed")
        permissions: list[str] = Field(
            default_factory=list, description="Granted permissions"
        )
        denied_permissions: list[str] = Field(
            default_factory=list, description="Denied permissions"
        )
        context: FlextModelsService.ServiceContext | None = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.context is None:
                self.context = FlextModelsService.ServiceContext()
            return self

    class OperationExecutionRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str = Field(
            max_length=c.Performance.MAX_OPERATION_NAME_LENGTH,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Operation name",
        )
        operation_callable: Callable[
            [t.ContainerValue], p.ResultLike[t.ContainerValue]
        ] = Field(description="Callable operation returning result")
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
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.arguments is None:
                self.arguments = FlextModelsService.ServiceParameters()
            if self.keyword_arguments is None:
                self.keyword_arguments = FlextModelsService.ServiceParameters()
            if self.retry_config is None:
                self.retry_config = FlextModelsService.RetryConfiguration()
            return self
