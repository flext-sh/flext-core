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

from flext_core import FlextModelFoundation, FlextRuntime, c, p, t


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

        config: Annotated[
            p.Settings,
            Field(description="Service configuration settings for runtime behavior."),
        ]
        context: Annotated[
            p.Context,
            Field(
                description="Execution context carrying correlation and tracing metadata."
            ),
        ]
        container: Annotated[
            p.Container,
            Field(description="Dependency injection container for service resolution."),
        ]

    class TraceContext(FlextModelFoundation.FrozenStrictModel):
        """Trace context for distributed tracing."""

        trace_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Distributed trace identifier shared across related service calls.",
                title="Trace Id",
                examples=["c8f2d73e-9870-4cba-b873-5b4a3f7b95f4"],
            ),
        ] = Field(default_factory=lambda: str(uuid.uuid4()))
        span_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Span identifier for the current service operation within a trace.",
                title="Span Id",
                examples=["9fd8d2fd-a4bc-4b15-9e8a-47f6c7dd6a11"],
            ),
        ] = Field(default_factory=lambda: str(uuid.uuid4()))
        parent_span_id: Annotated[
            t.NonEmptyStr | None,
            Field(
                default=None,
                description="Parent span identifier linking this span to its caller in the trace.",
            ),
        ] = None

    class ServiceRetryConfiguration(
        FlextModelFoundation.FrozenStrictModel,
        FlextModelFoundation.RetryConfigurationMixin,
    ):
        """Retry configuration for operations."""

        exponential_base: Annotated[
            t.BackoffMultiplier,
            Field(
                default=c.DEFAULT_BACKOFF_MULTIPLIER,
                description="Exponential backoff base used to calculate retry delay growth.",
                title="Exponential Base",
                examples=[2.0],
            ),
        ] = c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_timeout: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to retry the operation when a timeout occurs.",
            ),
        ] = True

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
            t.NonEmptyStr,
            Field(description="Service name"),
        ]
        method_name: Annotated[
            t.NonEmptyStr,
            Field(description="Method to execute"),
        ]
        parameters: Annotated[
            FlextModelsService.ServiceParameters | None,
            Field(
                default=None,
                description="Dynamic parameters passed to the service method.",
            ),
        ] = None
        context: Annotated[
            FlextModelsService.TraceContext | None,
            Field(
                default=None,
                description="Trace context for distributed tracing of this execution.",
            ),
        ] = None
        timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                le=c.MAX_TIMEOUT_SECONDS,
                description="Timeout from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        execution: Annotated[
            bool,
            Field(
                default=False,
                description="Whether to actually execute the service method or only validate.",
            ),
        ] = False
        enable_validation: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to run input validation before executing the request.",
            ),
        ] = True

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
            t.NonEmptyStr,
            Field(
                description="Operation name executed as part of the batch request.",
                title="Operation Name",
                examples=["create_user", "sync_records"],
            ),
        ]
        parameters: Annotated[
            FlextModelsService.ServiceParameters | None,
            Field(
                default=None,
                description="Dynamic parameters for this batch operation.",
            ),
        ] = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.parameters is None:
                self.parameters = FlextModelsService.ServiceParameters()
            return self

    class DomainServiceBatchRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: Annotated[
            t.NonEmptyStr,
            Field(description="Target service name for the batch request."),
        ]
        operations: Annotated[
            Sequence[FlextModelsService.BatchOperation],
            Field(
                min_length=c.DEFAULT_RETRY_DELAY_SECONDS,
                max_length=c.DEFAULT_SIZE,
                description="Ordered batch operations to execute for the target service.",
                title="Batch Operations",
                examples=[
                    [{"operation_name": "validate"}, {"operation_name": "persist"}],
                ],
            ),
        ] = Field(default_factory=lambda: list[FlextModelsService.BatchOperation]())
        parallel_execution: Annotated[
            bool,
            Field(
                default=False,
                description="Whether to execute batch operations in parallel.",
            ),
        ] = False
        stop_on_error: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to stop the batch on the first operation failure.",
            ),
        ] = True
        batch_size: Annotated[
            int,
            Field(
                default=c.MAX_ITEMS,
                description="Batch size from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.MAX_ITEMS
        timeout_per_operation: Annotated[
            float,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Timeout per operation from FlextSettings",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS

    class DomainServiceMetricsRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: Annotated[
            t.NonEmptyStr,
            Field(description="Target service name for metrics collection."),
        ]
        metric_types: t.StrSequence = Field(
            default_factory=lambda: [*c.DEFAULT_METRIC_CATEGORIES],
            description="Types of metrics to collect",
        )
        time_range_seconds: Annotated[
            t.PositiveInt,
            Field(
                default=c.MAX_TIMEOUT_SECONDS,
                description="Time window in seconds over which to aggregate metrics.",
            ),
        ] = c.MAX_TIMEOUT_SECONDS
        aggregation: Annotated[
            str,
            Field(
                default=c.Aggregation.AVG,
                description="Aggregation strategy applied when summarizing metric values.",
                title="Aggregation",
                examples=["avg", "sum", "max"],
            ),
        ] = c.Aggregation.AVG
        group_by: Annotated[
            t.StrSequence,
            Field(
                description="Metric dimensions used to group the resulting metric series.",
                title="Group By",
                examples=[["service_name", "handler_mode"]],
            ),
        ] = Field(default_factory=list)
        filters: Annotated[
            FlextModelsService.ServiceFilters | None,
            Field(
                default=None,
                description="Optional filter criteria to narrow the metrics query.",
            ),
        ] = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.filters is None:
                self.filters = FlextModelsService.ServiceFilters()
            return self

    class DomainServiceResourceRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: Annotated[
            t.NonEmptyStr,
            Field(
                default=c.DEFAULT_SERVICE_NAME,
                description="Target service name owning the resource.",
            ),
        ] = c.DEFAULT_SERVICE_NAME
        resource_type: Annotated[
            str,
            Field(
                default=c.DEFAULT_RESOURCE_TYPE,
                pattern=c.PATTERN_IDENTIFIER,
                description="Logical resource type targeted by the request, validated as an identifier.",
                title="Resource Type",
                examples=["user", "invoice", "job"],
            ),
        ] = c.DEFAULT_RESOURCE_TYPE
        resource_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Identifier of the specific resource to operate on.",
            ),
        ] = None
        resource_limit: Annotated[
            t.PositiveInt,
            Field(
                default=c.MAX_ITEMS,
                description="Maximum number of resources to retrieve or process in this request.",
                title="Resource Limit",
                examples=[100, 500],
            ),
        ] = c.MAX_ITEMS
        action: Annotated[
            str,
            Field(
                default=c.Action.GET,
                description="Requested operation to perform on the target resource type.",
                title="Action",
                examples=["get", "create", "update", "delete"],
            ),
        ] = c.Action.GET
        data: Annotated[
            FlextModelsService.ServiceData | None,
            Field(
                default=None,
                description="Payload data for create or update operations.",
            ),
        ] = None
        filters: Annotated[
            FlextModelsService.ServiceFilters | None,
            Field(
                default=None,
                description="Optional filter criteria to narrow resource selection.",
            ),
        ] = None

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

        resource: Annotated[t.NonEmptyStr, Field(description="Resource identifier")]
        user: Annotated[t.NonEmptyStr, Field(description="User identifier")]
        action: Annotated[t.NonEmptyStr, Field(description="Requested action")]
        allowed: Annotated[bool, Field(description="Whether access is allowed")]
        permissions: Annotated[
            t.StrSequence,
            Field(description="Granted permissions"),
        ] = Field(default_factory=list)
        denied_permissions: Annotated[
            t.StrSequence,
            Field(description="Denied permissions"),
        ] = Field(default_factory=list)
        context: Annotated[
            FlextModelsService.ServiceContext | None,
            Field(
                default=None,
                description="Additional execution context for the ACL evaluation.",
            ),
        ] = None

        @model_validator(mode="after")
        def apply_defaults(self) -> Self:
            """Apply default values for optional nested classes."""
            if self.context is None:
                self.context = FlextModelsService.ServiceContext()
            return self

    class OperationExecutionRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: Annotated[
            t.NonEmptyStr,
            Field(
                max_length=c.HTTP_STATUS_MIN,
                description="Operation name",
            ),
        ]
        operation_callable: Annotated[
            Callable[
                [t.ValueOrModel],
                p.Result[t.ValueOrModel],
            ],
            Field(description="Callable operation returning result"),
        ]
        arguments: Annotated[
            FlextModelsService.ServiceParameters | None,
            Field(
                default=None,
                description="Positional arguments passed to the operation callable.",
            ),
        ] = None
        keyword_arguments: Annotated[
            FlextModelsService.ServiceParameters | None,
            Field(
                default=None,
                description="Keyword arguments passed to the operation callable.",
            ),
        ] = None
        timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                le=c.MAX_TIMEOUT_SECONDS,
                description="Timeout from FlextSettings (Config has priority over Constants)",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        retry_config: Annotated[
            FlextModelsService.ServiceRetryConfiguration | None,
            Field(
                default=None,
                description="Retry configuration for the operation execution.",
            ),
        ] = None

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

        config_type: Annotated[
            type[BaseSettings] | None,
            Field(
                default=None,
                description="Settings class used to load runtime configuration.",
            ),
        ] = None
        config_overrides: Annotated[
            t.ScalarMapping | None,
            Field(
                default=None,
                description="Key-value overrides applied on top of the loaded configuration.",
            ),
        ] = None
        context: Annotated[
            p.Context | None,
            Field(
                default=None,
                description="Pre-built execution context to inject into the runtime.",
            ),
        ] = None
        subproject: Annotated[
            str | None,
            Field(
                default=None,
                description="Subproject name used to scope configuration and wiring.",
            ),
        ] = None
        services: Annotated[
            Mapping[str, t.RegisterableService] | None,
            Field(
                default=None,
                description="Named services to register in the dependency container.",
            ),
        ] = None
        factories: Annotated[
            Mapping[str, t.FactoryCallable] | None,
            Field(
                default=None,
                description="Named factory callables to register in the dependency container.",
            ),
        ] = None
        resources: Annotated[
            Mapping[str, t.ResourceCallable] | None,
            Field(
                default=None,
                description="Named lifecycle resources to register in the dependency container.",
            ),
        ] = None
        container_overrides: Annotated[
            t.ScalarMapping | None,
            Field(
                default=None,
                description="Provider overrides applied to the dependency container.",
            ),
        ] = None
        wire_modules: Annotated[
            Sequence[ModuleType | str] | None,
            Field(
                default=None,
                description="Modules to wire for dependency-injector resolution.",
            ),
        ] = None
        wire_packages: Annotated[
            t.StrSequence | None,
            Field(
                default=None,
                description="Package names to consider for dependency wiring.",
            ),
        ] = None
        wire_classes: Annotated[
            Sequence[type] | None,
            Field(
                default=None,
                description="Classes whose modules are wired for dependency resolution.",
            ),
        ] = None

    class DependencyContainerCreationOptions(FlextModelFoundation.ArbitraryTypesModel):
        """Options used to create and populate dependency container instances."""

        config: Annotated[
            t.ConfigMap | None,
            Field(
                default=None,
                title="Configuration",
                description="Optional configuration mapping bound to dependency container providers.",
            ),
        ] = None
        services: Annotated[
            Mapping[str, t.RegisterableService] | None,
            Field(
                default=None,
                title="Services",
                description="Object providers registered before optional wiring.",
            ),
        ] = None
        factories: Annotated[
            Mapping[str, t.FactoryCallable] | None,
            Field(
                default=None,
                title="Factories",
                description="Factory providers registered with singleton/factory semantics.",
            ),
        ] = None
        resources: Annotated[
            Mapping[str, t.ResourceCallable] | None,
            Field(
                default=None,
                title="Resources",
                description="Lifecycle resource providers registered before wiring.",
            ),
        ] = None
        wire_modules: Annotated[
            Sequence[ModuleType] | None,
            Field(
                default=None,
                title="Wire Modules",
                description="Modules wired for dependency-injector @inject resolution.",
            ),
        ] = None
        wire_packages: Annotated[
            t.StrSequence | None,
            Field(
                default=None,
                title="Wire Packages",
                description="Package names considered for dependency wiring.",
            ),
        ] = None
        wire_classes: Annotated[
            Sequence[type] | None,
            Field(
                default=None,
                title="Wire Classes",
                description="Classes whose modules are wired for dependency resolution.",
            ),
        ] = None
        factory_cache: Annotated[
            bool,
            Field(
                default=True,
                title="Factory Cache",
                description="Whether registered factories use singleton caching semantics.",
            ),
        ] = True


FlextRuntime.DependencyIntegration.ContainerCreationOptionsModel = (
    FlextModelsService.DependencyContainerCreationOptions
)


__all__ = ["FlextModelsService"]
