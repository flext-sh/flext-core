"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextvars
import time
from collections.abc import (
    Generator,
    Mapping,
)
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Self

from flext_core import FlextUtilitiesContextTracing, c, m, p, r, t, u


class FlextContext(FlextUtilitiesContextTracing, m.ArbitraryTypesModel):
    """Context manager for correlation, request data, and timing metadata.

    The dispatcher and decorators rely on FlextContext to move correlation IDs,
    service metadata, and timing details through CQRS handlers without mutating
    function signatures.

    Composed via MRO from:
    - FlextUtilitiesContextNormalization — static value normalization
    - FlextUtilitiesContextScope — scope variable access and state helpers
    - FlextUtilitiesContextCrud — get/set/has/remove/clear operations
    - FlextUtilitiesContextLifecycle — create/clone/merge/export
    - FlextUtilitiesContextTracing — Variables, Correlation, Service, Request, etc.
    """

    logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    initial_data: Annotated[
        m.ContextData | t.JsonValue | None,
        m.Field(description="Initial data for context scopes."),
    ] = None

    state: Annotated[
        m.ContextRuntimeState,
        m.Field(description="Runtime state for context scopes and metadata."),
    ] = m.Field(default_factory=lambda: m.ContextRuntimeState.create_default())

    _container_state: ClassVar[m.ContextContainerState] = m.ContextContainerState()

    def __init__(self, **data: t.JsonPayload) -> None:
        """Initialize FlextContext with optional initial data."""
        super().__init__(**data)
        context_data = m.ContextData()
        if self.initial_data is not None:
            if isinstance(self.initial_data, m.ContextData):
                context_data = self.initial_data
            elif isinstance(self.initial_data, dict):
                normalized_data: dict[str, t.Scalar] = {
                    str(key): u.to_scalar(raw_value)
                    for key, raw_value in self.initial_data.items()
                }
                context_data = m.ContextData(
                    data=normalized_data,
                )
        metadata_model = (
            context_data.metadata.model_copy()
            if isinstance(context_data.metadata, m.Metadata)
            else m.Metadata()
        )
        self.state = m.ContextRuntimeState.create_default(metadata=metadata_model)

    def clone(self) -> Self:
        """Create a clone of this context with independent scope storage."""
        cloned: Self = self.__class__.model_validate({
            "initial_data": self.initial_data
        })
        new_vars: dict[str, contextvars.ContextVar[m.ConfigMap | None]] = {}
        for scope_name, ctx_var in self.state.scope_vars.items():
            current = ctx_var.get()
            new_var: contextvars.ContextVar[m.ConfigMap | None] = (
                contextvars.ContextVar(f"{scope_name}_clone", default=None)
            )
            if current is not None:
                _ = new_var.set(current.model_copy())
            new_vars[scope_name] = new_var
        cloned.state = cloned.state.model_copy(
            update={
                "metadata": self.state.metadata.model_copy(),
                "statistics": self.state.statistics.model_copy(),
                "scope_vars": new_vars,
            },
        )
        return cloned

    @classmethod
    def create(cls, **_: t.JsonPayload) -> Self:
        """Factory: build a default context; kwargs reserved for future use."""
        return cls()

    @classmethod
    def resolve_container(cls) -> p.Container:
        """Get global container instance."""
        if cls._container_state.container is None:
            msg = c.ERR_RUNTIME_CONTAINER_NOT_INITIALIZED
            raise RuntimeError(msg)
        return cls._container_state.container

    @classmethod
    def configure_container(cls, container: p.Container) -> None:
        """Set the global container instance."""
        cls._container_state = cls._container_state.with_container(container)

    class Correlation:
        """Distributed tracing and correlation ID management utilities."""

        @staticmethod
        def resolve_correlation_id() -> str | None:
            """Get current correlation ID."""
            correlation_id = FlextContext.Variables.CorrelationId.get()
            return correlation_id if isinstance(correlation_id, str) else None

        @staticmethod
        @contextmanager
        def new_correlation(
            correlation_id: str | None = None,
            parent_id: str | None = None,
        ) -> Generator[str]:
            """Create correlation context scope."""
            if correlation_id is None:
                correlation_id = u.generate("correlation")
            current_correlation = FlextContext.Variables.CorrelationId.get()
            correlation_token = FlextContext.Variables.CorrelationId.set(correlation_id)
            parent_token = None
            if parent_id:
                parent_token = FlextContext.Variables.ParentCorrelationId.set(parent_id)
            elif isinstance(current_correlation, str):
                parent_token = FlextContext.Variables.ParentCorrelationId.set(
                    current_correlation,
                )
            try:
                yield correlation_id
            finally:
                FlextContext.Variables.CorrelationId.reset(correlation_token)
                if parent_token:
                    FlextContext.Variables.ParentCorrelationId.reset(parent_token)

        @staticmethod
        def apply_correlation_id(correlation_id: str | None) -> None:
            """Set correlation ID."""
            _ = FlextContext.Variables.CorrelationId.set(correlation_id)

    class Service:
        """Service identification and lifecycle context management utilities."""

        @staticmethod
        def fetch_service(service_name: str) -> p.Result[t.RegisterableService]:
            """Resolve service from global container using r."""
            container: p.Container = FlextContext.resolve_container()
            return container.resolve(service_name)

        @staticmethod
        def register_service(
            service_name: str,
            service: t.RegisterableService,
        ) -> p.Result[bool]:
            """Register service in global container using r."""
            container = FlextContext.resolve_container()
            _ = container.bind(service_name, service)
            if container.has(service_name):
                return r[bool].ok(True)
            return r[bool].fail_op(
                "register service in context container",
                f"Service '{service_name}' was not registered",
            )

        @staticmethod
        @contextmanager
        def service_context(
            service_name: str,
            version: str | None = None,
        ) -> Generator[None]:
            """Create service context scope."""
            _ = FlextContext.Variables.ServiceName.get()
            _ = FlextContext.Variables.ServiceVersion.get()
            name_token = FlextContext.Variables.ServiceName.set(service_name)
            version_token = None
            if version:
                version_token = FlextContext.Variables.ServiceVersion.set(version)
            try:
                yield
            finally:
                FlextContext.Variables.ServiceName.reset(name_token)
                if version_token:
                    FlextContext.Variables.ServiceVersion.reset(version_token)

    class Request:
        """Request-level context management utilities."""

        @staticmethod
        def resolve_operation_name() -> str | None:
            """Get the current operation name from context."""
            operation_name = FlextContext.Variables.OperationName.get()
            return str(operation_name) if operation_name is not None else None

        @staticmethod
        def apply_operation_name(operation_name: str) -> None:
            """Set operation name in context."""
            _ = FlextContext.Variables.OperationName.set(operation_name)

    class Performance:
        """Performance monitoring and timing context management utilities."""

        @staticmethod
        @contextmanager
        def timed_operation(
            operation_name: str | None = None,
        ) -> Generator[m.ConfigMap]:
            """Create timed operation context with performance tracking."""
            start_time = u.generate_datetime_utc()
            start_perf = time.perf_counter()
            metadata_payload = t.json_mapping_adapter().validate_python(
                {
                    str(c.MetadataKey.START_TIME): start_time.isoformat(),
                    str(c.ContextKey.OPERATION_NAME): operation_name or "",
                },
            )
            operation_metadata: m.ConfigMap = m.ConfigMap(
                root=dict(metadata_payload),
            )
            start_token = FlextContext.Variables.OperationStartTime.set(start_time)
            metadata_token = FlextContext.Variables.OperationMetadata.set(
                metadata_payload,
            )
            operation_token = None
            if operation_name:
                operation_token = FlextContext.Variables.OperationName.set(
                    operation_name,
                )
            try:
                yield operation_metadata
            finally:
                duration = time.perf_counter() - start_perf
                end_time = start_time + timedelta(seconds=duration)
                operation_metadata.update({
                    c.MetadataKey.END_TIME: end_time.isoformat(),
                    c.MetadataKey.DURATION_SECONDS: duration,
                })
                FlextContext.Variables.OperationStartTime.reset(start_token)
                FlextContext.Variables.OperationMetadata.reset(metadata_token)
                if operation_token:
                    FlextContext.Variables.OperationName.reset(operation_token)

    class Serialization:
        """Context serialization and deserialization utilities."""

        @staticmethod
        def export_full_context() -> Mapping[str, t.Scalar]:
            """Export current context as a flat dictionary of scalar values."""
            context_vars = FlextContext.Variables

            # Build context dictionary with direct scalar access
            result: dict[str, t.Scalar] = {}

            # Correlation IDs (strings)
            if (corr_id := context_vars.Correlation.CORRELATION_ID.get()) is not None:
                result[c.ContextKey.CORRELATION_ID] = str(corr_id)
            if (
                parent_corr := context_vars.Correlation.PARENT_CORRELATION_ID.get()
            ) is not None:
                result[c.ContextKey.PARENT_CORRELATION_ID] = str(parent_corr)

            # Service metadata (strings)
            if (svc_name := context_vars.Service.SERVICE_NAME.get()) is not None:
                result[c.ContextKey.SERVICE_NAME] = str(svc_name)
            if (svc_version := context_vars.Service.SERVICE_VERSION.get()) is not None:
                result[c.ContextKey.SERVICE_VERSION] = str(svc_version)

            # Request metadata (strings)
            if (user_id := context_vars.Request.USER_ID.get()) is not None:
                result[c.ContextKey.USER_ID] = str(user_id)
            if (req_id := context_vars.Request.REQUEST_ID.get()) is not None:
                result[c.ContextKey.REQUEST_ID] = str(req_id)

            # Performance metadata
            if (op_name := context_vars.Performance.OPERATION_NAME.get()) is not None:
                result[c.ContextKey.OPERATION_NAME] = str(op_name)

            # Operation start time as ISO string
            if (
                op_start := context_vars.Performance.OPERATION_START_TIME.get()
            ) is not None:
                if isinstance(op_start, datetime):
                    result[c.ContextKey.OPERATION_START_TIME] = op_start.isoformat()
                else:
                    result[c.ContextKey.OPERATION_START_TIME] = str(op_start)

            return result

    class Utilities:
        """Context management utility methods."""

        @staticmethod
        def clear_context() -> None:
            """Clear all context variables."""
            for context_var in [
                FlextContext.Variables.CorrelationId,
                FlextContext.Variables.ParentCorrelationId,
                FlextContext.Variables.ServiceName,
                FlextContext.Variables.ServiceVersion,
                FlextContext.Variables.UserId,
                FlextContext.Variables.RequestId,
                FlextContext.Variables.OperationName,
            ]:
                _ = context_var.set(None)
            _ = FlextContext.Variables.OperationStartTime.set(None)
            _ = FlextContext.Variables.OperationMetadata.set(None)
            _ = FlextContext.Variables.RequestTimestamp.set(None)

        @staticmethod
        def ensure_correlation_id() -> str:
            """Ensure correlation ID exists, creating one if needed."""
            correlation_id_value = FlextContext.Variables.CorrelationId.get()
            if isinstance(correlation_id_value, str) and correlation_id_value:
                return correlation_id_value
            new_correlation_id: str = u.generate("correlation")
            FlextContext.Correlation.apply_correlation_id(new_correlation_id)
            return new_correlation_id


__all__: t.StrSequence = ["FlextContext"]
