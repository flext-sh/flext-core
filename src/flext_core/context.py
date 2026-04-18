"""Context propagation utilities for dispatcher-coordinated workloads.

FlextContext tracks correlation metadata, request data, and timing information
through the dispatcher pipeline and into handlers, ensuring structured logs and
metrics remain consistent across threads and async boundaries.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Self, overload

from pydantic import PrivateAttr

from flext_core import c, m, p, r, t, u
from flext_core._utilities.context_tracing import FlextUtilitiesContextTracing


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

    _logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    initial_data: Annotated[
        m.ContextData | t.ConfigMap | None,
        m.Field(description="Initial data for context scopes."),
    ] = None

    _state: m.ContextRuntimeState = PrivateAttr(
        default_factory=lambda: m.ContextRuntimeState.create_default(),
    )

    def __init__(self, **data: t.ValueOrModel) -> None:
        """Initialize FlextContext with optional initial data."""
        super().__init__(**data)
        context_data = m.ContextData()
        if self.initial_data is not None:
            if isinstance(self.initial_data, m.ContextData):
                context_data = self.initial_data
            else:
                context_data = m.ContextData(
                    data=t.Dict(
                        root=dict(self.initial_data),
                    ),
                )
        metadata_model = (
            context_data.metadata.model_copy()
            if isinstance(context_data.metadata, m.Metadata)
            else m.Metadata()
        )
        self._state = m.ContextRuntimeState.create_default(metadata=metadata_model)
        if context_data.data:
            self._update_contextvar(
                c.ContextScope.GLOBAL,
                t.ConfigMap(root=context_data.data.root),
            )

    @overload
    @classmethod
    def create(cls, initial_data: t.ConfigMap | None = None) -> Self: ...

    @overload
    @classmethod
    def create(
        cls,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: t.ConfigMap | None = None,
    ) -> Self: ...

    @classmethod
    def create(
        cls,
        initial_data: t.ConfigMap | None = None,
        *,
        operation_id: str | None = None,
        user_id: str | None = None,
        metadata: t.ConfigMap | None = None,
        auto_correlation_id: bool = True,
    ) -> Self:
        """Factory method to create a new FlextContext instance."""
        if operation_id is not None or user_id is not None or metadata is not None:
            initial_data_dict: t.ConfigMap = t.ConfigMap(root={})
            if operation_id is not None:
                initial_data_dict[c.ContextKey.OPERATION_ID] = operation_id
            elif auto_correlation_id:
                initial_data_dict[c.ContextKey.OPERATION_ID] = u.generate(
                    "correlation",
                )
            if user_id is not None:
                initial_data_dict[c.ContextKey.USER_ID] = user_id
            if metadata is not None:
                initial_data_dict.update(dict(metadata))
            return cls(
                initial_data=m.ContextData(data=t.Dict(root=initial_data_dict.root)),
            )
        data_map = (
            initial_data
            if isinstance(initial_data, t.ConfigMap)
            else t.ConfigMap(initial_data)
            if initial_data is not None
            else t.ConfigMap(root={})
        )
        if auto_correlation_id and c.ContextKey.OPERATION_ID not in data_map:
            initial_data_dict_new: t.ConfigMap = data_map.model_copy()
            initial_data_dict_new[c.ContextKey.OPERATION_ID] = u.generate(
                "correlation",
            )
            return cls(
                initial_data=m.ContextData(
                    data=t.Dict(root=initial_data_dict_new.root),
                ),
            )
        return cls(initial_data=m.ContextData(data=t.Dict(root=data_map.root)))

    def clone(self) -> Self:
        """Create a clone of this context."""
        cloned: Self = self.__class__.model_validate({
            "initial_data": self.initial_data
        })
        for scope_name, ctx_var in self.iter_scope_vars().items():
            scope_dict = self._narrow_contextvar_to_configuration_dict(ctx_var.get())
            if scope_dict:
                cloned.set(
                    t.ConfigMap(root=dict(scope_dict)),
                    scope=scope_name,
                )
        cloned._state = cloned._state.model_copy(
            update={
                "metadata": self._state.metadata.model_copy(),
                "statistics": self._state.statistics.model_copy(),
            },
        )
        return cloned

    _container_state: ClassVar[m.ContextContainerState] = m.ContextContainerState()

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
        ) -> Generator[t.ConfigMap]:
            """Create timed operation context with performance tracking."""
            start_time = u.generate_datetime_utc()
            start_perf = time.perf_counter()
            operation_metadata: t.ConfigMap = t.ConfigMap(
                root={
                    c.MetadataKey.START_TIME: start_time.isoformat(),
                    c.ContextKey.OPERATION_NAME: operation_name,
                },
            )
            start_token = FlextContext.Variables.OperationStartTime.set(start_time)
            metadata_token = FlextContext.Variables.OperationMetadata.set(
                operation_metadata,
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
        def export_full_context() -> t.RecursiveContainerMapping:
            """Get current context as dictionary."""
            context_vars = FlextContext.Variables
            operation_metadata_raw = context_vars.Performance.OPERATION_METADATA.get()
            operation_metadata_value: t.RecursiveContainer = ""
            if operation_metadata_raw is not None:
                operation_metadata_value = FlextContext._to_normalized(
                    u.normalize_to_container(
                        u.normalize_to_metadata(operation_metadata_raw),
                    ),
                )
            raw_ctx: Mapping[str, t.ValueOrModel | t.ConfigMap | None] = {
                c.ContextKey.CORRELATION_ID: context_vars.Correlation.CORRELATION_ID.get(),
                c.ContextKey.PARENT_CORRELATION_ID: context_vars.Correlation.PARENT_CORRELATION_ID.get(),
                c.ContextKey.SERVICE_NAME: context_vars.Service.SERVICE_NAME.get(),
                c.ContextKey.SERVICE_VERSION: context_vars.Service.SERVICE_VERSION.get(),
                c.ContextKey.USER_ID: context_vars.Request.USER_ID.get(),
                c.ContextKey.OPERATION_NAME: context_vars.Performance.OPERATION_NAME.get(),
                c.ContextKey.REQUEST_ID: context_vars.Request.REQUEST_ID.get(),
                c.ContextKey.OPERATION_START_TIME: st.isoformat()
                if isinstance(
                    (st := context_vars.Performance.OPERATION_START_TIME.get()),
                    datetime,
                )
                else None,
                c.ContextKey.OPERATION_METADATA: operation_metadata_value,
            }
            return {
                k: FlextContext._to_normalized(v)
                for k, v in raw_ctx.items()
                if v is not None
            }

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
