"""Flext context module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_models/flext_context
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from flext_core import c, e, m, p, r, t, u
from flext_core._models._context._scope_ops import FlextContextScopeOps

# NOTE (multi-agent): mro-i6nq.12 — Generator is annotation-only; importing it
# under TYPE_CHECKING keeps the public runtime facade graph lazy. The module
# owns its __all__: the context facade re-exports from here, never the reverse.
if TYPE_CHECKING:
    from collections.abc import Generator


class FlextContext(FlextContextScopeOps, m.ManagedModel):
    """Scoped key-value context + correlation/service metadata facade.

    Scope store: `ctx.set(key, value)` / `ctx.get(key)` — instance-level.
    Contextvar ops: `FlextContext.apply_correlation_id(x)` — class-level via u.*.
    Container ops: `FlextContext.resolve_container()` — class-level.
    """

    model_config = m.ConfigDict(
        extra="forbid",
        validate_assignment=False,
        arbitrary_types_allowed=True,
    )

    _container_state: ClassVar[m.ContextContainerState] = m.ContextContainerState()

    @classmethod
    def create(cls, **initial_data: t.JsonPayload) -> p.Context:
        """Build a context instance seeded with initial scope values.

        Returns:
            The resulting ``p.Context``.

        """
        context = cls()
        for key, value in initial_data.items():
            _ = context.set(key, value)
        return context

    @classmethod
    def resolve_container(cls) -> p.Container:
        """Get the global DI container instance.

        Returns:
            The resulting ``p.Container``.

        Raises:
            RuntimeError: If ``cls._container_state.container is None``.

        """
        if cls._container_state.container is None:
            msg = c.ERR_RUNTIME_CONTAINER_NOT_INITIALIZED
            raise RuntimeError(msg)
        return cls._container_state.container

    @classmethod
    def configure_container(cls, container: p.Container) -> None:
        """Register the global DI container instance."""
        cls._container_state = cls._container_state.model_copy(
            update={"container": container},
        )

    @staticmethod
    def fetch_service(service_name: str) -> p.Result[t.RegisterableService]:
        """Resolve a named service from the global container.

        Returns:
            The resulting ``p.Result[t.RegisterableService]``.

        """
        return FlextContext.resolve_container().resolve(service_name)

    @staticmethod
    def register_service(
        service_name: str,
        service: t.RegisterableService,
    ) -> p.Result[bool]:
        """Register a named service in the global container.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        container = FlextContext.resolve_container()
        try:
            _ = container.bind(service_name, service)
        except e.ValidationError as exc:
            return r[bool].fail_op("register service", exc)
        return r[bool].ok(value=True)

    @staticmethod
    def resolve_correlation_id() -> str | None:
        """Get current correlation ID from process context.

        Returns:
            The resulting ``str | None``.

        """
        value = u.CORRELATION_ID.get()
        return value if isinstance(value, str) else None

    @staticmethod
    @contextmanager
    def new_correlation(
        correlation_id: str | None = None,
        parent_id: str | None = None,
    ) -> Generator[str]:
        """Scope a correlation ID, restoring the previous one on exit.

        Yields:
            Each ``str``.

        """
        if correlation_id is None:
            correlation_id = u.generate("correlation")
        current = u.CORRELATION_ID.get()
        corr_token = u.CORRELATION_ID.set(correlation_id)
        parent_token = (
            u.PARENT_CORRELATION_ID.set(parent_id)
            if parent_id
            else (
                u.PARENT_CORRELATION_ID.set(current)
                if isinstance(current, str)
                else None
            )
        )
        try:
            yield correlation_id
        finally:
            u.CORRELATION_ID.reset(corr_token)
            if parent_token:
                u.PARENT_CORRELATION_ID.reset(parent_token)

    @staticmethod
    def apply_correlation_id(correlation_id: str | None) -> None:
        """Set correlation ID in process context."""
        _ = u.CORRELATION_ID.set(correlation_id)

    @staticmethod
    def ensure_correlation_id() -> str:
        """Return current correlation ID, generating one if absent.

        Returns:
            Current correlation ID, generating one if absent.

        """
        current = u.CORRELATION_ID.get()
        if isinstance(current, str) and current:
            return current
        new_id: str = u.generate("correlation")
        _ = u.CORRELATION_ID.set(new_id)
        return new_id

    @staticmethod
    @contextmanager
    def service_context(
        service_name: str,
        version: str | None = None,
    ) -> Generator[None]:
        """Scope service name/version in process context."""
        name_token = u.SERVICE_NAME.set(service_name)
        version_token = u.SERVICE_VERSION.set(version) if version else None
        try:
            yield
        finally:
            u.SERVICE_NAME.reset(name_token)
            if version_token:
                u.SERVICE_VERSION.reset(version_token)

    @staticmethod
    def resolve_operation_name() -> str | None:
        """Get current operation name from process context.

        Returns:
            The resulting ``str | None``.

        """
        value = u.OPERATION_NAME.get()
        return str(value) if value is not None else None

    @staticmethod
    def apply_operation_name(operation_name: str) -> None:
        """Set operation name in process context."""
        _ = u.OPERATION_NAME.set(operation_name)

    @staticmethod
    @contextmanager
    def timed_operation(operation_name: str | None = None) -> Generator[m.ConfigMap]:
        """Scope a timed operation with performance metadata.

        Yields:
            Each ``m.ConfigMap``.

        """
        start_time = u.generate_datetime_utc()
        start_perf = time.perf_counter()
        payload = t.json_mapping_adapter().validate_python({
            str(c.MetadataKey.START_TIME): start_time.isoformat(),
            str(c.ContextKey.OPERATION_NAME): operation_name or "",
        })
        op_meta = m.ConfigMap.model_validate(payload)
        start_token = u.OPERATION_START_TIME.set(start_time)
        meta_token = u.OPERATION_METADATA.set(payload)
        op_token = u.OPERATION_NAME.set(operation_name) if operation_name else None
        try:
            yield op_meta
        finally:
            duration = time.perf_counter() - start_perf
            end_time = start_time + timedelta(seconds=duration)
            op_meta.update({
                c.MetadataKey.END_TIME: end_time.isoformat(),
                c.MetadataKey.DURATION_SECONDS: duration,
            })
            u.OPERATION_START_TIME.reset(start_token)
            u.OPERATION_METADATA.reset(meta_token)
            if op_token:
                u.OPERATION_NAME.reset(op_token)

    @staticmethod
    def export_full_context() -> t.MappingKV[str, t.Scalar]:
        """Export all active contextvar values as a flat mapping.

        Returns:
            The resulting ``t.MappingKV[str, t.Scalar]``.

        """
        result: dict[str, t.Scalar] = {}
        if (value := u.CORRELATION_ID.get()) is not None:
            result[c.ContextKey.CORRELATION_ID] = str(value)
        if (value := u.PARENT_CORRELATION_ID.get()) is not None:
            result[c.ContextKey.PARENT_CORRELATION_ID] = str(value)
        if (value := u.SERVICE_NAME.get()) is not None:
            result[c.ContextKey.SERVICE_NAME] = str(value)
        if (value := u.SERVICE_VERSION.get()) is not None:
            result[c.ContextKey.SERVICE_VERSION] = str(value)
        if (value := u.USER_ID.get()) is not None:
            result[c.ContextKey.USER_ID] = str(value)
        if (value := u.REQUEST_ID.get()) is not None:
            result[c.ContextKey.REQUEST_ID] = str(value)
        if (value := u.OPERATION_NAME.get()) is not None:
            result[c.ContextKey.OPERATION_NAME] = str(value)
        if (value := u.OPERATION_START_TIME.get()) is not None:
            result[c.ContextKey.OPERATION_START_TIME] = (
                value.isoformat() if isinstance(value, datetime) else str(value)
            )
        return result

    @staticmethod
    def clear_context() -> None:
        """Clear all contextvar proxies (process-global scope)."""
        for ctx_var in (
            u.CORRELATION_ID,
            u.PARENT_CORRELATION_ID,
            u.SERVICE_NAME,
            u.SERVICE_VERSION,
            u.USER_ID,
            u.REQUEST_ID,
            u.OPERATION_NAME,
        ):
            _ = ctx_var.set(None)
        _ = u.OPERATION_START_TIME.set(None)
        _ = u.OPERATION_METADATA.set(None)
        _ = u.REQUEST_TIMESTAMP.set(None)


__all__: t.StrSequence = ("FlextContext",)
