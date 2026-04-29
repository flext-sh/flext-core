"""FlextContext — scoped key-value store + contextvar facade.

Pure Pydantic v2 model (m.ManagedModel). Zero utility-chain inheritance.
Context-variable I/O (correlation IDs, service metadata) exclusively via u.*.
Auto-injected transparently by FlextService via __init_subclass__.

Per AGENTS.md §0.7: zero nested classes; all methods flat on the facade.
Per AGENTS.md §3.1: single concern — context data model + contextvar ops.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Self

from flext_core import (
    FlextConstants as c,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
    FlextUtilities as u,
)


class FlextContext(m.ManagedModel):
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

    data: Annotated[
        m.ConfigMap,
        m.Field(description="Scoped key-value payload for this context instance."),
    ] = m.Field(default_factory=lambda: m.ConfigMap(root={}))

    metadata: Annotated[
        m.Metadata,
        m.Field(description="Correlation and service metadata snapshot."),
    ] = m.Field(default_factory=m.Metadata)

    _container_state: ClassVar[m.ContextContainerState] = m.ContextContainerState()

    # --- Scope store (instance methods) ---

    def set(self, key: str, value: t.JsonPayload) -> p.Result[bool]:
        """Store a value in this context's scope."""
        self.data.update({key: value})
        return r[bool].ok(True)

    def get(self, key: str) -> p.Result[t.JsonPayload]:
        """Retrieve a value from this context's scope."""
        k = key
        if k not in self.data.root:
            return r[t.JsonPayload].fail(f"Key '{key}' not found in context")
        v = self.data.root[k]
        if v is None:
            return r[t.JsonPayload].fail(f"Key '{key}' has no value")
        return r[t.JsonPayload].ok(v)

    def has(self, key: str) -> bool:
        """Check if a key exists in this context's scope."""
        return key in self.data.root

    def keys(self) -> t.StrSequence:
        """Return all stored keys."""
        return tuple(self.data.root.keys())

    def values(self) -> list[t.JsonPayload]:
        """Return all stored values."""
        return list(self.data.root.values())

    def items(self) -> list[tuple[str, t.JsonPayload]]:
        """Return all key-value pairs."""
        return list(self.data.root.items())

    def resolve_metadata(self, key: str) -> p.Result[t.JsonPayload]:
        """Get a metadata value by key."""
        v = self.metadata.get(key) if hasattr(self.metadata, "get") else None
        if v is None:
            return r[t.JsonPayload].fail(f"Metadata key '{key}' not found")
        return r[t.JsonPayload].ok(v)

    def apply_metadata(self, key: str, value: t.JsonValue) -> None:
        """Set a metadata value by key."""
        if hasattr(self.metadata, "update"):
            self.metadata.update({key: value})

    def remove(self, key: str) -> None:
        """Remove a key from this context's scope."""
        self.data.root.pop(key, None)

    def clear(self) -> None:
        """Clear all stored keys from this context's scope."""
        self.data.root.clear()

    def merge(
        self,
        other: FlextContext | Mapping[str, t.JsonPayload],
    ) -> Self:
        """Merge another context or mapping into this context's scope."""
        if isinstance(other, FlextContext):
            self.data.update(other.data.root)
        else:
            self.data.update(dict(other.items()))
        return self

    def clone(self) -> Self:
        """Create an independent copy of this context scope."""
        return self.__class__(
            data=m.ConfigMap(root=dict(self.data.root)),
            metadata=self.metadata.model_copy(),
        )

    def export(
        self,
        *,
        as_dict: bool = True,
    ) -> dict[str, t.JsonPayload] | Self:
        """Export scope contents. Returns dict when as_dict=True (default)."""
        if as_dict:
            return dict(self.data.root)
        return self

    @classmethod
    def create(cls, **_: t.JsonPayload) -> p.Context:
        """Factory: build a default context instance."""
        return cls()

    # --- Container management ---

    @classmethod
    def resolve_container(cls) -> p.Container:
        """Get the global DI container instance."""
        if cls._container_state.container is None:
            msg = c.ERR_RUNTIME_CONTAINER_NOT_INITIALIZED
            raise RuntimeError(msg)
        return cls._container_state.container

    @classmethod
    def configure_container(cls, container: p.Container) -> None:
        """Register the global DI container instance."""
        cls._container_state = cls._container_state.with_container(container)

    @staticmethod
    def fetch_service(service_name: str) -> p.Result[t.RegisterableService]:
        """Resolve a named service from the global container."""
        return FlextContext.resolve_container().resolve(service_name)

    @staticmethod
    def register_service(
        service_name: str,
        service: t.RegisterableService,
    ) -> p.Result[bool]:
        """Register a named service in the global container."""
        container = FlextContext.resolve_container()
        _ = container.bind(service_name, service)
        return (
            r[bool].ok(True)
            if container.has(service_name)
            else r[bool].fail(f"Service '{service_name}' was not registered")
        )

    # --- Correlation ID ops (contextvar via u.*) ---

    @staticmethod
    def resolve_correlation_id() -> str | None:
        """Get current correlation ID from process context."""
        v = u.CORRELATION_ID.get()
        return v if isinstance(v, str) else None

    @staticmethod
    @contextmanager
    def new_correlation(
        correlation_id: str | None = None,
        parent_id: str | None = None,
    ) -> Generator[str]:
        """Scope a correlation ID, restoring the previous one on exit."""
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
        """Return current correlation ID, generating one if absent."""
        current = u.CORRELATION_ID.get()
        if isinstance(current, str) and current:
            return current
        new_id: str = u.generate("correlation")
        _ = u.CORRELATION_ID.set(new_id)
        return new_id

    # --- Service context ops (contextvar via u.*) ---

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

    # --- Operation ops (contextvar via u.*) ---

    @staticmethod
    def resolve_operation_name() -> str | None:
        """Get current operation name from process context."""
        v = u.OPERATION_NAME.get()
        return str(v) if v is not None else None

    @staticmethod
    def apply_operation_name(operation_name: str) -> None:
        """Set operation name in process context."""
        _ = u.OPERATION_NAME.set(operation_name)

    @staticmethod
    @contextmanager
    def timed_operation(
        operation_name: str | None = None,
    ) -> Generator[m.ConfigMap]:
        """Scope a timed operation with performance metadata."""
        start_time = u.generate_datetime_utc()
        start_perf = time.perf_counter()
        payload = t.json_mapping_adapter().validate_python({
            str(c.MetadataKey.START_TIME): start_time.isoformat(),
            str(c.ContextKey.OPERATION_NAME): operation_name or "",
        })
        op_meta: m.ConfigMap = m.ConfigMap(root=dict(payload))
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

    # --- Contextvar snapshot / clear ---

    @staticmethod
    def export_full_context() -> Mapping[str, t.Scalar]:
        """Export all active contextvar values as a flat mapping."""
        result: dict[str, t.Scalar] = {}
        if (v := u.CORRELATION_ID.get()) is not None:
            result[c.ContextKey.CORRELATION_ID] = str(v)
        if (v := u.PARENT_CORRELATION_ID.get()) is not None:
            result[c.ContextKey.PARENT_CORRELATION_ID] = str(v)
        if (v := u.SERVICE_NAME.get()) is not None:
            result[c.ContextKey.SERVICE_NAME] = str(v)
        if (v := u.SERVICE_VERSION.get()) is not None:
            result[c.ContextKey.SERVICE_VERSION] = str(v)
        if (v := u.USER_ID.get()) is not None:
            result[c.ContextKey.USER_ID] = str(v)
        if (v := u.REQUEST_ID.get()) is not None:
            result[c.ContextKey.REQUEST_ID] = str(v)
        if (v := u.OPERATION_NAME.get()) is not None:
            result[c.ContextKey.OPERATION_NAME] = str(v)
        if (v := u.OPERATION_START_TIME.get()) is not None:
            result[c.ContextKey.OPERATION_START_TIME] = (
                v.isoformat() if isinstance(v, datetime) else str(v)
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


__all__: t.StrSequence = ["FlextContext"]
