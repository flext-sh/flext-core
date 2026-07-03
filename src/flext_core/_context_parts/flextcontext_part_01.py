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

from typing import Annotated, ClassVar, Self

from flext_core import c, m, p, r, t, u


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
        if key not in self.metadata.attributes:
            return r[t.JsonPayload].fail(f"Metadata key '{key}' not found")
        raw_value: t.JsonValue = self.metadata.attributes[key]
        return r[t.JsonPayload].ok(u.normalize_to_container(raw_value))

    def apply_metadata(self, key: str, value: t.JsonValue) -> None:
        """Set a metadata value by key."""
        updated_attributes = dict(self.metadata.attributes)
        updated_attributes[key] = value
        self.metadata = self.metadata.model_copy(
            update={
                "attributes": t.json_mapping_adapter().validate_python(
                    updated_attributes,
                ),
            },
        )

    def remove(self, key: str) -> None:
        """Remove a key from this context's scope."""
        self.data.root.pop(key, None)

    def clear(self) -> None:
        """Clear all stored keys from this context's scope."""
        self.data.root.clear()

    def merge(
        self,
        other: p.Context | t.MappingKV[str, t.JsonPayload],
    ) -> Self:
        """Merge another context or mapping into this context's scope."""
        if isinstance(other, p.Context):
            self.data.root.update(other.items())
        else:
            self.data.update(other)
        return self

    def clone(self) -> Self:
        """Create an independent copy of this context scope."""
        return self.__class__(
            data=self.data.model_copy(deep=True),
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
    def create(cls, **initial_data: t.JsonPayload) -> p.Context:
        """Factory: build a context instance seeded with initial scope values."""
        context = cls()
        for key, value in initial_data.items():
            _ = context.set(key, value)
        return context

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

    @staticmethod
    def resolve_correlation_id() -> str | None:
        """Get current correlation ID from process context."""
        v = u.CORRELATION_ID.get()
        return v if isinstance(v, str) else None


__all__: list[str] = ["FlextContext"]
