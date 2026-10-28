"""Operational context facade over model-owned state and context utilities.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self

from flext_core import FlextExceptions, FlextResult, FlextUtilities, c, m, p, t
from flext_core._utilities import FlextUtilitiesPydantic


class FlextContext(
    m.ContextState,
    FlextUtilities,
):
    """Scoped store, metadata and shared structlog context-variable operations."""

    @classmethod
    def create(cls, **initial_data: t.JsonPayload) -> p.Context:
        """Create a context seeded with the supplied scope values.

        Returns:
            The initialized context.

        """
        context = cls()
        for key, value in initial_data.items():
            _ = context.set(key, value)
        return context

    @classmethod
    def resolve_container(cls) -> p.Container:
        """Return the configured container, failing if none was configured.

        Raises:
            RuntimeError: If no container has been configured.

        """
        if cls._container_state.container is None:
            raise RuntimeError(c.ERR_RUNTIME_CONTAINER_NOT_INITIALIZED)
        return cls._container_state.container

    @classmethod
    def configure_container(cls, container: p.Container) -> None:
        """Bind the container used by the service namespace."""
        cls._container_state = cls._container_state.model_copy(
            update={"container": container},
        )

    @staticmethod
    def fetch_service(service_name: str) -> p.Result[p.RegisterableService]:
        """Resolve a named service through the configured container.

        Returns:
            The resolved service or its failure.

        """
        return FlextContext.resolve_container().resolve(service_name)

    @staticmethod
    def register_service(
        service_name: str,
        service: p.RegisterableService,
    ) -> p.Result[bool]:
        """Register a service, preserving the validation failure boundary.

        Returns:
            Registration success or validation failure.

        """
        try:
            _ = FlextContext.resolve_container().bind(service_name, service)
        except FlextExceptions.ValidationError as exc:
            return FlextResult[bool].fail_op("register service", exc)
        return FlextResult[bool].ok(value=True)

    def set(self, key: str, value: t.JsonPayload) -> p.Result[bool]:
        """Store a value in this context's scope.

        Returns:
            A successful storage result.

        """
        self.data.root[key] = value
        return FlextResult[bool].ok(value=True)

    def get(self, key: str) -> p.Result[t.JsonPayload]:
        """Return a stored value or an explicit missing/null failure."""
        if key not in self.data.root:
            return FlextResult[t.JsonPayload].fail(f"Key '{key}' not found in context")
        value = self.data.root[key]
        if value is None:
            return FlextResult[t.JsonPayload].fail(f"Key '{key}' has no value")
        return FlextResult[t.JsonPayload].ok(value)

    def has(self, key: str) -> bool:
        """Whether the scope contains the key, including null-valued entries.

        Returns:
            Whether the key is present.

        """
        return key in self.data.root

    def keys(self) -> t.StrSequence:
        """Return stored keys in insertion order."""
        return tuple(self.data.root)

    def values(self) -> list[t.JsonPayload]:
        """Return stored values in insertion order."""
        return list(self.data.root.values())

    def items(self) -> list[tuple[str, t.JsonPayload]]:
        """Return stored key-value pairs in insertion order."""
        return list(self.data.root.items())

    def resolve_metadata(self, key: str) -> p.Result[t.JsonPayload]:
        """Return a metadata attribute or an explicit missing/null failure."""
        if key not in self.metadata.attributes:
            return FlextResult[t.JsonPayload].fail(f"Metadata key '{key}' not found")
        value = self.metadata.attributes[key]
        if value is None:
            return FlextResult[t.JsonPayload].fail(f"Metadata key '{key}' has no value")
        return FlextResult[t.JsonPayload].ok(value)

    def apply_metadata(self, key: str, value: t.JsonValue) -> None:
        """Replace metadata with a snapshot containing the supplied attribute."""
        attributes: t.MutableMappingKV[str, t.JsonValue] = dict(
            self.metadata.attributes
        )
        attributes[key] = value
        snapshot = FlextUtilitiesPydantic.normalize_to_json_mapping(attributes)
        self.metadata = self.metadata.model_copy(update={"attributes": snapshot})

    def remove(self, key: str) -> None:
        """Remove a scope key if present."""
        self.data.root.pop(key, None)

    def clear(self) -> None:
        """Clear the instance scope without changing process context variables."""
        self.data.root.clear()

    def merge(self, other: p.Context | t.MappingKV[str, t.JsonPayload]) -> Self:
        """Merge a context or mapping into this instance and return it.

        Returns:
            This context with merged values.

        """
        self.data.root.update(other.items() if isinstance(other, p.Context) else other)
        return self

    def clone(self) -> Self:
        """Deep-copy scope data and copy the immutable metadata snapshot.

        Returns:
            An independent context copy.

        """
        return self.__class__(
            data=self.data.model_copy(deep=True),
            metadata=self.metadata.model_copy(),
        )

    def export(self, *, as_dict: bool = True) -> t.MappingKV[str, t.JsonPayload] | Self:
        """Export a scope mapping or return this context unchanged.

        Returns:
            A scope snapshot or this context.

        """
        return dict(self.data.root) if as_dict else self


__all__: t.StrSequence = ("FlextContext",)
