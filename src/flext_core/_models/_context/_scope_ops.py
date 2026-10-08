"""Flext context scope operations mixin.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Self

from flext_core import m, p, r, t, u


class FlextContextScopeOps(m.ManagedModel):
    """Instance-scope operations shared by the ``FlextContext`` model.

    Scope store operations: set/get/has/keys/values/items, metadata
    helpers, and clone/merge/export transforms.
    """

    data: Annotated[
        m.ConfigMap,
        m.Field(description="Scoped key-value payload for this context instance."),
    ] = m.Field(default_factory=lambda: m.ConfigMap(root={}))

    metadata: Annotated[
        m.Metadata,
        m.Field(description="Correlation and service metadata snapshot."),
    ] = m.Field(default_factory=m.Metadata)

    def set(self, key: str, value: t.JsonPayload) -> p.Result[bool]:
        """Store a value in this context's scope.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        self.data.update({key: value})
        return r[bool].ok(value=True)

    def get(self, key: str) -> p.Result[t.JsonPayload]:
        """Retrieve a value from this context's scope.

        Returns:
            The resulting ``p.Result[t.JsonPayload]``.

        """
        if key not in self.data.root:
            return r[t.JsonPayload].fail(f"Key '{key}' not found in context")
        value = self.data.root[key]
        if value is None:
            return r[t.JsonPayload].fail(f"Key '{key}' has no value")
        return r[t.JsonPayload].ok(value)

    def has(self, key: str) -> bool:
        """Check if a key exists in this context's scope.

        Returns:
            The resulting ``bool``.

        """
        return key in self.data.root

    def keys(self) -> t.StrSequence:
        """Return all stored keys.

        Returns:
            All stored keys.

        """
        return tuple(self.data.root.keys())

    def values(self) -> list[t.JsonPayload]:
        """Return all stored values.

        Returns:
            All stored values.

        """
        return list(self.data.root.values())

    def items(self) -> list[tuple[str, t.JsonPayload]]:
        """Return all key-value pairs.

        Returns:
            All key-value pairs.

        """
        return list(self.data.root.items())

    def resolve_metadata(self, key: str) -> p.Result[t.JsonPayload]:
        """Get a metadata value by key.

        Returns:
            The resulting ``p.Result[t.JsonPayload]``.

        """
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

    def merge(self, other: p.Context | t.MappingKV[str, t.JsonPayload]) -> Self:
        """Merge another context or mapping into this context's scope.

        Args:
            other: The context or mapping whose scope joins this one.

        Returns:
            The resulting ``Self``.

        """
        if isinstance(other, p.Context):
            self.data.root.update(other.items())
        else:
            self.data.update(other)
        return self

    def clone(self) -> Self:
        """Create an independent copy of this context scope.

        Returns:
            The resulting ``Self``.

        """
        return self.__class__(
            data=self.data.model_copy(deep=True),
            metadata=self.metadata.model_copy(),
        )

    def export(self, *, as_dict: bool = True) -> t.MappingKV[str, t.JsonPayload] | Self:
        """Export scope contents. Returns dict when as_dict=True (default).

        Args:
            as_dict: Return a plain dict when true, else ``self``.

        Returns:
            The resulting ``t.MappingKV[str, t.JsonPayload] | Self``.

        """
        if as_dict:
            return dict(self.data.root)
        return self


__all__: t.StrSequence = ("FlextContextScopeOps",)
