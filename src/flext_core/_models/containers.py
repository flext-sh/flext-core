"""Container RootModel wrappers for the FLEXT runtime.

Pydantic v2 ``RootModel`` subclasses with explicit mapping/sequence APIs.
Uses **composition over multi-inheritance**: each container wraps a
``root`` field (validated dict/list) and exposes an explicit key/value
or index API. Multi-inheritance with ``collections.abc`` mixins would
clash with ``BaseModel.__iter__`` (TupleGenerator vs Iterator[K]), so
this module avoids ABC inheritance entirely.

Consumers that need a plain ``dict`` from a validated ``ConfigMap``/
``Dict`` MUST call ``.root`` on the validated instance (see consumers
in ``_exceptions``, ``result``, ``_utilities.collection``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, JsonValue

from flext_core import t
from flext_core._models import FlextModelsPydantic

if TYPE_CHECKING:
    from collections.abc import ItemsView, KeysView, ValuesView


class FlextModelsContainers:
    """Pydantic RootModel container namespace."""

    class ValidatorCallable(
        FlextModelsPydantic.RootModel[t.ValidatorCallable],
    ):
        """Callable validator container rooted in a scalar-or-model transform."""

        root: Annotated[
            t.ValidatorCallable,
            FlextModelsPydantic.Field(
                title="Validator Callable",
                description=(
                    "Callable that validates or transforms one scalar/model input "
                    "value."
                ),
                examples=["identity_validator"],
            ),
        ]

        def __call__(
            self,
            value: t.ScalarOrModel,
        ) -> t.ScalarOrModel:
            return self.root(value)

    class _MappingRootBase(
        FlextModelsPydantic.RootModel[t.MutableMappingKV[str, JsonValue | FlextModelsPydantic.BaseModel]],
    ):
        """MRO base providing the explicit mapping API for dict-rooted RootModels.

        Both ``Dict`` and ``ConfigMap`` differ only in the ``root`` field
        description; the 13 mapping methods are identical. This base
        consolidates the methods so subclasses only declare the typed
        ``root`` field with the appropriate description.
        """

        def __getitem__(self, key: str) -> JsonValue | BaseModel:
            return self.root[key]

        def __setitem__(self, key: str, value: JsonValue | BaseModel) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __contains__(self, key: str) -> bool:
            return key in self.root

        def __len__(self) -> int:
            return len(self.root)

        def __bool__(self) -> bool:
            return bool(self.root)

        def keys(self) -> KeysView[str]:
            """Return a view over the mapping's string keys."""
            return self.root.keys()

        def values(self) -> ValuesView[JsonValue | BaseModel]:
            """Return a view over the stored JSON payloads."""
            return self.root.values()

        def items(self) -> ItemsView[str, JsonValue | BaseModel]:
            """Return a view over the key-payload pairs."""
            return self.root.items()

        def get(
            self,
            key: str,
            default: JsonValue | BaseModel | None = None,
        ) -> JsonValue | BaseModel | None:
            """Return the payload stored under ``key``.

            Args:
                key: The string key to look up.
                default: The value returned when the key is absent.

            Returns:
                The stored payload, or ``default`` when the key is absent.

            """
            return self.root.get(key, default)

        def update(
            self,
            other: t.MappingKV[str, JsonValue | BaseModel],
        ) -> None:
            """Merge ``other`` into the mapping, overwriting existing keys.

            Args:
                other: The key-payload mapping merged into this container.

            """
            self.root.update(other)

        def clear(self) -> None:
            """Remove every key-payload pair from the mapping."""
            self.root.clear()

        def pop(
            self,
            key: str,
            *args: JsonValue | BaseModel,
        ) -> JsonValue | BaseModel:
            """Remove ``key`` and return its payload.

            Args:
                key: The string key to remove.
                *args: Optional single default returned when the key is
                    absent; without it a missing key raises ``KeyError``.

            Returns:
                The removed payload, or the supplied default.

            """
            return self.root.pop(key, *args)

        def popitem(self) -> tuple[str, JsonValue | BaseModel]:
            """Remove the most recently inserted key-payload pair.

            Returns:
                The removed ``(key, payload)`` tuple.

            """
            return self.root.popitem()

        def setdefault(
            self,
            key: str,
            default: JsonValue | BaseModel,
        ) -> JsonValue | BaseModel:
            """Return the payload under ``key``, storing ``default`` if absent.

            Args:
                key: The string key to look up.
                default: The payload stored (and returned) when the key is
                    absent.

            Returns:
                The stored payload after the call.

            """
            return self.root.setdefault(key, default)

    class Dict(_MappingRootBase):
        """Runtime dictionary container with explicit mapping API."""

        root: Annotated[
            t.MutableMappingKV[str, JsonValue | FlextModelsPydantic.BaseModel],
            FlextModelsPydantic.Field(
                description="Validated runtime key-value mapping.",
            ),
        ]

    class ConfigMap(_MappingRootBase):
        """Runtime configuration mapping with explicit mapping API."""

        root: Annotated[
            t.MutableMappingKV[str, JsonValue | FlextModelsPydantic.BaseModel],
            FlextModelsPydantic.Field(
                description="Validated runtime configuration mapping.",
            ),
        ]

    class ObjectList(
        FlextModelsPydantic.RootModel[list[JsonValue | BaseModel]],
    ):
        """Runtime list container rooted in validated values.

        Consumers iterate via ``.root`` for the validated list.
        """

        root: Annotated[
            list[JsonValue | BaseModel],
            FlextModelsPydantic.Field(description="Validated runtime sequence."),
        ]

        def __len__(self) -> int:
            return len(self.root)

        def __bool__(self) -> bool:
            return bool(self.root)


__all__: list[str] = ["FlextModelsContainers"]
