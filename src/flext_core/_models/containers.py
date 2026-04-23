"""Container RootModel wrappers for the FLEXT runtime.

Pydantic v2 ``RootModel`` subclasses with explicit mapping/sequence APIs.
Uses composition over multi-inheritance: the ``root`` field holds the
underlying container and the class exposes an explicit dict-like or
list-like surface — no ``MutableMapping`` / ``Sequence`` ABC inheritance,
which would collide with ``BaseModel.__iter__``'s ``TupleGenerator``
signature (pydantic yields ``(field_name, value)`` tuples whereas
``Mapping`` yields keys).

Consumers that need a plain ``dict`` from validation should call
``.root`` on the validated instance (see consumers in ``_exceptions``,
``result``, ``_utilities.collection``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    ItemsView,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Annotated

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.services import FlextTypesServices
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up


class FlextModelsContainers:
    """Pydantic RootModel container namespace."""

    class ValidatorCallable(mp.RootModel[FlextTypesServices.ValidatorCallable]):
        """Callable validator container rooted in a scalar-or-model transform."""

        root: Annotated[
            FlextTypesServices.ValidatorCallable,
            up.Field(
                title="Validator Callable",
                description="Callable that validates or transforms one scalar/model input value.",
                examples=["identity_validator"],
            ),
        ]

        def __call__(
            self,
            value: FlextTypesServices.ScalarOrModel,
        ) -> FlextTypesServices.ScalarOrModel:
            return self.root(value)

    class Dict(mp.RootModel[dict[str, FlextTypesServices.JsonPayload]]):
        """Runtime dictionary container with explicit mapping API."""

        root: Annotated[
            dict[str, FlextTypesServices.JsonPayload],
            up.Field(description="Validated runtime key-value mapping."),
        ]

        def __getitem__(self, key: str) -> FlextTypesServices.JsonPayload:
            return self.root[key]

        def __setitem__(
            self,
            key: str,
            value: FlextTypesServices.JsonPayload,
        ) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __contains__(self, key: object) -> bool:
            return key in self.root

        def __len__(self) -> int:
            return len(self.root)

        def __bool__(self) -> bool:
            return bool(self.root)

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def values(self) -> ValuesView[FlextTypesServices.JsonPayload]:
            return self.root.values()

        def items(self) -> ItemsView[str, FlextTypesServices.JsonPayload]:
            return self.root.items()

        def get(
            self,
            key: str,
            default: FlextTypesServices.JsonPayload | None = None,
        ) -> FlextTypesServices.JsonPayload | None:
            return self.root.get(key, default)

        def update(
            self,
            other: Mapping[str, FlextTypesServices.JsonPayload],
        ) -> None:
            self.root.update(other)

        def clear(self) -> None:
            self.root.clear()

        def pop(
            self,
            key: str,
            *args: FlextTypesServices.JsonPayload,
        ) -> FlextTypesServices.JsonPayload:
            return self.root.pop(key, *args)

        def popitem(self) -> tuple[str, FlextTypesServices.JsonPayload]:
            return self.root.popitem()

        def setdefault(
            self,
            key: str,
            default: FlextTypesServices.JsonPayload,
        ) -> FlextTypesServices.JsonPayload:
            return self.root.setdefault(key, default)

    class ConfigMap(mp.RootModel[dict[str, FlextTypesServices.JsonPayload]]):
        """Runtime configuration mapping with explicit mapping API."""

        root: Annotated[
            dict[str, FlextTypesServices.JsonPayload],
            up.Field(description="Validated runtime configuration mapping."),
        ]

        def __getitem__(self, key: str) -> FlextTypesServices.JsonPayload:
            return self.root[key]

        def __setitem__(
            self,
            key: str,
            value: FlextTypesServices.JsonPayload,
        ) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __contains__(self, key: object) -> bool:
            return key in self.root

        def __len__(self) -> int:
            return len(self.root)

        def __bool__(self) -> bool:
            return bool(self.root)

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def values(self) -> ValuesView[FlextTypesServices.JsonPayload]:
            return self.root.values()

        def items(self) -> ItemsView[str, FlextTypesServices.JsonPayload]:
            return self.root.items()

        def get(
            self,
            key: str,
            default: FlextTypesServices.JsonPayload | None = None,
        ) -> FlextTypesServices.JsonPayload | None:
            return self.root.get(key, default)

        def update(
            self,
            other: Mapping[str, FlextTypesServices.JsonPayload],
        ) -> None:
            self.root.update(other)

        def clear(self) -> None:
            self.root.clear()

        def pop(
            self,
            key: str,
            *args: FlextTypesServices.JsonPayload,
        ) -> FlextTypesServices.JsonPayload:
            return self.root.pop(key, *args)

        def popitem(self) -> tuple[str, FlextTypesServices.JsonPayload]:
            return self.root.popitem()

        def setdefault(
            self,
            key: str,
            default: FlextTypesServices.JsonPayload,
        ) -> FlextTypesServices.JsonPayload:
            return self.root.setdefault(key, default)

    class ObjectList(mp.RootModel[list[FlextTypesServices.JsonPayload]]):
        """Runtime list container rooted in validated values.

        Exposes ``root`` as the canonical accessor.
        """

        root: Annotated[
            list[FlextTypesServices.JsonPayload],
            up.Field(description="Validated runtime sequence."),
        ]

        def __len__(self) -> int:
            return len(self.root)

        def __bool__(self) -> bool:
            return bool(self.root)


__all__: list[str] = ["FlextModelsContainers"]
