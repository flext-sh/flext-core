"""FlextTypingContainers - Pydantic RootModel container types for payload handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import (
    ItemsView,
    KeysView,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from typing import Annotated, override

from flext_core import FlextModelsPydantic, FlextTypingBase, FlextUtilitiesPydantic


class FlextTypingContainers:
    """Container type system for FLEXT."""

    class RootDictModel[DictValueT](
        FlextModelsPydantic.RootModel[MutableMapping[str, DictValueT]]
    ):
        """Dict-backed RootModel with full dict protocol.

        Wraps typed dict in Pydantic v2 validation, exposes all dict methods
        (__getitem__, __setitem__, keys(), items(), get(), etc.).
        """

        def __getitem__(self, key: str) -> DictValueT:
            return self.root[key]

        def __setitem__(self, key: str, value: DictValueT) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __len__(self) -> int:
            return len(self.root)

        def __contains__(self, key: str) -> bool:
            return key in self.root

        def clear(self) -> None:
            self.root.clear()

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.get(key, default)

        def items(self) -> ItemsView[str, DictValueT]:
            return self.root.items()

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            return self.root.setdefault(key, default)

        def update(self, other: Mapping[str, DictValueT]) -> None:
            self.root.update(other)

        @override
        def __iter__(self) -> typing.Generator[tuple[str, DictValueT]]:
            yield from self.root.items()

        def values(self) -> ValuesView[DictValueT]:
            return self.root.values()

    class ObjectList(
        FlextModelsPydantic.RootModel[Sequence[FlextTypingBase.Container]]
    ):
        """Ordered list of strongly-typed container values for batch operations."""

        root: Annotated[
            Sequence[FlextTypingBase.Container],
            FlextUtilitiesPydantic.Field(
                title="Object List",
                description=(
                    "Ordered container values for batch operations "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[["item-1", 2, True]],
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=list)

    class Dict(
        RootDictModel[
            FlextTypingBase.RecursiveContainer | FlextModelsPydantic.BaseModel
        ]
    ):
        """Validated dict payload for requests, responses, and data transfer.

        Type-safe MutableMapping[str, RecursiveContainer | BaseModel] with full dict protocol.
        """

        root: Annotated[
            MutableMapping[
                str,
                FlextTypingBase.RecursiveContainer | FlextModelsPydantic.BaseModel,
            ],
            FlextUtilitiesPydantic.Field(
                title="Dictionary Payload",
                description=(
                    "Dictionary payload storing strict container values "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[{"request_id": "req-123", "retry_count": 3, "dry_run": True}],
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=dict)

        @override
        def get(
            self,
            key: str,
            default: (
                FlextTypingBase.RecursiveContainer
                | FlextModelsPydantic.BaseModel
                | None
            ) = None,
        ) -> FlextTypingBase.RecursiveContainer | FlextModelsPydantic.BaseModel | None:
            value = self.root.get(key, default)
            if isinstance(value, Mapping) and not isinstance(
                value,
                FlextModelsPydantic.BaseModel,
            ):
                return dict(value)
            return value

    class ConfigMap(
        RootDictModel[
            FlextTypingBase.Container
            | FlextModelsPydantic.BaseModel
            | FlextTypingBase.RecursiveContainerMapping
            | FlextTypingBase.RecursiveContainerList
        ]
    ):
        """Configuration container for settings and environment parameters.

        Semantically distinct Dict for configuration (not data).
        Type allows nested structures and models via recursive container types.
        """
