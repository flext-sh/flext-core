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

from pydantic import BaseModel, Field, RootModel

from flext_core._typings.base import FlextTypingBase


class FlextTypingContainers:
    """Container type system for FLEXT."""

    class RootDictModel[DictValueT](RootModel[MutableMapping[str, DictValueT]]):
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

    class ObjectList(RootModel[Sequence[FlextTypingBase.Container]]):
        """Ordered list of strongly-typed container values for batch operations."""

        root: Annotated[
            Sequence[FlextTypingBase.Container],
            Field(
                default_factory=list,
                title="Object List",
                description=(
                    "Ordered container values for batch operations "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[["item-1", 2, True]],
            ),
        ]

    class Dict(RootDictModel[FlextTypingBase.NormalizedValue | BaseModel]):
        """Validated dict payload for requests, responses, and data transfer.

        Type-safe MutableMapping[str, NormalizedValue | BaseModel] with full dict protocol.
        """

        root: Annotated[
            MutableMapping[str, FlextTypingBase.NormalizedValue | BaseModel],
            Field(
                default_factory=dict,
                title="Dictionary Payload",
                description=(
                    "Dictionary payload storing strict container values "
                    "(scalar, BaseModel, or Path)."
                ),
                examples=[{"request_id": "req-123", "retry_count": 3, "dry_run": True}],
            ),
        ]

        @override
        def get(
            self,
            key: str,
            default: FlextTypingBase.NormalizedValue | BaseModel | None = None,
        ) -> FlextTypingBase.NormalizedValue | BaseModel | None:
            value = self.root.get(key, default)
            if isinstance(value, Mapping) and not isinstance(value, BaseModel):
                return dict(value.items())
            return value

    class ConfigMap(RootDictModel[FlextTypingBase.NormalizedValue | BaseModel]):
        """Configuration container for settings and environment parameters.

        Semantically distinct Dict for configuration (not data).
        """

        root: Annotated[
            MutableMapping[str, FlextTypingBase.NormalizedValue | BaseModel],
            Field(
                default_factory=dict,
                title="Configuration Map",
                description="Configuration entries keyed by normalized setting names.",
                examples=[
                    {"timeout_seconds": 30, "environment": "dev", "debug": False},
                ],
            ),
        ]
