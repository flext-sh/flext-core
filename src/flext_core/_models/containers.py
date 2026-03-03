"""Container models for the FLEXT type system.

Pydantic RootModel-based container classes for dictionaries, lists, and validators.
These are the concrete implementations exposed through FlextModels (``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, ItemsView, KeysView, Mapping, ValuesView
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

from flext_core.typings import FlextTypes as t


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``m.ConfigMap``, ``m.Dict``, etc.
    """

    class ObjectList(RootModel[list[t.ContainerValue]]):
        """Sequence of container values for batch operations."""

        root: list[t.ContainerValue]

    @typing.runtime_checkable
    class _RootDictProtocol[RootValueT](typing.Protocol):
        root: dict[str, RootValueT]

    class _RootDictModel[DictValueT](RootModel[dict[str, DictValueT]]):
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

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.get(key, default)

        def items(self) -> ItemsView[str, DictValueT]:
            return self.root.items()

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def values(self) -> ValuesView[DictValueT]:
            return self.root.values()

        def update(self, other: Mapping[str, DictValueT]) -> None:
            self.root.update(other)

        def clear(self) -> None:
            self.root.clear()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            return self.root.setdefault(key, default)

    class Dict(_RootDictModel[t.ContainerValue]):
        """Generic dictionary container. Use ``m.Dict``."""

        root: dict[str, t.ContainerValue] = Field(default_factory=dict)

    class ConfigMap(_RootDictModel[t.ContainerValue]):
        """Configuration map container. Use ``m.ConfigMap``."""

        root: dict[str, t.ContainerValue] = Field(default_factory=dict)

    class ServiceMap(_RootDictModel[t.ContainerValue]):
        """Service registry map container. Use ``m.ServiceMap``."""

        root: dict[str, t.ContainerValue] = Field(default_factory=dict)

    class ErrorMap(_RootDictModel[int | str | dict[str, int]]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]] = Field(default_factory=dict)

    class FactoryMap(_RootDictModel[t.FactoryCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, t.FactoryCallable] = Field(default_factory=dict)

    class ResourceMap(_RootDictModel[t.ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, t.ResourceCallable] = Field(default_factory=dict)

    class ValidatorCallable(
        RootModel[Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]]
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]

        def __call__(
            self, value: t.Scalar | BaseModel | None
        ) -> t.Scalar | BaseModel | None:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[
            dict[
                str,
                Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None],
            ]
        ]
    ):
        """Shared API for validator map containers."""

        def items(
            self,
        ) -> ItemsView[
            str, Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]
        ]:
            """Get validator items."""
            validated: dict[
                str,
                Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None],
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.items()

        def values(
            self,
        ) -> ValuesView[
            Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None],
        ]:
            """Get validator values."""
            validated: dict[
                str,
                Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None],
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: dict[
            str, Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]
        ] = Field(default_factory=dict)

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: dict[
            str, Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]
        ] = Field(default_factory=dict)

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: dict[
            str, Callable[[t.Scalar | BaseModel | None], t.Scalar | BaseModel | None]
        ] = Field(default_factory=dict)

    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True, extra="forbid"
        )
        results: list[t.Scalar | None] = []
        errors: list[tuple[int, str]] = []
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)


__all__ = ["FlextModelsContainers"]
