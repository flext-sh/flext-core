"""Container models for the FLEXT type system.

Pydantic RootModel-based container classes for dictionaries, lists, and validators.
These are the concrete implementations exposed through FlextModels (``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, ItemsView, KeysView, Mapping, Sequence, ValuesView
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

type GeneralValueType = (
    str
    | int
    | float
    | bool
    | datetime
    | BaseModel
    | Path
    | Sequence[GeneralValueType]
    | Mapping[str, GeneralValueType]
    | None
)
type _ContainerValue = GeneralValueType
type _ScalarML = str | int | float | bool | datetime | None
type _FactoryRegistrationCallable = Callable[[], _ScalarML | Sequence[_ScalarML]]
type _ResourceCallable = Callable[[], _ContainerValue]


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``m.ConfigMap``, ``m.Dict``, etc.
    """

    class ObjectList(RootModel[list[_ContainerValue]]):
        """Sequence of container values for batch operations."""

        root: list[_ContainerValue]

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

    class Dict(_RootDictModel[_ContainerValue]):
        """Generic dictionary container. Use ``m.Dict``."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ConfigMap(_RootDictModel[_ContainerValue]):
        """Configuration map container. Use ``m.ConfigMap``."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ServiceMap(_RootDictModel[_ContainerValue]):
        """Service registry map container. Use ``m.ServiceMap``."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ErrorMap(_RootDictModel[int | str | dict[str, int]]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]] = Field(default_factory=dict)

    class FactoryMap(_RootDictModel[_FactoryRegistrationCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, _FactoryRegistrationCallable] = Field(default_factory=dict)

    class ResourceMap(_RootDictModel[_ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, _ResourceCallable] = Field(default_factory=dict)

    class ValidatorCallable(
        RootModel[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]

        def __call__(self, value: _ScalarML | BaseModel) -> _ScalarML | BaseModel:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]]
    ):
        """Shared API for validator map containers."""

        def items(
            self,
        ) -> ItemsView[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]:
            """Get validator items."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.items()

        def values(
            self,
        ) -> ValuesView[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],]:
            """Get validator values."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True, extra="forbid"
        )
        results: list[_ScalarML] = []
        errors: list[tuple[int, str]] = []
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)


__all__ = ["FlextModelsContainers"]
