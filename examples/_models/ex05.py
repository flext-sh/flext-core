"""Example 05 mixins models."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, override

from examples import t
from flext_core import m, u


class ExamplesFlextCoreModelsEx05:
    """Example 05 mixins model namespace."""

    @unique
    class Ex05StatusEnum(StrEnum):
        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    class Ex05UserModel(m.Value):
        name: str
        status: ExamplesFlextCoreModelsEx05.Ex05StatusEnum
        age: int

        @u.field_validator("status", mode="before")
        @classmethod
        def normalize_status(
            cls,
            value: str | ExamplesFlextCoreModelsEx05.Ex05StatusEnum,
        ) -> ExamplesFlextCoreModelsEx05.Ex05StatusEnum:
            if isinstance(value, ExamplesFlextCoreModelsEx05.Ex05StatusEnum):
                return value
            return ExamplesFlextCoreModelsEx05.Ex05StatusEnum(value)

    class Ex05HandlerBad(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    class Ex05HandlerLike(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
        data: t.ConfigMap = u.Field(default_factory=lambda: t.ConfigMap(root={}))

    class Ex05GoodProcessor(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        def process(self) -> bool:
            return True

        @classmethod
        @override
        def validate(
            cls,
            value: t.ConfigMap,
        ) -> ExamplesFlextCoreModelsEx05.Ex05GoodProcessor:
            return cls.model_validate(value)

    class Ex05BadProcessor(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)


# Module-level re-exports for package __init__.py API
Ex05StatusEnum = ExamplesFlextCoreModelsEx05.Ex05StatusEnum
Ex05UserModel = ExamplesFlextCoreModelsEx05.Ex05UserModel
Ex05HandlerBad = ExamplesFlextCoreModelsEx05.Ex05HandlerBad
Ex05HandlerLike = ExamplesFlextCoreModelsEx05.Ex05HandlerLike
Ex05GoodProcessor = ExamplesFlextCoreModelsEx05.Ex05GoodProcessor
Ex05BadProcessor = ExamplesFlextCoreModelsEx05.Ex05BadProcessor
