"""Example 05 mixins models."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, override

from flext_core import m, u


class ExamplesFlextCoreModelsEx05:
    """Example 05 mixins model namespace."""

    @unique
    class StatusEnum(StrEnum):
        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    class UserModel(m.Value):
        name: str = u.Field(description="User display name")
        status: ExamplesFlextCoreModelsEx05.Ex05StatusEnum = u.Field(
            description="User account status"
        )
        age: int = u.Field(description="User age in years")

        @u.field_validator("status", mode="before")
        @classmethod
        def normalize_status(
            cls,
            value: str | ExamplesFlextCoreModelsEx05.Ex05StatusEnum,
        ) -> ExamplesFlextCoreModelsEx05.Ex05StatusEnum:
            if isinstance(value, ExamplesFlextCoreModelsEx05.Ex05StatusEnum):
                return value
            return ExamplesFlextCoreModelsEx05.Ex05StatusEnum(value)

    class HandlerBad(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    class HandlerLike(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
        data: m.ConfigMap = u.Field(default_factory=lambda: m.ConfigMap(root={}))

    class GoodProcessor(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        def process(self) -> bool:
            return True

        @classmethod
        @override
        def validate(
            cls,
            value: m.ConfigMap,
        ) -> ExamplesFlextCoreModelsEx05.Ex05GoodProcessor:
            return cls.model_validate(value)

    class BadProcessor(m.Value):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
