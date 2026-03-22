"""Example 05 mixins models."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import override

from pydantic import ConfigDict, Field, field_validator

from flext_core import m, t


@unique
class Ex05StatusEnum(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class Ex05UserModel(m.Value):
    name: str
    status: Ex05StatusEnum
    age: int

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str | Ex05StatusEnum) -> Ex05StatusEnum:
        if isinstance(value, Ex05StatusEnum):
            return value
        if isinstance(value, str):
            return Ex05StatusEnum(value)
        msg = "invalid status"
        raise TypeError(msg)


class Ex05HandlerBad(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)


class Ex05HandlerLike(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)
    data: t.ConfigMap = Field(default_factory=lambda: t.ConfigMap(root={}))


class Ex05GoodProcessor(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

    def process(self) -> bool:
        return True

    @classmethod
    @override
    def validate(cls, value: t.ConfigMap) -> Ex05GoodProcessor:
        return cls.model_validate(value)


class Ex05BadProcessor(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)
