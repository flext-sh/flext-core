"""Example 05 mixins models."""

from __future__ import annotations

from enum import StrEnum
from typing import override

from pydantic import ConfigDict, Field, field_validator

from flext_core import m


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
    def normalize_status(cls, value: object) -> Ex05StatusEnum:
        if isinstance(value, Ex05StatusEnum):
            return value
        if isinstance(value, str):
            return Ex05StatusEnum(value)
        msg = "invalid status"
        raise TypeError(msg)


class Ex05HandlerBad(m.Value):
    model_config = ConfigDict(frozen=False)


class Ex05HandlerLike(m.Value):
    model_config = ConfigDict(frozen=False)
    data: m.ConfigMap = Field(default_factory=lambda: m.ConfigMap(root={}))


class Ex05GoodProcessor(m.Value):
    model_config = ConfigDict(frozen=False)

    def process(self) -> bool:
        return True

    @classmethod
    @override
    def validate(cls, value: object) -> Ex05GoodProcessor:
        return cls(value)


class Ex05BadProcessor(m.Value):
    model_config = ConfigDict(frozen=False)
