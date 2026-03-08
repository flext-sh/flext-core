"""Example 05 mixins models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import field_validator

from flext_core import m, t


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
    def normalize_status(cls, value: t.ContainerValue) -> Ex05StatusEnum:
        if isinstance(value, Ex05StatusEnum):
            return value
        if isinstance(value, str):
            return Ex05StatusEnum(value)
        msg = "invalid status"
        raise TypeError(msg)
