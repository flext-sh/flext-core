"""Example 05 mixins models."""

from __future__ import annotations

from typing import override

from pydantic import BaseModel, ConfigDict

from flext_core import t


class Ex05HandlerBad(BaseModel):
    """Intentionally incomplete handler protocol stub."""

    model_config = ConfigDict(frozen=False)


class Ex05GoodProcessor(BaseModel):
    """Processor that satisfies protocol checks."""

    model_config = ConfigDict(frozen=False)

    def process(self) -> bool:
        """Return successful processing state."""
        return True

    @classmethod
    @override
    def validate(cls, value: t.ContainerValue) -> Ex05GoodProcessor:
        """Validate processor payload."""
        return cls.model_validate(value)

    def _protocol_name(self) -> str:
        return "HasModelDump"


class Ex05BadProcessor(BaseModel):
    """Processor that intentionally fails protocol checks."""

    model_config = ConfigDict(frozen=False)

    def _protocol_name(self) -> str:
        return "HasModelDump"
