"""Typed protocols for container-based interfaces."""

from __future__ import annotations

from typing import Protocol, Self

from flext_core import c, r
from flext_core._models.typed_containers import ConfigMapping, ConfigValue, LogContext


class SupportsConfigAccess(Protocol):
    def get_config(self, key: str) -> r[ConfigValue]: ...
    def set_config(self, key: str, value: ConfigValue) -> r[bool]: ...


class SupportsLogging(Protocol):
    def log(
        self,
        level: c.Settings.LogLevel,
        message: str,
        context: LogContext,
    ) -> None: ...


class SupportsSerialization(Protocol):
    def to_dict(self) -> ConfigMapping: ...

    @classmethod
    def from_dict(cls, data: ConfigMapping) -> Self: ...


__all__ = [
    "SupportsConfigAccess",
    "SupportsLogging",
    "SupportsSerialization",
]
