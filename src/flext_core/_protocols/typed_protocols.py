"""Typed protocols for configuration, logging, and serialization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, Self

from flext_core import c, r
from flext_core._models.typed_containers import ConfigMapping, ConfigValue, LogContext


class SupportsConfigAccess(Protocol):
    """Protocol for objects that provide config access."""

    def get_config(self, key: str) -> r[ConfigValue]: ...
    def set_config(self, key: str, value: ConfigValue) -> r[bool]: ...


class SupportsLogging(Protocol):
    """Protocol for structured logging with typed context."""

    def log(
        self,
        level: c.Settings.LogLevel,
        message: str,
        context: LogContext,
    ) -> None: ...


class SupportsSerialization(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> ConfigMapping: ...

    @classmethod
    def from_dict(cls, data: ConfigMapping) -> Self: ...


class FlextTypedProtocols:
    """Namespace container for typed protocols."""

    SupportsConfigAccess = SupportsConfigAccess
    SupportsLogging = SupportsLogging
    SupportsSerialization = SupportsSerialization


__all__ = [
    "FlextTypedProtocols",
    "SupportsConfigAccess",
    "SupportsLogging",
    "SupportsSerialization",
]
