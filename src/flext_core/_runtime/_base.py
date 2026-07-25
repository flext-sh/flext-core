"""Base runtime helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

from dependency_injector import providers

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.logging import FlextConstantsLogging as cl
from flext_core._typings.base import FlextTypingBase as tb

if TYPE_CHECKING:
    from flext_core._protocols.logging import FlextProtocolsLogging as pl
    from flext_core._typings.services import FlextTypesServices as ts


class FlextRuntimeBase:
    """Foundational runtime helpers shared by higher runtime namespaces."""

    Metadata: ClassVar[type[pl.Metadata] | None] = None

    @classmethod
    def _require_metadata_model(cls) -> type[pl.Metadata]:
        """Return the bound metadata model class or raise a runtime contract error."""
        metadata_cls = cls.Metadata
        if metadata_cls is None:
            msg = ce.ERR_RUNTIME_METADATA_MODEL_NOT_BOUND
            raise RuntimeError(msg)
        return metadata_cls

    @staticmethod
    def create_instance[T](class_type: type[T]) -> T:
        """Create an instance through ``object.__new__`` with type validation."""
        instance = object.__new__(class_type)
        if not isinstance(instance, class_type):
            msg = f"object.__new__ did not return instance of {class_type.__name__}"
            raise TypeError(msg)
        return instance

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        if not isinstance(providers, ModuleType):
            msg = "dependency_injector.providers is not a module"
            raise TypeError(msg)
        return providers

    @staticmethod
    def resolve_effective_log_level(
        *,
        trace: bool,
        debug: bool,
        log_level: cl.LogLevel,
    ) -> cl.LogLevel:
        """Resolve log level: DEBUG if trace, INFO if debug, else log_level."""
        if trace:
            return cl.LogLevel.DEBUG
        if debug:
            return cl.LogLevel.INFO
        return log_level

    @staticmethod
    def normalize_alnum(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase the result."""
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def to_scalar(item: ts.GuardInput | None) -> tb.Scalar:
        """Coerce any runtime value to ``t.Scalar``."""
        if item is None:
            return ""
        return item if isinstance(item, tb.SCALAR_TYPES) else str(item)


__all__: list[str] = ["FlextRuntimeBase"]
