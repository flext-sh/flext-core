"""Base runtime helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from flext_core import c
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from flext_core._protocols.loggings import FlextProtocolsLogging
    from flext_core._typings.services import FlextTypesServices


# mro-i6nq.8: Keep the runtime base free of unused provider passthroughs.
class FlextRuntimeBase:
    """Foundational runtime helpers shared by higher runtime namespaces."""

    Metadata: ClassVar[type[FlextProtocolsLogging.Metadata] | None] = None

    @classmethod
    def _require_metadata_model(cls) -> type[FlextProtocolsLogging.Metadata]:
        """Return the bound metadata model class or raise a runtime contract error.

        Returns:
            The bound metadata model class or raise a runtime contract error.

        Raises:
            RuntimeError: If ``metadata_cls is None``.

        """
        metadata_cls = cls.Metadata
        if metadata_cls is None:
            msg = c.ERR_RUNTIME_METADATA_MODEL_NOT_BOUND
            raise RuntimeError(msg)
        return metadata_cls

    @staticmethod
    def create_instance[T](class_type: type[T]) -> T:
        """Create an instance through ``object.__new__`` with type validation.

        Returns:
            The resulting ``T``.

        Raises:
            TypeError: If object.__new__ did not return instance of.

        """
        instance = object.__new__(class_type)
        if not isinstance(instance, class_type):
            msg = f"object.__new__ did not return instance of {class_type.__name__}"
            raise TypeError(msg)
        return instance

    @staticmethod
    def ensure_utc_datetime(value: datetime | None) -> datetime | None:
        """Attach UTC timezone to naive datetimes while preserving None.

        Returns:
            The resulting ``datetime | None``.

        """
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @staticmethod
    def resolve_effective_log_level(
        *,
        trace: bool,
        debug: bool,
        log_level: c.LogLevel,
    ) -> c.LogLevel:
        """Resolve log level: DEBUG if trace, INFO if debug, else log_level.

        Returns:
            The resulting ``c.LogLevel``.

        """
        if trace:
            return c.LogLevel.DEBUG
        if debug:
            return c.LogLevel.INFO
        return log_level

    @staticmethod
    def normalize_alnum(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase the result.

        Returns:
            The resulting ``str``.

        """
        return "".join(ch for ch in text.lower() if ch.isalnum())

    @staticmethod
    def to_scalar(item: FlextTypesServices.GuardInput | None) -> FlextTypingBase.Scalar:
        """Coerce any runtime value to ``t.Scalar``.

        Returns:
            The resulting ``tb.Scalar``.

        """
        if item is None:
            return ""
        return item if isinstance(item, c.SCALAR_TYPES) else str(item)


__all__: list[str] = ["FlextRuntimeBase"]
