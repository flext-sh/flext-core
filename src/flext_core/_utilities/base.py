"""Utilities base facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter as PydanticTypeAdapter

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase


class FlextUtilitiesBase:
    """Bootstrap-safe constructors inherited by the public utilities facade.

    Adapter construction preserves Pydantic's signature without importing
    models or runtime. Composed consumers use ``u.type_adapter``; lower
    bootstrap owners consume this same constructor before ``u`` exists.
    """

    # Runtime/model bootstrap consumers cannot depend on the composed facade.
    type_adapter = PydanticTypeAdapter

    class StaticExport[T]:
        """Expose a native value without binding or narrowing callable overloads."""

        __slots__ = ("_value",)

        def __init__(self, value: T) -> None:
            """Retain the original value and its complete inferred type."""
            self._value = value

        def __get__(
            self,
            instance: FlextProtocolsBase.AttributeProbe | None,
            owner: type | None = None,
        ) -> T:
            """Return the identical native value for class and instance access.

            Returns:
                The unchanged original value.

            """
            return self._value

    @staticmethod
    def create_instance[T](class_type: type[T]) -> T:
        """Allocate through the native constructor without invoking initialization.

        Returns:
            The instance allocated by Python for the requested class.

        """
        return object.__new__(class_type)


__all__: list[str] = ["FlextUtilitiesBase"]
