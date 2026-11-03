# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/collection_models
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextProtocolsCollectionModels:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class TransformSpec(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        normalize: bool
        strip_none: bool
        strip_empty: bool
        map_keys: Mapping[str, str] | None
        filter_keys: set[str] | None
        exclude_keys: set[str] | None

    @runtime_checkable
    class GuardCheckSpec(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def eq(self) -> FlextTypesPydantic.JsonValue | None:
            """Require the value to equal this value."""
            ...

        @property
        def ne(self) -> FlextTypesPydantic.JsonValue | None:
            """Require the value to differ from this value."""
            ...

        gt: float | str | None
        gte: float | str | None
        lt: float | str | None
        lte: float | str | None
        is_: type | None
        not_: type | None

        @property
        def in_(self) -> FlextTypingBase.JsonList | None:
            """Require the value to be present in this sequence."""
            ...

        @property
        def not_in(self) -> FlextTypingBase.JsonList | None:
            """Require the value to not be present in this sequence."""
            ...

        none: bool | None
        empty: bool | None
        contains: str | None
        starts: str | None
        ends: str | None


__all__: list[str] = [
    "FlextProtocolsCollectionModels",
]
