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
    class TransformSpec(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def normalize(self) -> bool:
            """Normalize the mapping values before the pipeline."""
            ...

        @property
        def strip_none(self) -> bool:
            """Drop entries whose value is None."""
            ...

        @property
        def strip_empty(self) -> bool:
            """Drop entries whose value is empty."""
            ...

        @property
        def map_keys(self) -> Mapping[str, str] | None:
            """Optional rename map applied to keys."""
            ...

        @property
        def filter_keys(self) -> set[str] | None:
            """Keep only these keys when provided."""
            ...

        @property
        def exclude_keys(self) -> set[str] | None:
            """Drop these keys when provided."""
            ...

    @runtime_checkable
    class GuardCheckSpec(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def eq(self) -> FlextTypesPydantic.JsonValue | None:
            """Require the value to equal this value."""
            ...

        @property
        def ne(self) -> FlextTypesPydantic.JsonValue | None:
            """Require the value to differ from this value."""
            ...

        @property
        def gt(self) -> float | str | None:
            """Require value to be greater than this (sortable: numeric or string)."""
            ...

        @property
        def gte(self) -> float | str | None:
            """Structural member projected from the owning model.

            Require value to be greater than or equal to this (sortable: numeric or
            string).
            """
            ...

        @property
        def lt(self) -> float | str | None:
            """Require value to be less than this (sortable: numeric or string)."""
            ...

        @property
        def lte(self) -> float | str | None:
            """Structural member projected from the owning model.

            Require value to be less than or equal to this (sortable: numeric or
            string).
            """
            ...

        @property
        def is_(self) -> type | None:
            """Require the value to be an instance of this type."""
            ...

        @property
        def not_(self) -> type | None:
            """Require the value to not be an instance of this type."""
            ...

        @property
        def in_(self) -> FlextTypingBase.JsonList | None:
            """Require the value to be present in this sequence."""
            ...

        @property
        def not_in(self) -> FlextTypingBase.JsonList | None:
            """Require the value to not be present in this sequence."""
            ...

        @property
        def none(self) -> bool | None:
            """When True, require None. When False, require non-None."""
            ...

        @property
        def empty(self) -> bool | None:
            """When True, require empty value; when False, require non-empty."""
            ...

        @property
        def contains(self) -> str | None:
            """Require string or iterable value to contain this item."""
            ...

        @property
        def starts(self) -> str | None:
            """Require string value to start with this prefix."""
            ...

        @property
        def ends(self) -> str | None:
            """Require string value to end with this suffix."""
            ...


__all__: list[str] = [
    "FlextProtocolsCollectionModels",
]
