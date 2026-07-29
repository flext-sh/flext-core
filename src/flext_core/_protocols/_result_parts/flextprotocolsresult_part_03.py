"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols._result_parts.flextprotocolsresult_part_02 import (
    FlextProtocolsResult as FlextProtocolsResultPart02,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextProtocolsResult(FlextProtocolsResultPart02):
    @runtime_checkable
    class ResultMonadic[T](Protocol):
        """Monadic chaining operations on result.

        Provides map, flat_map, fold, and lash for composing
        result-returning operations in a railway-oriented style.
        All return types use the protocol (never the concrete carrier).
        T variance is automatically inferred as invariant (consumed via callbacks).
        """

        def flat_map[U](
            self, func: Callable[[T], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]:
            """Chain operations that return structural FLEXT results."""
            ...

        def fold[U](
            self, on_failure: Callable[[str], U], on_success: Callable[[T], U]
        ) -> U:
            """Reduce result into a single value."""
            ...

        def lash(
            self, func: Callable[[str], FlextProtocolsResult.Result[T]]
        ) -> FlextProtocolsResult.Result[T]:
            """Recover from failure using another structural result."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocolsResult.Result[U]:
            """Transform the success value."""
            ...

    @runtime_checkable
    class ResultTappable[T](Protocol):
        """Side-effect and error transform operations.

        Provides tap, tap_error, flow_through, map_error, and map_or
        for observing or transforming result state without consuming it.
        T variance is automatically inferred as invariant (consumed via callbacks).
        """

        def flow_through(
            self, *funcs: Callable[[T], FlextProtocolsResult.Result[T]]
        ) -> FlextProtocolsResult.Result[T]:
            """Apply multiple Result-returning steps in sequence."""
            ...

        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextProtocolsResult.Result[T]:
            """Transform the failure message."""
            ...

        @overload
        def map_or(self, default: None, func: None = None) -> T | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> T | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...
        def map_or[U](self, default: U, func: Callable[[T], U] | None = None) -> U | T:
            """Map success value or return default."""
            ...

        def tap(self, func: Callable[[T], None]) -> FlextProtocolsResult.Result[T]:
            """Apply a side effect to the success value."""
            ...

        def tap_error(self, func: Callable[[str], None]) -> Self:
            """Apply a side effect to the failure value."""
            ...

    @runtime_checkable
    class ResultRecoverable[T](Protocol):
        """Fallback and predicate-based result operations."""

        def filter(
            self, predicate: Callable[[T], bool]
        ) -> FlextProtocolsResult.Result[T]:
            """Keep the value only when the predicate passes."""
            ...

        def recover[U](
            self, func: Callable[[str], U]
        ) -> FlextProtocolsResult.Result[T | U]:
            """Recover a failure into a success value."""
            ...

    @runtime_checkable
    class ResultConvertible[T](Protocol):
        """Conversion helpers from raw result payloads to validated types."""

        def to_model[U: mp.BaseModel](
            self, model: type[U]
        ) -> FlextProtocolsResult.Result[U]:
            """Convert the success payload into a validated model."""
            ...


__all__: list[str] = ["FlextProtocolsResult"]
