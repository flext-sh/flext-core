"""Type-safe result type for operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._protocols.result import FlextProtocolsResult as prt

from ._result.base import JsonDict
from ._result.behavior import FlextResultBehavior
from ._result.composition import FlextResultComposition
from ._result.construction import FlextResultConstruction
from ._result.transforms import FlextResultTransforms
from ._result.unwrap import FlextResultUnwrap


class _FlextResult[T](
    FlextResultUnwrap[T],
    FlextResultComposition[T],
    FlextResultTransforms[T],
    FlextResultConstruction[T],
    FlextResultBehavior[T],
):
    """Type-safe result with monadic railway-oriented operations."""

    def __init__(
        self,
        error_code: str | None = None,
        error_data: JsonDict | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
        exception: BaseException | None = None,
    ) -> None:
        """Initialize a result with value, error, or exception state."""
        super().__init__(
            error_code=error_code,
            error_data=error_data,
            value=value,
            error=error,
            success=success,
            exception=exception,
        )


if TYPE_CHECKING:
    from typing import override

    from flext_core import p, t

    class FlextResult[T](_FlextResult[T], prt.Result[T]):
        """Type-safe result with monadic railway-oriented operations."""

        @classmethod
        @override
        def ok[V](cls, value: V) -> FlextResult[V]:
            """Create a successful result carrying ``value``."""
            ...

        @classmethod
        @override
        def fail(
            cls,
            error: str | None,
            *,
            error_code: str | None = None,
            error_data: t.JsonMapping | t.ConfigModelInput | None = None,
            exception: BaseException | None = None,
        ) -> FlextResult[T]:
            """Create a failed result with the given error payload."""
            ...

        @classmethod
        @override
        def fail_op(
            cls, operation: str, exc: Exception | str | None = None
        ) -> FlextResult[T]:
            """Create a failed result for a named operation."""
            ...

        @classmethod
        @override
        def from_failure(cls, source: p.FailureLike) -> FlextResult[T]:
            """Rebuild this concrete facade from any failed result-like."""
            ...

        @classmethod
        @override
        def from_result[V](cls, source: p.Result[V]) -> FlextResult[V]:
            """Copy an abstract result into this concrete facade."""
            ...

else:

    class FlextResult[T](_FlextResult[T]):
        """Type-safe result with monadic railway-oriented operations."""


r = FlextResult


__all__: list[str] = ["FlextResult", "r"]
