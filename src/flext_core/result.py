"""Type-safe result type for operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._protocols.result import FlextProtocolsResult as prt
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
    from collections.abc import Callable

    from flext_core import p, t

    class FlextResult[T](prt.Result[T]):
        """Type-safe result with monadic railway-oriented operations.

        The instance contract is inherited from ``p.Result``; the stub declares
        only the class-level constructors and combinators the protocol lacks.
        """

        @classmethod
        def ok(cls, value: T) -> p.Result[T]:
            """Create a successful result carrying ``value``."""
            ...

        @classmethod
        def fail(
            cls,
            error: str | None,
            *,
            error_code: str | None = None,
            error_data: t.JsonMapping | t.ConfigModelInput | None = None,
            exception: BaseException | None = None,
        ) -> p.Result[T]:
            """Create a failed result with the given error payload."""
            ...

        @classmethod
        def fail_op(
            cls, operation: str, exc: Exception | str | None = None
        ) -> p.Result[T]:
            """Create a failed result for a named operation."""
            ...

        @classmethod
        def from_failure(cls, source: p.FailureLike) -> p.Result[T]:
            """Rebuild this concrete facade from any failed result-like."""
            ...

        @classmethod
        def from_result[V](cls, source: prt.Result[V]) -> p.Result[V]:
            """Copy an abstract result into this concrete facade."""
            ...

        @classmethod
        def from_validation[ModelT: t.BaseModelType](
            cls, data: object, model: type[ModelT]
        ) -> p.Result[ModelT]:
            """Validate data against a Pydantic model and return a result."""
            ...

        @classmethod
        def successful_result(cls, obj: object) -> bool:
            """Check if object is a successful result."""
            ...

        @classmethod
        def failed_result(cls, obj: object) -> bool:
            """Check if object is a failed result."""
            ...

        @classmethod
        def require_error(cls, source: p.FailureLike) -> str:
            """Return the error of a failed result-like or a loud default."""
            ...

        @classmethod
        def traverse[V, U](
            cls,
            items: t.SequenceOf[V],
            func: Callable[[V], p.Result[U]],
            *,
            fail_fast: bool = True,
        ) -> p.Result[t.SequenceOf[U]]:
            """Collect one result per item, short-circuiting on failure."""
            ...

        @classmethod
        def accumulate_errors[ValueT](
            cls, *results: p.Result[ValueT]
        ) -> p.Result[t.SequenceOf[ValueT]]:
            """Collect every failure payload across results before failing."""
            ...

        @classmethod
        def with_resource[R, U](
            cls,
            factory: Callable[[], R],
            op: Callable[[R], p.Result[U]],
            cleanup: Callable[[R], None] | None = None,
        ) -> p.Result[U]:
            """Run one operation over an owned resource with guaranteed cleanup."""
            ...

        @classmethod
        def create_from_callable[V](
            cls, func: Callable[[], V | None], error_code: str | None = None
        ) -> p.Result[V]:
            """Lift a nullable callable into a result with a loud error code."""
            ...

else:
    FlextResult = _FlextResult


r = FlextResult


__all__: list[str] = ["FlextResult", "r"]
