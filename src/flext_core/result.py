"""Type-safe result type for operations."""

from __future__ import annotations

from ._result.base import JsonDict
from ._result.behavior import FlextResultBehavior
from ._result.composition import FlextResultComposition
from ._result.construction import FlextResultConstruction
from ._result.transforms import FlextResultTransforms
from ._result.unwrap import FlextResultUnwrap


class FlextResult[T](
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


r = FlextResult

__all__: list[str] = ["FlextResult", "r"]
