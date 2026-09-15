"""Structural result and model-dump contracts for FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from flext_core import m

    from .._typings.base import FlextTypingBase as t
    from .._typings.services import FlextTypesServices as ts


ResultViewT_co = TypeVar("ResultViewT_co", covariant=True)


class FlextProtocolsResult:
    """Single structural result contract used across FLEXT."""

    @runtime_checkable
    class ResultView(Protocol[ResultViewT_co]):
        """Covariant read-only result state observed by assertions and reporting."""

        @property
        def error(self) -> str | None: ...
        @property
        def error_code(self) -> str | None: ...
        @property
        def error_data(self) -> t.JsonMapping | None: ...
        @property
        def success(self) -> bool: ...
        @property
        def exception(self) -> BaseException | None: ...
        @property
        def failure(self) -> bool: ...
        @property
        def value(self) -> ResultViewT_co: ...

    @runtime_checkable
    class FailureLike(Protocol):
        @property
        def error(self) -> str | None: ...
        @property
        def error_code(self) -> str | None: ...
        @property
        def error_data(self) -> t.JsonMapping | None: ...
        @property
        def exception(self) -> BaseException | None: ...
        @property
        def failure(self) -> bool: ...
        @property
        def success(self) -> bool: ...

    @runtime_checkable
    class Result[T](Protocol):
        """Structural railway result contract; invariant payload."""

        @property
        def error(self) -> str | None: ...
        @property
        def error_code(self) -> str | None: ...
        @property
        def error_data(self) -> t.JsonMapping | None: ...
        @property
        def success(self) -> bool: ...
        @property
        def exception(self) -> BaseException | None: ...
        @property
        def failure(self) -> bool: ...
        @property
        def value(self) -> T: ...

        def __enter__(self) -> Self: ...

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None: ...

        def __or__[D](self, default: D) -> T | D: ...

        def unwrap(self) -> T: ...
        def unwrap_or[D](self, default: D) -> T | D: ...
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D: ...

        def flat_map[U](
            self, func: Callable[[T], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]: ...

        def fold[U](
            self, on_failure: Callable[[str], U], on_success: Callable[[T], U]
        ) -> U: ...

        def lash[U](
            self, func: Callable[[str], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[T | U]: ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocolsResult.Result[U]: ...

        def flow_through(
            self, *funcs: Callable[[T], FlextProtocolsResult.Result[T]]
        ) -> FlextProtocolsResult.Result[T]: ...

        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextProtocolsResult.Result[T]: ...

        @overload
        def map_or(self, default: None, func: None = None) -> T | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> T | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...
        def map_or[U](
            self, default: U, func: Callable[[T], U] | None = None
        ) -> U | T: ...

        def tap(self, func: Callable[[T], None]) -> FlextProtocolsResult.Result[T]: ...

        def tap_error(
            self, func: Callable[[str], None]
        ) -> FlextProtocolsResult.Result[T]: ...

        def filter(
            self, predicate: Callable[[T], bool]
        ) -> FlextProtocolsResult.Result[T]: ...

        def recover[U](
            self, func: Callable[[str], U]
        ) -> FlextProtocolsResult.Result[T | U]: ...

        def to_model[U: m.BaseModel](
            self, model: type[U]
        ) -> FlextProtocolsResult.Result[U]: ...

        def __bool__(self) -> bool: ...

    @runtime_checkable
    class SuccessCheckable(Protocol):
        @property
        def success(self) -> bool: ...
        @property
        def failure(self) -> bool: ...

    @runtime_checkable
    class StructuredError(Protocol):
        @property
        def error_domain(self) -> str | None: ...
        @property
        def error_code(self) -> str | None: ...
        @property
        def error_message(self) -> str | None: ...
        @property
        def message(self) -> str: ...
        @property
        def metadata(self) -> m.Metadata: ...

        def matches_error_domain(self, domain: str) -> bool: ...

    @runtime_checkable
    class HasModelDump(Protocol):
        def model_dump(
            self, *, mode: str = "python"
        ) -> t.MappingKV[str, ts.JsonPayload | None]: ...

    class ResultFactory(Protocol):
        """Structural factory contract for the concrete result family."""

        @classmethod
        def reject_banned_result_parameterization(cls) -> None: ...

        @staticmethod
        def reject_banned_success_payload(value: object) -> None: ...

        @classmethod
        def require_error(cls, source: FlextProtocolsResult.FailureLike) -> str: ...

        @classmethod
        def fail(
            cls,
            error: str | None,
            *,
            error_code: str | None = None,
            error_data: t.JsonMapping | None = None,
            exception: BaseException | None = None,
        ) -> object: ...

        def __init__(self, *, value: object, success: bool) -> None: ...


__all__: list[str] = ["FlextProtocolsResult"]
