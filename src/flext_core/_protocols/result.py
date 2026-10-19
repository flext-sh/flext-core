"""Structural result and model-dump contracts for FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar, overload, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from flext_core import m


_ResultViewT_co = TypeVar("_ResultViewT_co", covariant=True)


class FlextProtocolsResult:
    """Single structural result contract used across FLEXT."""

    type ModelInput = t.ModelInput[FlextProtocolsResult.HasModelDump]
    type ConfigModelInput = t.ConfigModelInput[FlextProtocolsResult.HasModelDump]
    type MetadataInput = t.MetadataInput[FlextProtocolsResult.HasModelDump]

    @runtime_checkable
    class ResultView(Protocol[_ResultViewT_co]):
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
        def value(self) -> _ResultViewT_co: ...

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
    class ResultState[T](Protocol):
        """Structural state surface of the railway result contract."""

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

    @runtime_checkable
    class ResultOps[T](Protocol):
        """Structural monadic-operation surface of the result contract."""

        def unwrap(self) -> T: ...
        def unwrap_or[D](self, default: D) -> T | D: ...
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D: ...

        def flat_map[U](
            self,
            func: Callable[[T], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[U]: ...

        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U: ...

        def lash[U](
            self,
            func: Callable[[str], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[T | U]: ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocolsResult.Result[U]: ...

        @overload
        def flow_through(self) -> FlextProtocolsResult.Result[T]: ...

        @overload
        def flow_through[U](
            self,
            first: Callable[[T | U], FlextProtocolsResult.Result[U]],
            /,
            *funcs: Callable[[T | U], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[T | U]: ...

        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextProtocolsResult.Result[T]: ...

        @overload
        def map_or(self, default: None, func: None = None) -> T | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> T | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...
        def map_or[U](
            self,
            default: U,
            func: Callable[[T], U] | None = None,
        ) -> U | T: ...

        def tap(self, func: Callable[[T], None]) -> FlextProtocolsResult.Result[T]: ...

        def tap_error(
            self,
            func: Callable[[str], None],
        ) -> FlextProtocolsResult.Result[T]: ...

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> FlextProtocolsResult.Result[T]: ...

        def recover[U](
            self,
            func: Callable[[str], U],
        ) -> FlextProtocolsResult.Result[T | U]: ...

        def to_model[U: m.BaseModel](
            self,
            model: type[U],
        ) -> FlextProtocolsResult.Result[U]: ...

        def __bool__(self) -> bool: ...

    @runtime_checkable
    class Result[T](ResultState[T], ResultOps[T], Protocol):
        """Covariant structural railway interface over result operations and state."""

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
            self,
            *,
            mode: str = "python",
            exclude: set[str] | None = None,
        ) -> t.MappingKV[str, t.JsonPayload | None]: ...


__all__: list[str] = ["FlextProtocolsResult"]
