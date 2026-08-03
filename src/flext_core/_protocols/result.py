"""Structural result and model-dump contracts for FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar, overload, runtime_checkable

from flext_core._models.pydantic import FlextModelsPydantic as mp

ResultT_co = TypeVar("ResultT_co", covariant=True)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from flext_core._typings.base import FlextTypingBase as t
    from flext_core._typings.services import FlextTypesServices as ts
    from flext_core import FlextModels as m


class FlextProtocolsResult:
    """Single structural result contract used across FLEXT."""

    @runtime_checkable
    class Result(Protocol[ResultT_co]):
        """Structural railway result contract; covariant payload."""

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
        def value(self) -> ResultT_co: ...

        def __enter__(self) -> Self: ...

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None: ...

        def __or__[D](self, default: D) -> ResultT_co | D: ...

        def unwrap(self) -> ResultT_co: ...
        def unwrap_or[D](self, default: D) -> ResultT_co | D: ...
        def unwrap_or_else[D](self, func: Callable[[], D]) -> ResultT_co | D: ...

        def flat_map[U](
            self, func: Callable[..., FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]: ...

        def fold[U](
            self, on_failure: Callable[[str], U], on_success: Callable[..., U]
        ) -> U: ...

        def lash[U](
            self, func: Callable[[str], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[ResultT_co | U]: ...

        def map[U](self, func: Callable[..., U]) -> FlextProtocolsResult.Result[U]: ...

        def flow_through(
            self, *funcs: Callable[..., FlextProtocolsResult.Result[ResultT_co]]
        ) -> FlextProtocolsResult.Result[ResultT_co]: ...

        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextProtocolsResult.Result[ResultT_co]: ...

        @overload
        def map_or(self, default: None, func: None = None) -> ResultT_co | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> ResultT_co | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[..., U]) -> U: ...
        def map_or[U](
            self, default: U, func: Callable[..., U] | None = None
        ) -> U | ResultT_co: ...

        def tap(
            self, func: Callable[..., None]
        ) -> FlextProtocolsResult.Result[ResultT_co]: ...

        def tap_error(
            self, func: Callable[[str], None]
        ) -> FlextProtocolsResult.Result[ResultT_co]: ...

        def filter(
            self, predicate: Callable[..., bool]
        ) -> FlextProtocolsResult.Result[ResultT_co]: ...

        def recover[U](
            self, func: Callable[[str], U]
        ) -> FlextProtocolsResult.Result[ResultT_co | U]: ...

        def to_model[U: mp.BaseModel](
            self, model: type[U]
        ) -> FlextProtocolsResult.Result[U]: ...

        def __bool__(self) -> bool: ...


    @runtime_checkable
    class HasModelDump(Protocol):
        def model_dump(
            self, *, mode: str = "python"
        ) -> t.MappingKV[str, ts.JsonPayload | None]: ...


__all__: list[str] = ["FlextProtocolsResult"]
