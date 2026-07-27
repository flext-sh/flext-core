"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, overload, override

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols._result_parts.flextprotocolsresult_part_01 import (
    FlextProtocolsResult as FlextProtocolsResultPart01,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from flext_core._typings.base import FlextTypingBase as t


class FlextProtocolsResult(FlextProtocolsResultPart01):
    class Result[T_co](ABC):
        """Nominal public result contract for direct static typing across FLEXT.

        Pure ABC (not inheriting ResultLike Protocol) so Pydantic BaseModel
        subclasses (FlextResult) can inherit without metaclass conflict
        between ``_ProtocolMeta`` and ``ModelMetaclass``. ResultLike remains
        available as a @runtime_checkable Protocol for structural isinstance.
        """

        @property
        @abstractmethod
        def error(self) -> str | None: ...

        @property
        @abstractmethod
        def error_code(self) -> str | None: ...

        @property
        @abstractmethod
        def error_data(self) -> t.JsonMapping | None: ...

        @property
        @abstractmethod
        def success(self) -> bool: ...

        @property
        @abstractmethod
        def exception(self) -> BaseException | None: ...

        @property
        @abstractmethod
        def failure(self) -> bool: ...

        @property
        @abstractmethod
        def value(self) -> T_co: ...

        @abstractmethod
        def __enter__(self) -> Self: ...

        @abstractmethod
        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None: ...

        @overload
        def __or__(self, default: T_co) -> T_co: ...

        @overload
        def __or__[D](self, default: D) -> T_co | D: ...

        @abstractmethod
        def __or__[D](self, default: T_co | D) -> T_co | D: ...

        @abstractmethod
        def unwrap(self) -> T_co: ...

        @abstractmethod
        def unwrap_or[D](self, default: D) -> T_co | D: ...

        @abstractmethod
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T_co | D: ...

        @abstractmethod
        def flat_map[U](
            self, func: Callable[[T_co], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def fold[U](
            self, on_failure: Callable[[str], U], on_success: Callable[[T_co], U]
        ) -> U: ...

        @abstractmethod
        def lash(
            self, func: Callable[[str], FlextProtocolsResult.Result[T_co]]
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def map[U](
            self, func: Callable[[T_co], U]
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def flow_through(
            self, *funcs: Callable[[T_co], FlextProtocolsResult.Result[T_co]]
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @overload
        def map_or(self, default: None, func: None = None) -> T_co | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> T_co | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[[T_co], U]) -> U: ...

        @abstractmethod
        def map_or[U](
            self, default: U, func: Callable[[T_co], U] | None = None
        ) -> U | T_co: ...

        @abstractmethod
        def tap(
            self, func: Callable[[T_co], None]
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def tap_error(self, func: Callable[[str], None]) -> Self: ...

        @abstractmethod
        def filter(
            self, predicate: Callable[[T_co], bool]
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def recover[U](
            self, func: Callable[[str], U]
        ) -> FlextProtocolsResult.Result[T_co | U]: ...

        @abstractmethod
        def to_model[U: mp.BaseModel](
            self, model: type[U]
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def __bool__(self) -> bool: ...

        @override
        def __repr__(self) -> str:
            """Default repr for Result ABC."""
            return f"{type(self).__name__}(success={self.success})"


__all__: list[str] = ["FlextProtocolsResult"]
