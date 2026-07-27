"""Shared typing contract for FlextResult implementation mixins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flext_core._protocols.result import FlextProtocolsResult as p

if TYPE_CHECKING:
    from returns.result import Result

    from flext_core import FlextTypes as t
    from flext_core._models.containers import FlextModelsContainers as mc


class FlextResultBehaviorMixin[T](p.Result[T], ABC):
    """Internal contract supplied by the public FlextResult facade."""

    _exception: BaseException | None
    _result: Result[T, str] | None

    def __init__(
        self,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
        exception: BaseException | None = None,
    ) -> None: ...

    @staticmethod
    @abstractmethod
    def _validate_error_data(
        error_data: t.JsonMapping | t.ConfigModelInput | None,
    ) -> mc.ConfigMap | None: ...


__all__: list[str] = ["FlextResultBehaviorMixin"]
