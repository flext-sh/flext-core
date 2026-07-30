"""Internal data model for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, PrivateAttr
from returns.result import Result

if TYPE_CHECKING:
    from flext_core import t


class FlextResultBase[T](BaseModel):
    """Internal result data container — T cannot be None (returns convention)."""

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: t.JsonDict | None = None

    _payload: T = PrivateAttr()
    _exception: BaseException | None = PrivateAttr(default=None)
    _result: Result[T, str] = PrivateAttr()

    @staticmethod
    def _validate_error_data(
        error_data: t.JsonMapping | t.ConfigModelInput | None,
    ) -> t.JsonDict | None:
        from flext_core._runtime._metadata import FlextRuntimeMetadata as FlextRuntime

        normalized = FlextRuntime.normalize_model_input_mapping(error_data)
        if normalized is None:
            return None
        return dict(normalized)

    def __init__(
        self,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
        exception: BaseException | None = None,
    ) -> None:
        if success and value is None:
            msg = "Success result requires a non-None value"
            raise ValueError(msg)
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=self._validate_error_data(error_data),
        )
        if success:
            self._payload = value  # type: ignore[assignment]
            self._result = cast("Result[T, str]", Result.from_value(value))
        else:
            self._result = Result.from_failure(error if error is not None else "")
            if exception is not None:
                self._exception = exception


__all__: list[str] = ["FlextResultBase"]
