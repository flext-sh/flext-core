"""Internal data model for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast

from pydantic import BaseModel, PrivateAttr
from returns.result import Result

from flext_core._protocols.result import FlextProtocolsResult as prt
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._typings.services import FlextTypesServices as ts

type JsonMapping = Mapping[str, tp.JsonValue]
type JsonDict = dict[str, tp.JsonValue]
type ConfigModelInput = prt.HasModelDump | JsonMapping


T = TypeVar("T")


class FlextResultBase[T](BaseModel):
    """Internal result data container — T cannot be None (returns convention)."""

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: JsonDict | None = None

    _payload: T = PrivateAttr()
    _exception: BaseException | None = PrivateAttr(default=None)
    _result: Result[T, str] = PrivateAttr()

    @staticmethod
    def validate_error_data(
        error_data: t.JsonMapping | ts.ConfigModelInput | None,
    ) -> JsonDict | None:
        from flext_core._runtime._metadata import FlextRuntimeMetadata as FlextRuntime

        normalized = FlextRuntime.normalize_model_input_mapping(error_data)
        if normalized is None:
            return None
        return dict(normalized)

    @staticmethod
    def _validate_success_value[V](value: V | None) -> V:
        if value is None:
            msg = "Success result payload cannot be None"
            raise ValueError(msg)
        return value

    def __init__(
        self,
        error_code: str | None = None,
        error_data: JsonMapping | ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
        exception: BaseException | None = None,
    ) -> None:
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=self.validate_error_data(error_data),
        )
        if success:
            validated_value = self._validate_success_value(value)
            self._payload = validated_value
            self._result = cast("Result[T, str]", Result.from_value(validated_value))
        else:
            self._result = Result.from_failure(error if error is not None else "")
            if exception is not None:
                self._exception = exception


__all__: list[str] = ["FlextResultBase"]
