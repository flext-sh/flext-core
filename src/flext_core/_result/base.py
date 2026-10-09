"""Internal data model for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, PrivateAttr

from flext_core import c
from flext_core._runtime._metadata import FlextRuntimeMetadata
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.services import FlextTypesServices
from flext_core.typings import JsonDict


class FlextResultBase[T](BaseModel):
    """Internal data container for FlextResult.

    Population keeps pydantic's own ``(**data)`` contract; the result family
    builds instances only through ``create_success`` and ``create_failure``,
    which validate the public fields and then bind the private payload.
    """

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True,
        "extra": "forbid",
    }

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: JsonDict | None = None

    _payload: T = PrivateAttr()
    _exception: BaseException | None = PrivateAttr(default=None)

    @classmethod
    def create_success(cls, value: T) -> Self:
        """Build a successful result carrying ``value``.

        Returns:
            The validated success instance.

        """
        cls.reject_banned_result_parameterization()
        cls.reject_banned_success_payload(value)
        instance: Self = cls.model_validate({"success": True})
        instance._payload = value
        return instance

    @classmethod
    def create_failure(
        cls,
        error: str,
        error_code: str | None,
        error_data: JsonDict | None,
        exception: BaseException | None,
    ) -> Self:
        """Build a failed result from already-normalized failure state.

        Returns:
            The validated failure instance.

        """
        cls.reject_banned_result_parameterization()
        instance: Self = cls.model_validate({
            "success": False,
            "error": error,
            "error_code": error_code,
            "error_data": error_data,
        })
        instance._exception = exception
        return instance

    @classmethod
    def reject_banned_result_parameterization(cls) -> None:
        """Reject ``FlextResult[None]`` and ``FlextResult[object]`` specializations.

        Raises:
            ValueError: If ``arg0 is None or arg0 is type(None)``; or if ``arg0 is
                object``.

        """
        args = cls.__pydantic_generic_metadata__["args"]
        if not args:
            return
        arg0 = args[0]
        if arg0 is None or arg0 is type(None):
            raise ValueError(c.ERR_RESULT_TYPE_PARAM_NONE_FORBIDDEN)
        if arg0 is object:
            raise ValueError(c.ERR_RESULT_TYPE_PARAM_OBJECT_FORBIDDEN)

    @staticmethod
    def reject_banned_success_payload(value: T) -> None:
        """Reject ``None`` and bare ``object()`` as success payloads.

        Raises:
            ValueError: If ``value is None``; or if ``type(value) is object``.

        """
        if value is None:
            raise ValueError(c.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_NONE)
        if type(value) is object:
            raise ValueError(c.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_OBJECT)

    @staticmethod
    def validate_error_data(
        error_data: FlextTypingBase.JsonMapping
        | FlextTypesServices.ConfigModelInput
        | None,
    ) -> JsonDict | None:
        normalized = FlextRuntimeMetadata.normalize_model_input_mapping(error_data)
        if normalized is None:
            return None
        return dict(normalized)


__all__: list[str] = ["FlextResultBase"]
