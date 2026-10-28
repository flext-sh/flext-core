"""Internal data model for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self

from flext_core import c, t
from flext_core._models import FlextModelsPydantic
from flext_core._protocols import FlextProtocolsResult
from flext_core._typings import FlextTypingBase
from flext_core._utilities import (
    FlextUtilitiesPydantic,
    FlextUtilitiesTypingAdapterJson,
)


class FlextResultBase[T](FlextModelsPydantic.BaseModel):
    """Internal data container for FlextResult.

    Population keeps pydantic's own ``(**data)`` contract; the result family
    builds instances only through ``create_success`` and ``create_failure``,
    which validate the public fields and then bind the private payload.
    """

    model_config = FlextModelsPydantic.ConfigDict(extra="forbid")

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: t.JsonDict | None = None

    _payload: T = FlextModelsPydantic.PrivateAttr()
    _exception: BaseException | None = FlextModelsPydantic.PrivateAttr(default=None)

    @classmethod
    def create_success(cls, value: T) -> Self:
        """Build a successful result carrying ``value``.

        Args:
            value: The success payload to bind.

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
        error_data: t.JsonDict | None,
        exception: BaseException | None,
    ) -> Self:
        """Build a failed result from already-normalized failure state.

        Args:
            error: The human-readable error message.
            error_code: Optional machine-readable error code.
            error_data: Optional structured error payload.
            exception: Optional exception bound to the failure.

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
            ValueError: If the first type argument is ``None``/``NoneType`` or
                ``object``.

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
            ValueError: If ``value`` is ``None`` or its exact type is ``object``.

        """
        if value is None:
            raise ValueError(c.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_NONE)
        if type(value) is object:
            raise ValueError(c.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_OBJECT)

    @staticmethod
    def validate_error_data(
        error_data: FlextTypingBase.JsonMapping
        | FlextProtocolsResult.ConfigModelInput
        | None,
    ) -> t.JsonDict | None:
        """Validate and normalize error data into a plain JSON mapping.

        Args:
            error_data: The model, mapping, or config input to normalize.

        Returns:
            The resulting ``JsonDict``, or ``None`` when ``error_data``
            normalizes to ``None``.

        """
        if error_data is None:
            return None
        payload = (
            error_data.model_dump(mode="json")
            if isinstance(error_data, FlextProtocolsResult.HasModelDump)
            else error_data
        )
        return FlextUtilitiesTypingAdapterJson.json_dict_adapter().validate_python(
            FlextUtilitiesPydantic.normalize_to_json_value(payload),
            strict=True,
        )


__all__: list[str] = ["FlextResultBase"]
