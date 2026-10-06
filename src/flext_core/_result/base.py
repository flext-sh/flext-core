"""Internal data model for FlextResult.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, PrivateAttr

from flext_core import c
from flext_core._runtime._metadata import FlextRuntimeMetadata
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.services import FlextTypesServices
from flext_core.typings import ConfigModelInput, JsonDict, JsonMapping, T


class FlextResultBase[T](BaseModel):
    """Internal data container for FlextResult."""

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: JsonDict | None = None

    _payload: T = PrivateAttr()
    _exception: BaseException | None = PrivateAttr(default=None)

    @classmethod
    def reject_banned_result_parameterization(cls) -> None:
        """Reject ``FlextResult[None]`` and ``FlextResult[object]`` specializations.

        Raises:
            ValueError: If ``arg0 is None or arg0 is type(None)``; or if ``arg0 is
                object``.

        """
        meta = getattr(cls, "__pydantic_generic_metadata__", None)
        if not isinstance(meta, dict):
            return
        args = meta.get("args") or ()
        if not args:
            return
        arg0 = args[0]
        if arg0 is None or arg0 is type(None):
            raise ValueError(c.ERR_RESULT_TYPE_PARAM_NONE_FORBIDDEN)
        if arg0 is object:
            raise ValueError(c.ERR_RESULT_TYPE_PARAM_OBJECT_FORBIDDEN)

    @staticmethod
    def reject_banned_success_payload(value: object) -> None:
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
        type(self).reject_banned_result_parameterization()
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=self.validate_error_data(error_data),
        )
        if success:
            self.reject_banned_success_payload(value)
            self._payload = cast("T", value)
        elif exception is not None:
            self._exception = exception


__all__: list[str] = ["FlextResultBase"]
