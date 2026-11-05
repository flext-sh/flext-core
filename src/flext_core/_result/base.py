"""Data model and validation contracts for the FlextResult railway container.

Defines the fundamental Pydantic data schema, field constraints, immutability
guarantees, and parameterization guards for all result instances across the platform.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-005 (Config, Settings, Constants, Templates, Schemas SSOT):
      ``flext/docs/architecture/adr/005-config-settings-constants-templates-schemas-ssot.md``
    - :mod:`flext_core._result.behavior`: Behavioral extensions and property accessors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Self

from flext_core import p, t
from flext_core._constants.errors import FlextConstantsErrors
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities import (
    FlextUtilitiesPydantic,
    FlextUtilitiesTypingAdapterJson,
)


class FlextResultBase[T](FlextModelsPydantic.BaseModel):
    """Underlying data container and strict validator for railway results.

    Enforces payload constraints, schema invariants, and explicit parameterization
    guards (forbidding untyped ``object`` and bare ``None``).

    Type Parameters:
        T: The payload type carried upon successful completion.

    Attributes:
        success: Whether the result indicates a successful outcome.
        error: Descriptive failure message, or ``None`` when successful.
        error_code: Standardized error code identifier, or ``None``.
        error_data: Structured context dictionary associated with failures.

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
        """Instantiate a validated successful result binding the provided payload.

        Args:
            value: The success payload to carry within the result.

        Returns:
            Self: A newly created success instance.

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
        """Instantiate a validated failure result binding normalized error state.

        Args:
            error: Descriptive failure message.
            error_code: Optional machine-readable error code.
            error_data: Optional structured JSON-compatible context dictionary.
            exception: Optional underlying exception causing the failure.

        Returns:
            Self: A newly created failure instance.

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
        """Enforce strict typing by rejecting banned generic parameters.

        Guarantees that results are never parameterized with bare ``None`` or
        untargeted ``object`` types, ensuring static type checking remains sound.

        Raises:
            ValueError: If the type argument is ``None``, ``type(None)``, or ``object``.

        """
        args = cls.__pydantic_generic_metadata__["args"]
        if not args:
            return
        arg0 = args[0]
        if arg0 is None or arg0 is type(None):
            raise ValueError(FlextConstantsErrors.ERR_RESULT_TYPE_PARAM_NONE_FORBIDDEN)
        if arg0 is object:
            raise ValueError(
                FlextConstantsErrors.ERR_RESULT_TYPE_PARAM_OBJECT_FORBIDDEN
            )

    @staticmethod
    def reject_banned_success_payload(value: T) -> None:
        """Validate that a success payload does not contain disallowed values.

        Raises:
            ValueError: If ``value`` is ``None`` or an instance of exact ``object``.

        """
        if value is None:
            raise ValueError(
                FlextConstantsErrors.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_NONE
            )
        if type(value) is object:
            raise ValueError(
                FlextConstantsErrors.ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_OBJECT
            )

    @staticmethod
    def validate_error_data(
        error_data: t.JsonMapping | p.ConfigModelInput | None,
    ) -> t.JsonDict | None:
        """Normalize structured error context into a strict JSON-compatible dictionary.

        Args:
            error_data: Raw error metadata mapping or Pydantic model input.

        Returns:
            t.JsonDict | None: A normalized JSON dictionary, or ``None`` if input
                normalizes to None.

        """
        if error_data is None:
            return None
        payload = (
            error_data.model_dump(mode="json")
            if isinstance(error_data, p.HasModelDump)
            else error_data
        )
        return FlextUtilitiesTypingAdapterJson.json_dict_adapter().validate_python(
            FlextUtilitiesPydantic.normalize_to_json_value(payload),
            strict=True,
        )


__all__: list[str] = ["FlextResultBase"]
