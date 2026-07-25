"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextConstants as c, FlextProtocols as p
from flext_core._exceptions.template import FlextExceptionsTemplate
from flext_core._models.exception_params import FlextModelsExceptionParams as m

from .flextexceptionsfactories_part_01 import (
    FlextExceptionsFactories as FlextExceptionsFactoriesPart01,
)

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextExceptionsFactories(FlextExceptionsFactoriesPart01):
    @staticmethod
    def fail_type_mismatch[TResult](
        details: m.ServiceLookupParams | str,
        actual: str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical type-mismatch message.

        Usage::

            return e.fail_type_mismatch("FlextUtilitiesLogging", type(svc).__name__)
            return e.fail_type_mismatch(
                m.ServiceLookupParams(
                    service_name="connection",
                    expected_type="ldap3.Connection",
                    actual_type=type(raw).__name__,
                ),
            )

        """
        params = (
            details
            if isinstance(details, m.ServiceLookupParams)
            else m.ServiceLookupParams(
                expected_type=details,
                actual_type=actual,
            )
        )
        expected_type = (
            params.expected_type
            if params.expected_type is not None
            else c.DEFAULT_EMPTY_STRING
        )
        msg = FlextExceptionsTemplate.render_template(
            c.ERR_SERVICE_TYPE_MISMATCH,
            type_name=expected_type,
            params=params,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            default_error_code=error_code or c.ErrorCode.TYPE_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_validation[TResult](
        details: m.ValidationErrorParams | str | None = None,
        *,
        error_code: str | None = None,
        error: Exception | str | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical validation-failed message.

        Usage::

            return e.fail_validation("config_key", error=exc)
            return e.fail_validation(
                m.ValidationErrorParams(field="config_key", value=raw_value),
                error=exc,
            )

        """
        params = (
            details
            if isinstance(details, m.ValidationErrorParams)
            else m.ValidationErrorParams(field=details)
        )
        field_name = params.field if params.field is not None else "input"
        base_msg = (
            FlextExceptionsTemplate.render_template(
                c.ERR_TEMPLATE_FAILED_WITH_ERROR,
                operation=f"validate {field_name}",
                error=str(error),
                params=params,
            )
            if error is not None
            else FlextExceptionsTemplate.render_template(
                c.ERR_TEMPLATE_VALIDATION_FAILED_FOR_FIELD,
                field=field_name,
                params=params,
            )
        )
        return FlextExceptionsFactories._fail_result(
            base_msg,
            params,
            options=m.ExceptionFactoryOptions(error=error)
            if error is not None
            else None,
            default_error_code=error_code or c.ErrorCode.VALIDATION_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_config_error[TResult](
        config_key: str,
        config_source: str | None = None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical configuration-error message.

        Usage::

            return e.fail_config_error("database.url", "env")

        """
        options, error = FlextExceptionsFactories._resolve_options(options)
        params = m.ConfigurationErrorParams(
            config_key=config_key,
            config_source=config_source,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"read config key {config_key!r}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.CONFIGURATION_ERROR,
            result_type=result_type,
        )


__all__: list[str] = ["FlextExceptionsFactories"]
