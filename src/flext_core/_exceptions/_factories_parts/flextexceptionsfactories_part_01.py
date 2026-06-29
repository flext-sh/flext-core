"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, TypeVar

from flext_core import (
    FlextConstants as c,
    FlextExceptionsTemplate,
    FlextModelsExceptionParams as m,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
)

TExceptionParams = TypeVar("TExceptionParams", bound=mp.BaseModel)

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextExceptionsFactories:
    """Centralized fail_* factory helpers returning r[T].fail(...).

    Eliminates the 4-line boilerplate across all modules.
    """

    @staticmethod
    def _result_type[TValue](
        result_type: type[FlextResult[TValue]] | None = None,
    ) -> type[FlextResult[TValue]]:
        """Resolve FlextResult lazily to avoid runtime import cycles."""
        if result_type is not None:
            return result_type
        result_module = import_module("flext_core")
        result_cls: type[FlextResult[TValue]] = result_module.FlextResult
        return result_cls

    @staticmethod
    def _failure_message(
        operation: str,
        *,
        params: mp.BaseModel | None = None,
        error: Exception | str | None = None,
    ) -> str:
        """Render the canonical failure message with or without an error cause."""
        if error is None:
            template_without_error = c.ERR_TEMPLATE_FAILED_WITH_ERROR.split(": ", 1)[0]
            message: str = FlextExceptionsTemplate.render_template(
                template_without_error,
                operation=operation,
                params=params,
            )
            return message
        message_with_error: str = FlextExceptionsTemplate.render_template(
            c.ERR_TEMPLATE_FAILED_WITH_ERROR,
            operation=operation,
            error=str(error),
            params=params,
        )
        return message_with_error

    @staticmethod
    def _resolve_options(
        options: m.ExceptionFactoryOptions | None = None,
    ) -> tuple[m.ExceptionFactoryOptions, Exception | str | None]:
        resolved_options = (
            options if options is not None else m.ExceptionFactoryOptions()
        )
        return resolved_options, resolved_options.error

    @staticmethod
    def _normalize_params(
        params: TExceptionParams | None,
        params_type: type[TExceptionParams],
        update: dict[str, object | None],
    ) -> TExceptionParams:
        if params is None:
            return params_type.model_validate(update)
        return params.model_copy(
            update={key: value for key, value in update.items() if value is not None}
        )

    @staticmethod
    def _fail_result[
        TResult,
    ](
        message: str,
        params: mp.BaseModel | None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        default_error_code: str,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        options, error = FlextExceptionsFactories._resolve_options(options)
        return FlextExceptionsFactories._result_type(result_type).fail(
            message,
            error_code=options.error_code or default_error_code,
            error_data=FlextExceptionsTemplate.result_error_data(
                params,
                cause=str(error) if error is not None else None,
            ),
            exception=error if isinstance(error, BaseException) else None,
        )

    @staticmethod
    def fail_operation[TResult](
        operation: str,
        exc: Exception | str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical operation-error message.

        Usage::

            return e.fail_operation("resolve factory service", exc)

        """
        params = m.OperationErrorParams(
            operation=operation,
            reason=str(exc) if exc is not None else None,
        )
        msg = FlextExceptionsFactories._failure_message(
            operation,
            params=params,
            error=exc,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=m.ExceptionFactoryOptions(error=exc) if exc is not None else None,
            default_error_code=error_code or c.ErrorCode.OPERATION_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_not_found[TResult](
        resource_type: str,
        resource_id: str,
        *,
        error_code: str | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical not-found message.

        Usage::

            return e.fail_not_found("service", name)

        """
        params = m.NotFoundErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
        )
        msg = FlextExceptionsTemplate.render_template(
            c.ERR_SERVICE_NOT_FOUND,
            name=resource_id,
            resource_type=resource_type.capitalize(),
            params=params,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            default_error_code=error_code or c.ErrorCode.NOT_FOUND_ERROR,
            result_type=result_type,
        )


__all__: list[str] = ["FlextExceptionsFactories"]
