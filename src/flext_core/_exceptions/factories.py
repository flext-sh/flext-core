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
    FlextTypes as t,
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
        options = options or m.ExceptionFactoryOptions()
        options = m.ExceptionFactoryOptions.model_validate(options.model_dump())
        return options, options.error

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
            params=params,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            default_error_code=error_code or c.ErrorCode.NOT_FOUND_ERROR,
            result_type=result_type,
        )

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

            return e.fail_type_mismatch("FlextLogger", type(svc).__name__)
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

    @staticmethod
    def fail_connection[TResult](
        host: str,
        *,
        params: m.ConnectionErrorParams | None = None,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical connection-error message.

        Usage::

            return e.fail_connection(
                "ldap.example.com",
                params=m.ConnectionErrorParams(host="ldap.example.com", port=636),
                options=m.ExceptionFactoryOptions(error=exc),
            )

        """
        options, error = FlextExceptionsFactories._resolve_options(options)
        params = params or m.ConnectionErrorParams(host=host)
        msg = FlextExceptionsFactories._failure_message(
            f"connect to {host}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.CONNECTION_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_timeout[TResult](
        timeout_seconds: float,
        operation: str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical timeout message.

        Usage::

            return e.fail_timeout(30.0, "fetch_users")

        """
        params = m.TimeoutErrorParams(
            timeout_seconds=timeout_seconds,
            operation=operation,
        )
        op_label = operation or "operation"
        msg = FlextExceptionsFactories._failure_message(
            f"{op_label} (timeout={timeout_seconds}s)",
            params=params,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            default_error_code=error_code or c.ErrorCode.TIMEOUT_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_auth[TResult](
        auth_method: str | None = None,
        user_id: str | None = None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical authentication-error message.

        Usage::

            return e.fail_auth("ldap", user_id)

        """
        options, error = FlextExceptionsFactories._resolve_options(options)
        params = m.AuthenticationErrorParams(
            auth_method=auth_method,
            user_id=user_id,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"authenticate user {user_id or 'unknown'}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.AUTHENTICATION_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_authz[TResult](
        user_id: str,
        resource: str,
        permission: str | None = None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical authorization-error message.

        Usage::

            return e.fail_authz(user_id, "admin.panel", "write")

        """
        options, error = FlextExceptionsFactories._resolve_options(options)
        params = m.AuthorizationErrorParams(
            user_id=user_id,
            resource=resource,
            permission=permission,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"authorize {user_id!r} on {resource!r}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.AUTHORIZATION_ERROR,
            result_type=result_type,
        )

    @staticmethod
    def fail_conflict[TResult](
        resource_type: str,
        resource_id: str,
        reason: str | None = None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[FlextResult[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical conflict message.

        Usage::

            return e.fail_conflict("user", user_id, "already active")

        """
        options, error = FlextExceptionsFactories._resolve_options(options)
        params = m.ConflictErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
            conflict_reason=reason,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"create {resource_type} {resource_id!r}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactories._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.ALREADY_EXISTS,
            result_type=result_type,
        )


__all__: t.MutableSequenceOf[str] = ["FlextExceptionsFactories"]
