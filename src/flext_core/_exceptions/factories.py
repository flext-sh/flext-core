"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextExceptionsTemplate,
    FlextModelsExceptionParams as m,
    FlextModelsPydantic as mp,
    c,
    p,
    r,
    t,
)


class FlextExceptionsFactories:
    """Centralized fail_* factory helpers returning r[T].fail(...).

    Eliminates the 4-line boilerplate across all modules.
    """

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
            return FlextExceptionsTemplate.render_template(
                template_without_error,
                operation=operation,
                params=params,
            )
        return FlextExceptionsTemplate.render_error_template(
            c.ERR_TEMPLATE_FAILED_WITH_ERROR,
            operation=operation,
            error=error,
            params=params,
        )

    @staticmethod
    def fail_operation[TResult](
        operation: str,
        exc: Exception | str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical operation-error message.

        Usage::

            return e.fail_operation("resolve factory service", exc)

        """
        _ = result_type
        params = m.OperationErrorParams(
            operation=operation,
            reason=str(exc) if exc is not None else None,
        )
        msg = FlextExceptionsFactories._failure_message(
            operation,
            params=params,
            error=exc,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.OPERATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
            exception=exc if isinstance(exc, BaseException) else None,
        )

    @staticmethod
    def fail_not_found[TResult](
        resource_type: str,
        resource_id: str,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical not-found message.

        Usage::

            return e.fail_not_found("service", name)

        """
        _ = result_type
        params = m.NotFoundErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
        )
        msg = FlextExceptionsTemplate.render_template(
            c.ERR_SERVICE_NOT_FOUND,
            name=resource_id,
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.NOT_FOUND_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_type_mismatch[TResult](
        expected: str,
        actual: str,
        *,
        service_name: str | None = None,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical type-mismatch message.

        Usage::

            return e.fail_type_mismatch("FlextLogger", type(svc).__name__)

        """
        _ = result_type
        params = m.ServiceLookupParams(
            service_name=service_name,
            expected_type=expected,
            actual_type=actual,
        )
        msg = FlextExceptionsTemplate.render_template(
            c.ERR_SERVICE_TYPE_MISMATCH,
            type_name=expected,
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.TYPE_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_validation[TResult](
        field: str | None = None,
        value: t.JsonValue | None = None,
        *,
        error_code: str | None = None,
        error: Exception | str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical validation-failed message.

        Usage::

            return e.fail_validation("config_key", raw_value, error=exc)

        """
        _ = result_type
        params = m.ValidationErrorParams(field=field, value=value)
        base_msg = (
            FlextExceptionsTemplate.render_error_template(
                c.ERR_TEMPLATE_FAILED_WITH_ERROR,
                operation=f"validate {field or 'input'}",
                error=error,
                params=params,
            )
            if error is not None
            else FlextExceptionsTemplate.render_template(
                c.ERR_TEMPLATE_VALIDATION_FAILED_FOR_FIELD,
                field=field or "input",
                params=params,
            )
        )
        return r[TResult].fail(
            base_msg,
            error_code=error_code or c.ErrorCode.VALIDATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(
                params,
                cause=str(error) if error is not None else None,
            ),
            exception=error if isinstance(error, BaseException) else None,
        )

    @staticmethod
    def fail_config_error[TResult](
        config_key: str,
        config_source: str | None = None,
        *,
        error: Exception | str | None = None,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical configuration-error message.

        Usage::

            return e.fail_config_error("database.url", "env")

        """
        _ = result_type
        params = m.ConfigurationErrorParams(
            config_key=config_key,
            config_source=config_source,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"read config key {config_key!r}",
            params=params,
            error=error,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.CONFIGURATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
            exception=error if isinstance(error, BaseException) else None,
        )

    @staticmethod
    def fail_connection[TResult](
        host: str,
        port: int | None = None,
        *,
        timeout: t.Numeric | None = None,
        error: Exception | str | None = None,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical connection-error message.

        Usage::

            return e.fail_connection("ldap.example.com", 636, error=exc)

        """
        _ = result_type
        params = m.ConnectionErrorParams(host=host, port=port, timeout=timeout)
        msg = FlextExceptionsFactories._failure_message(
            f"connect to {host}",
            params=params,
            error=error,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.CONNECTION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
            exception=error if isinstance(error, BaseException) else None,
        )

    @staticmethod
    def fail_timeout[TResult](
        timeout_seconds: float,
        operation: str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical timeout message.

        Usage::

            return e.fail_timeout(30.0, "fetch_users")

        """
        _ = result_type
        params = m.TimeoutErrorParams(
            timeout_seconds=timeout_seconds,
            operation=operation,
        )
        op_label = operation or "operation"
        msg = FlextExceptionsFactories._failure_message(
            f"{op_label} (timeout={timeout_seconds}s)",
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.TIMEOUT_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_auth[TResult](
        auth_method: str | None = None,
        user_id: str | None = None,
        *,
        error: Exception | str | None = None,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical authentication-error message.

        Usage::

            return e.fail_auth("ldap", user_id)

        """
        _ = result_type
        params = m.AuthenticationErrorParams(
            auth_method=auth_method,
            user_id=user_id,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"authenticate user {user_id or 'unknown'}",
            params=params,
            error=error,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.AUTHENTICATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
            exception=error if isinstance(error, BaseException) else None,
        )

    @staticmethod
    def fail_authz[TResult](
        user_id: str,
        resource: str,
        permission: str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical authorization-error message.

        Usage::

            return e.fail_authz(user_id, "admin.panel", "write")

        """
        _ = result_type
        params = m.AuthorizationErrorParams(
            user_id=user_id,
            resource=resource,
            permission=permission,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"authorize {user_id!r} on {resource!r}",
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.AUTHORIZATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_conflict[TResult](
        resource_type: str,
        resource_id: str,
        reason: str | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical conflict message.

        Usage::

            return e.fail_conflict("user", user_id, "already active")

        """
        _ = result_type
        params = m.ConflictErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
            conflict_reason=reason,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"create {resource_type} {resource_id!r}",
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.ALREADY_EXISTS,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_rate_limit[TResult](
        limit: int,
        window_seconds: int,
        retry_after: t.Numeric | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical rate-limit message.

        Usage::

            return e.fail_rate_limit(100, 60, retry_after=30)

        """
        _ = result_type
        params = m.RateLimitErrorParams(
            limit=limit,
            window_seconds=window_seconds,
            retry_after=retry_after,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"rate limit ({limit}/{window_seconds}s)",
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.OPERATION_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )

    @staticmethod
    def fail_circuit_breaker[TResult](
        service_name: str,
        failure_count: int | None = None,
        reset_timeout: t.Numeric | None = None,
        *,
        error_code: str | None = None,
        result_type: type[TResult] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical circuit-breaker message.

        Usage::

            return e.fail_circuit_breaker("ldap_service", failure_count=5)

        """
        _ = result_type
        params = m.CircuitBreakerErrorParams(
            service_name=service_name,
            failure_count=failure_count,
            reset_timeout=reset_timeout,
        )
        msg = FlextExceptionsFactories._failure_message(
            f"call {service_name!r} (circuit open)",
            params=params,
        )
        return r[TResult].fail(
            msg,
            error_code=error_code or c.ErrorCode.EXTERNAL_SERVICE_ERROR,
            error_data=FlextExceptionsTemplate.result_error_data(params),
        )


__all__: list[str] = ["FlextExceptionsFactories"]
