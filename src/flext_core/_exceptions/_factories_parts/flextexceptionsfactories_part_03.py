"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import (
    FlextConstants as c,
    FlextModelsExceptionParams as m,
    FlextProtocols as p,
)

from .flextexceptionsfactories_part_02 import (
    FlextExceptionsFactories as FlextExceptionsFactoriesPart02,
)

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextExceptionsFactories(FlextExceptionsFactoriesPart02):
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


__all__: list[str] = ["FlextExceptionsFactories"]
