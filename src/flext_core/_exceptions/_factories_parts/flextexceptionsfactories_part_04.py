"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from flext_core import (
    FlextConstants as c,
    FlextModelsExceptionParams as m,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
)

from .flextexceptionsfactories_part_03 import (
    FlextExceptionsFactories as FlextExceptionsFactoriesPart03,
)

TExceptionParams = TypeVar("TExceptionParams", bound=mp.BaseModel)


if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextExceptionsFactories(FlextExceptionsFactoriesPart03):
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


__all__: list[str] = ["FlextExceptionsFactories"]
