"""Fail DSL factory methods — return r[T].fail(...) directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, m, p
from flext_core._exceptions._factories import flextexceptionsfactories_part_03

if TYPE_CHECKING:
    from flext_core import r


class FlextExceptionsFactoriesPart04(
    flextexceptionsfactories_part_03.FlextExceptionsFactoriesPart03
):
    @staticmethod
    def fail_conflict[TResult](
        resource_type: str,
        resource_id: str,
        reason: str | None = None,
        *,
        options: m.ExceptionFactoryOptions | None = None,
        result_type: type[r[TResult]] | None = None,
    ) -> p.Result[TResult]:
        """Return r[T].fail with a canonical conflict message.

        Usage::

            return e.fail_conflict(c.ContextScope.USER, user_id, "already active")

        Returns:
            R[T].fail with a canonical conflict message.

        """
        options, error = FlextExceptionsFactoriesPart04._resolve_options(options)
        params = m.ConflictErrorParams(
            resource_type=resource_type,
            resource_id=resource_id,
            conflict_reason=reason,
        )
        msg = FlextExceptionsFactoriesPart04._failure_message(
            f"create {resource_type} {resource_id!r}",
            params=params,
            error=error,
        )
        return FlextExceptionsFactoriesPart04._fail_result(
            msg,
            params,
            options=options,
            default_error_code=c.ErrorCode.ALREADY_EXISTS,
            result_type=result_type,
        )


__all__: list[str] = ["FlextExceptionsFactoriesPart04"]
