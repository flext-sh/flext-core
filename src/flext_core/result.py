"""Type-safe result type for operations.

The composed private base lives in the ``_result`` family package
(one top-level class per module, NS-000); this namespace module publishes
the public concrete facade and the canonical ``r`` letter alias.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Never, override

from flext_core._result import FlextResultComposed

if TYPE_CHECKING:
    from flext_core import p, t


class FlextResult[T](FlextResultComposed[T]):
    """Public concrete result facade; runtime and typing share one MRO.

    Instances carry either a success payload or a failure message plus the
    exception that produced it. Every operation returns a new ``FlextResult``
    so failures propagate explicitly instead of raising at the call site;
    use the ``unwrap``/``behavior`` helpers to extract payloads or bind the
    error. ``r`` is the canonical letter alias for this class.
    """

    @staticmethod
    @override
    def _create_success[V](value: V) -> p.Result[V]:
        """Construct a payload using the complete public result family.

        Returns:
            The validated public success result.

        """
        return FlextResult[V].create_success(value)

    @staticmethod
    @override
    def _create_failure(
        error: str,
        error_code: str | None,
        error_data: t.JsonDict | None,
        exception: BaseException | None,
    ) -> p.Result[Never]:
        """Bind normalized failure state after the public MRO is complete.

        Returns:
            The public failure result with no possible success payload.

        """
        return FlextResult[Never].create_failure(
            error,
            error_code,
            error_data,
            exception,
        )


r = FlextResult


__all__: list[str] = ["FlextResult", "r"]
