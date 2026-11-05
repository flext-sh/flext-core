"""Type-safe monadic result primitive for railway-oriented operations.

Publishes the concrete ``FlextResult`` facade and its canonical shorthand alias
``r``. Encapsulates either a typed success payload or a structured failure
carrying causal exception metadata, error codes, and contextual data, preventing
unhandled exceptions across platform boundaries.

Composes the internal ``_result`` hierarchy through C3 method resolution order,
unifying construction, transformation, composition, and unwrapping capabilities.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-027 (Facade Single-Owner Law):
      ``flext/docs/architecture/adr/027-facade-single-owner-law.md``
    - :mod:`flext_core._result.base`: Foundational result data model.
    - :mod:`flext_core.service`: Domain service base consuming railway results.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Never, override

from flext_core._result import FlextResultComposed

if TYPE_CHECKING:
    # The facades are annotations only; importing them at runtime re-enters the
    # package initializer while it is still publishing its lazy exports.
    from flext_core import p, t


class FlextResult[T](FlextResultComposed[T]):
    """Public concrete railway result facade.

    Carries either a strongly-typed success payload of type ``T`` or structured
    failure state containing error messages, codes, context dictionaries, and
    underlying causal exceptions. Enforces explicit error handling and propagation
    across functional boundaries.

    Type Parameters:
        T: The payload type carried upon successful completion.

    Attributes:
        success: Whether the result represents a successful operation.
        error: Human-readable failure description, or ``None`` on success.
        error_code: Machine-readable identifier for domain errors.
        error_data: Structured JSON-compatible context dictionary.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
        - :class:`flext_core._result.result.FlextResultComposed`: Composed base class.

    """

    @staticmethod
    @override
    def _create_success[V](value: V) -> p.Result[V]:
        """Construct a validated success result carrying the given payload.

        Args:
            value: The payload instance bound to the success result.

        Returns:
            p.Result[V]: A validated success result instance wrapping ``value``.

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
        """Construct a normalized failure result with immutable error context.

        Args:
            error: Descriptive failure message.
            error_code: Optional machine-readable error code.
            error_data: Optional structured JSON-serializable metadata.
            exception: Optional underlying exception that caused the failure.

        Returns:
            p.Result[Never]: A failure result instance with no success payload.

        """
        return FlextResult[Never].create_failure(
            error,
            error_code,
            error_data,
            exception,
        )


r = FlextResult
"""Canonical platform shorthand alias for ``FlextResult``.

Supports concise railway-oriented expressions such as ``r.ok(val)`` and
``r.fail(msg)``.
"""

__all__: list[str] = ["FlextResult", "r"]
