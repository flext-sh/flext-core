"""Immutability matcher — assignment rejection for frozen and constant surfaces.

Owns the dynamic attribute assignment that a negative immutability test performs.
Holding it here keeps the mechanism in library code, so test suites assert
immutability without spelling a constant-name assignment at every call site.
"""

from __future__ import annotations

import re


class FlextTestsMatchersImmutabilityMixin:
    """Assignment-rejection matcher exposed under ``Tests.Matchers``."""

    class Tests:
        """Container for test utility storages and aliases."""

        class Matchers:
            """Test matchers with powerful generalist methods."""

            @staticmethod
            def rejects_assignment(
                target: object,
                field: str,
                value: object,
                *,
                expected: type[BaseException] | tuple[type[BaseException], ...],
                match: str | None = None,
            ) -> None:
                """Assert assigning ``value`` to ``field`` on ``target`` is rejected.

                Args:
                    target: Instance or class under test.
                    field: Attribute name the assignment targets.
                    value: Value the target must refuse.
                    expected: Exception type(s) the rejection must raise.
                    match: Optional regex the rejection message must contain.

                Raises:
                    AssertionError: If the assignment is accepted, or the raised
                        error does not match ``expected`` / ``match``.

                """
                label = getattr(target, "__name__", type(target).__name__)
                try:
                    setattr(target, field, value)
                except expected as exc:
                    if match is not None and not re.search(match, str(exc)):
                        message = (
                            f"{label}.{field} rejected the assignment but the "
                            f"reason did not match {match!r}: {exc}"
                        )
                        raise AssertionError(message) from exc
                    return
                message = (
                    f"{label}.{field} accepted the assignment {value!r}; "
                    f"the surface is mutable"
                )
                raise AssertionError(message)


__all__: list[str] = ["FlextTestsMatchersImmutabilityMixin"]
