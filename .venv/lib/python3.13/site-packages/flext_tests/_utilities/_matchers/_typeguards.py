"""Type-guard helpers for matchers — Group A.

Static methods used internally by ``FlextTestsMatchersUtilities``. Composed
via MRO from ``flext_tests._utilities.matchers``.
"""

from __future__ import annotations

from flext_tests._typings.matchers import ApproxBase
from flext_infra import u
from flext_tests import c, p, t
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsMatchersTypeGuardsMixin:
    """Type compatibility and equality-guard helpers."""

    @staticmethod
    def matches_runtime_type(
        value: p.AttributeProbe, expected_type: type | tuple[type, ...]
    ) -> bool:
        """Check runtime type compatibility using canonical parent guards."""
        if isinstance(expected_type, tuple):
            return any(u.instance_of(value, item) for item in expected_type)
        return u.instance_of(value, expected_type)

    @staticmethod
    def prepare_eq_ne_payloads(
        actual_payload: t.Tests.TestobjectSerializable,
        eq_value: p.AttributeProbe | None,
        ne_value: p.AttributeProbe | None,
        *,
        msg: str | None,
        default_msg: str,
    ) -> tuple[
        t.Tests.TestobjectSerializable | None, t.Tests.TestobjectSerializable | None
    ]:
        if isinstance(eq_value, ApproxBase):
            if actual_payload != eq_value:
                raise AssertionError(msg or default_msg)
            eq_value = None
        if isinstance(ne_value, ApproxBase):
            if actual_payload == ne_value:
                raise AssertionError(msg or default_msg)
            ne_value = None
        eq_payload = (
            FlextTestsPayloadUtilities.to_payload(eq_value)
            if eq_value is not None
            else None
        )
        ne_payload = (
            FlextTestsPayloadUtilities.to_payload(ne_value)
            if ne_value is not None
            else None
        )
        if eq_payload is not None and not isinstance(
            eq_payload, c.Tests.MATCHER_GUARD_EQ_TYPES
        ):
            if actual_payload != eq_payload:
                raise AssertionError(msg or default_msg)
            eq_payload = None
        if ne_payload is not None and not isinstance(
            ne_payload, c.Tests.MATCHER_GUARD_EQ_TYPES
        ):
            if actual_payload == ne_payload:
                raise AssertionError(msg or default_msg)
            ne_payload = None
        return (eq_payload, ne_payload)


__all__: list[str] = ["FlextTestsMatchersTypeGuardsMixin"]
