"""Assertion-raising helpers for matchers — Group B.

Static methods used internally by ``FlextTestsMatchersUtilities``.
"""

from __future__ import annotations

from collections.abc import Sized
from typing import Never

from flext_tests import c, p, t


class FlextTestsMatchersAssertionsMixin:
    """Centralized AssertionError factories with structured messages."""

    @staticmethod
    def raise_match_assertion(
        template: str,
        *,
        msg: str | None,
        container: p.AttributeProbe,
        item: p.AttributeProbe,
    ) -> Never:
        """Raise AssertionError with ``msg`` or formatted ``template``."""
        raise AssertionError(msg or template.format(container=container, item=item))

    @staticmethod
    def assert_len_match(
        *,
        payload: t.Tests.TestobjectSerializable,
        sized: p.AttributeProbe,
        length_spec: int | tuple[int, int],
        msg: str | None,
    ) -> None:
        """Raise AssertionError if ``payload`` length doesn't match ``length_spec``."""
        if isinstance(payload, Sized):
            payload_len = len(payload)
            match length_spec:
                case int():
                    if payload_len == length_spec:
                        return
                case (min_len, max_len):
                    if min_len <= payload_len <= max_len:
                        return
        actual_len = len(sized) if isinstance(sized, Sized) else 0
        # mro-j47u: both declared matcher forms fail with assertion semantics.
        match length_spec:
            case int() as exact_length:
                raise AssertionError(
                    msg
                    or c.Tests.ERR_LEN_EXACT_FAILED.format(
                        expected=exact_length, actual=actual_len
                    )
                )
            case (min_length, max_length):
                raise AssertionError(
                    msg
                    or c.Tests.ERR_LEN_RANGE_FAILED.format(
                        min=min_length, max=max_length, actual=actual_len
                    )
                )


__all__: list[str] = ["FlextTestsMatchersAssertionsMixin"]
