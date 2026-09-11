"""Containment helpers for matchers.

Static methods used internally by ``FlextTestsMatchersUtilities``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_tests import c, p, t
from flext_tests._utilities._matchers._assertions import (
    FlextTestsMatchersAssertionsMixin,
)
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsMatchersContainmentMixin:
    """Shared ``has``/``lacks`` containment checks."""

    @staticmethod
    def check_has_lacks(
        value: p.AttributeProbe,
        has: p.AttributeProbe | None,
        lacks: p.AttributeProbe | None,
        msg: str | None,
        *,
        as_str: bool = False,
    ) -> None:
        """Shared has/lacks containment check for ok(), fail(), and that()."""
        if has is not None:
            has_items = (
                list(has)
                if isinstance(has, Sequence) and not isinstance(has, t.STR_BINARY_TYPES)
                else [has]
            )
            for item in has_items:
                if as_str:
                    check_str = str(item)
                    target = str(value)
                    if check_str not in target:
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_CONTAINS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )
                else:
                    check_val = FlextTestsPayloadUtilities.to_normalized_value(
                        FlextTestsPayloadUtilities.to_payload(item)
                    )
                    target_raw = FlextTestsPayloadUtilities.to_normalized_value(
                        FlextTestsPayloadUtilities.to_payload(value)
                    )
                    if not isinstance(target_raw, (Mapping, list, str)):
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_CONTAINS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )
                    contains_item = (
                        isinstance(check_val, str) and check_val in target_raw
                        if isinstance(target_raw, Mapping)
                        else str(check_val) in target_raw
                        if isinstance(target_raw, str)
                        else any(candidate == check_val for candidate in target_raw)
                    )
                    if not contains_item:
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_CONTAINS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )
        if lacks is not None:
            lacks_items = (
                list(lacks)
                if isinstance(lacks, Sequence)
                and not isinstance(lacks, t.STR_BINARY_TYPES)
                else [lacks]
            )
            for item in lacks_items:
                if as_str:
                    check_str = str(item)
                    target = str(value)
                    if check_str in target:
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_LACKS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )
                else:
                    check_val = FlextTestsPayloadUtilities.to_normalized_value(
                        FlextTestsPayloadUtilities.to_payload(item)
                    )
                    target_raw_2 = FlextTestsPayloadUtilities.to_normalized_value(
                        FlextTestsPayloadUtilities.to_payload(value)
                    )
                    if not isinstance(target_raw_2, (Mapping, list, str)):
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_LACKS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )
                    contains_item = (
                        isinstance(check_val, str) and check_val in target_raw_2
                        if isinstance(target_raw_2, Mapping)
                        else str(check_val) in target_raw_2
                        if isinstance(target_raw_2, str)
                        else any(candidate == check_val for candidate in target_raw_2)
                    )
                    if contains_item:
                        FlextTestsMatchersAssertionsMixin.raise_match_assertion(
                            c.Tests.ERR_LACKS_FAILED,
                            msg=msg,
                            container=value,
                            item=item,
                        )


__all__: list[str] = ["FlextTestsMatchersContainmentMixin"]
