"""Service base for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_tests import s as tests_s

from tests import c
from tests import p, t

if TYPE_CHECKING:
    from tests import p


class TestsFlextServiceBase[TDomainResult: t.JsonPayload | t.SequenceOf[t.JsonPayload]](
    tests_s[TDomainResult],
):
    """Project-local test service base with flext-core result typing."""

    @override
    def execute(self) -> p.Result[TDomainResult]:
        """Execute domain service logic - must be implemented by subclasses."""
        msg = c.Tests.SUBCLASSES_MUST_IMPLEMENT_EXECUTE
        raise NotImplementedError(msg)


s = TestsFlextServiceBase

__all__: list[str] = ["TestsFlextServiceBase", "s"]
