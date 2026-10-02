"""Service base for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_tests import FlextTestsServiceBase as _FlextTestsServiceBase

from tests import c

if TYPE_CHECKING:
    from tests import p


class TestsFlextServiceBase[TDomainResult: p.Base = p.Base](
    _FlextTestsServiceBase[TDomainResult],
):
    """Project-local test service base with flext-core result typing."""

    @override
    def execute(self) -> p.Result[TDomainResult]:
        """Execute domain service logic - must be implemented by subclasses."""
        msg = c.Tests.SUBCLASSES_MUST_IMPLEMENT_EXECUTE
        raise NotImplementedError(msg)


s = TestsFlextServiceBase

__all__: list[str] = ["TestsFlextServiceBase", "s"]
