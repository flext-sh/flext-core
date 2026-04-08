from __future__ import annotations

from tests import TestsFlextCoreProtocols


class TestsFlextUnitProtocols(TestsFlextCoreProtocols):
    """Unit test protocols — inherits full MRO from TestsFlextCoreProtocols."""


p = TestsFlextUnitProtocols
__all__ = ["TestsFlextUnitProtocols", "p"]
