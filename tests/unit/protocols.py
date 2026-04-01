from __future__ import annotations

from tests.protocols import FlextCoreTestProtocols


class FlextUnitTestProtocols(FlextCoreTestProtocols):
    """Unit test protocols — inherits full MRO from FlextCoreTestProtocols."""


p = FlextUnitTestProtocols
__all__ = ["FlextUnitTestProtocols", "p"]
