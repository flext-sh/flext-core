"""Namespace enforcement tests."""

from __future__ import annotations

from tests.unit.test_enforcement_namespace_part_01 import (
    TestsFlextEnforcementNamespacePart01,
)
from tests.unit.test_enforcement_namespace_part_02 import (
    TestsFlextEnforcementNamespacePart02,
)


class TestsFlextEnforcementNamespace(
    TestsFlextEnforcementNamespacePart01,
    TestsFlextEnforcementNamespacePart02,
):
    """Namespace enforcement tests."""

    __test__ = True
