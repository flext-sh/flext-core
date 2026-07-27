"""Service case factory helper namespace."""

from __future__ import annotations

from tests._utilities.case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
from tests._utilities.case_service_factories import (
    TestsFlextUtilitiesCaseServiceFactoriesMixin,
)


class TestsFlextUtilitiesCaseFactoriesMixin(
    TestsFlextUtilitiesCaseGeneratorsMixin, TestsFlextUtilitiesCaseServiceFactoriesMixin
):
    """Service case factory helpers."""


__all__: list[str] = ["TestsFlextUtilitiesCaseFactoriesMixin"]
