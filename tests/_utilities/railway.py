"""Railway helper namespace for flext-core tests."""

from __future__ import annotations

from .railway_cases import TestsFlextUtilitiesRailwayCasesMixin
from .railway_pipelines import TestsFlextUtilitiesRailwayPipelinesMixin
from .railway_services import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesRailwayMixin(
    TestsFlextUtilitiesRailwayPipelinesMixin,
    TestsFlextUtilitiesRailwayServicesMixin,
    TestsFlextUtilitiesRailwayCasesMixin,
):
    """Railway helpers."""


__all__: list[str] = ["TestsFlextUtilitiesRailwayMixin"]
