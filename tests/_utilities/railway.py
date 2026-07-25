"""Railway helper namespace for flext-core tests."""

from __future__ import annotations

from tests._utilities.railway_cases import TestsFlextUtilitiesRailwayCasesMixin
from tests._utilities.railway_pipelines import TestsFlextUtilitiesRailwayPipelinesMixin
from tests._utilities.railway_services import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesRailwayMixin(
    TestsFlextUtilitiesRailwayPipelinesMixin,
    TestsFlextUtilitiesRailwayServicesMixin,
    TestsFlextUtilitiesRailwayCasesMixin,
):
    """Railway helpers."""


__all__: list[str] = ["TestsFlextUtilitiesRailwayMixin"]
