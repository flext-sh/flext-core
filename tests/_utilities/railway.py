"""Railway helper namespace for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._utilities import (
    TestsFlextUtilitiesRailwayCases,
    TestsFlextUtilitiesRailwayPipelines,
    TestsFlextUtilitiesRailwayServices,
)


class TestsFlextUtilitiesRailway(
    TestsFlextUtilitiesRailwayPipelines,
    TestsFlextUtilitiesRailwayServices,
    TestsFlextUtilitiesRailwayCases,
):
    """Railway helpers."""


__all__: list[str] = ["TestsFlextUtilitiesRailway"]
