"""Railway helper namespace for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from tests._utilities import (
    TestsFlextUtilitiesRailwayCasesMixin,
    TestsFlextUtilitiesRailwayPipelinesMixin,
    TestsFlextUtilitiesRailwayServicesMixin,
)


class TestsFlextUtilitiesRailwayMixin(
    TestsFlextUtilitiesRailwayPipelinesMixin,
    TestsFlextUtilitiesRailwayServicesMixin,
    TestsFlextUtilitiesRailwayCasesMixin,
):
    """Railway helpers."""


__all__: list[str] = ["TestsFlextUtilitiesRailwayMixin"]
