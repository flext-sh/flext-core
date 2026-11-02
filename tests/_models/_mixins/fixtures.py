"""Fixture dictionary model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._models._mixins import (
    TestsFlextModelsFixturePayloads,
    TestsFlextModelsFixtureSuite,
)


class TestsFlextModelsFixtureDicts(
    TestsFlextModelsFixturePayloads,
    TestsFlextModelsFixtureSuite,
):
    """Fixture dictionary model helpers."""


__all__: list[str] = ["TestsFlextModelsFixtureDicts"]
