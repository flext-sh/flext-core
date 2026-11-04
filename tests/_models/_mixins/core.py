"""Core shared model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._models._mixins import (
    TestsFlextModelsCoreErrors,
    TestsFlextModelsCorePublic,
    TestsFlextModelsCoreState,
)


class TestsFlextModelsCore(
    TestsFlextModelsCoreState,
    TestsFlextModelsCoreErrors,
    TestsFlextModelsCorePublic,
):
    """Core shared model helpers."""


__all__: list[str] = ["TestsFlextModelsCore"]
