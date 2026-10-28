"""Static test data model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._models._mixins import (
    TestsFlextModelsTestDataIdentityMixin,
    TestsFlextModelsTestDataValuesMixin,
)


class TestsFlextModelsTestDataMixin(
    TestsFlextModelsTestDataValuesMixin,
    TestsFlextModelsTestDataIdentityMixin,
):
    """Static test data model helpers."""


__all__: list[str] = ["TestsFlextModelsTestDataMixin"]
