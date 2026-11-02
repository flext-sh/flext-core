# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/mixins
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._protocols._mixins import (
    TestsFlextProtocolsMixinsContainer,
    TestsFlextProtocolsMixinsCoreErrors,
    TestsFlextProtocolsMixinsCorePublic,
    TestsFlextProtocolsMixinsCoreState,
    TestsFlextProtocolsMixinsDomain,
    TestsFlextProtocolsMixinsGuardsMapper,
    TestsFlextProtocolsMixinsServiceCaseCore,
    TestsFlextProtocolsMixinsServiceCaseReliability,
    TestsFlextProtocolsMixinsServiceCaseValidation,
    TestsFlextProtocolsMixinsTestDataIdentity,
    TestsFlextProtocolsMixinsTestDataValues,
)


class TestsFlextProtocolsMixins(
    TestsFlextProtocolsMixinsContainer,
    TestsFlextProtocolsMixinsCoreState,
    TestsFlextProtocolsMixinsCoreErrors,
    TestsFlextProtocolsMixinsCorePublic,
    TestsFlextProtocolsMixinsDomain,
    TestsFlextProtocolsMixinsGuardsMapper,
    TestsFlextProtocolsMixinsServiceCaseReliability,
    TestsFlextProtocolsMixinsServiceCaseValidation,
    TestsFlextProtocolsMixinsServiceCaseCore,
    TestsFlextProtocolsMixinsTestDataValues,
    TestsFlextProtocolsMixinsTestDataIdentity,
):
    """Own the structural contracts declared by its model source."""


__all__: list[str] = [
    "TestsFlextProtocolsMixins",
]
