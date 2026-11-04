# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Protocols. Mixins package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._protocols._mixins.container import TestsFlextProtocolsMixinsContainer
    from tests._protocols._mixins.core_errors import TestsFlextProtocolsMixinsCoreErrors
    from tests._protocols._mixins.core_public import TestsFlextProtocolsMixinsCorePublic
    from tests._protocols._mixins.core_state import TestsFlextProtocolsMixinsCoreState
    from tests._protocols._mixins.domain import TestsFlextProtocolsMixinsDomain
    from tests._protocols._mixins.guards_mapper import (
        TestsFlextProtocolsMixinsGuardsMapper,
    )
    from tests._protocols._mixins.service_case_core import (
        TestsFlextProtocolsMixinsServiceCaseCore,
    )
    from tests._protocols._mixins.service_case_reliability import (
        TestsFlextProtocolsMixinsServiceCaseReliability,
    )
    from tests._protocols._mixins.service_case_validation import (
        TestsFlextProtocolsMixinsServiceCaseValidation,
    )
    from tests._protocols._mixins.test_data_identity import (
        TestsFlextProtocolsMixinsTestDataIdentity,
    )
    from tests._protocols._mixins.test_data_values import (
        TestsFlextProtocolsMixinsTestDataValues,
    )


__all__: tuple[str, ...] = (
    "TestsFlextProtocolsMixinsContainer",
    "TestsFlextProtocolsMixinsCoreErrors",
    "TestsFlextProtocolsMixinsCorePublic",
    "TestsFlextProtocolsMixinsCoreState",
    "TestsFlextProtocolsMixinsDomain",
    "TestsFlextProtocolsMixinsGuardsMapper",
    "TestsFlextProtocolsMixinsServiceCaseCore",
    "TestsFlextProtocolsMixinsServiceCaseReliability",
    "TestsFlextProtocolsMixinsServiceCaseValidation",
    "TestsFlextProtocolsMixinsTestDataIdentity",
    "TestsFlextProtocolsMixinsTestDataValues",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "TestsFlextProtocolsMixinsContainer": ".container",
        "TestsFlextProtocolsMixinsCoreErrors": ".core_errors",
        "TestsFlextProtocolsMixinsCorePublic": ".core_public",
        "TestsFlextProtocolsMixinsCoreState": ".core_state",
        "TestsFlextProtocolsMixinsDomain": ".domain",
        "TestsFlextProtocolsMixinsGuardsMapper": ".guards_mapper",
        "TestsFlextProtocolsMixinsServiceCaseCore": ".service_case_core",
        "TestsFlextProtocolsMixinsServiceCaseReliability": ".service_case_reliability",
        "TestsFlextProtocolsMixinsServiceCaseValidation": ".service_case_validation",
        "TestsFlextProtocolsMixinsTestDataIdentity": ".test_data_identity",
        "TestsFlextProtocolsMixinsTestDataValues": ".test_data_values",
    },
    public_exports=__all__,
)
