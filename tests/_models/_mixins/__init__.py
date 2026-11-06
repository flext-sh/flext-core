# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models. Mixins package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._models._mixins.container import TestsFlextModelsContainer
    from tests._models._mixins.core import TestsFlextModelsCore
    from tests._models._mixins.core_errors import TestsFlextModelsCoreErrors
    from tests._models._mixins.core_public import TestsFlextModelsCorePublic
    from tests._models._mixins.core_state import TestsFlextModelsCoreState
    from tests._models._mixins.domain import TestsFlextModelsDomain
    from tests._models._mixins.fixture_payloads import TestsFlextModelsFixturePayloads
    from tests._models._mixins.fixture_suite import TestsFlextModelsFixtureSuite
    from tests._models._mixins.fixtures import TestsFlextModelsFixtureDicts
    from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapper
    from tests._models._mixins.schema_owner import TestsFlextModelsSchemaOwner
    from tests._models._mixins.schema_owner_heir import TestsFlextModelsSchemaOwnerHeir
    from tests._models._mixins.service_case_core import TestsFlextModelsServiceCaseCore
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliability,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidation,
    )
    from tests._models._mixins.service_cases import TestsFlextModelsServiceCases
    from tests._models._mixins.test_data import TestsFlextModelsTestData
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentity,
    )
    from tests._models._mixins.test_data_values import TestsFlextModelsTestDataValues


__all__: tuple[str, ...] = (
    "TestsFlextModelsContainer",
    "TestsFlextModelsCore",
    "TestsFlextModelsCoreErrors",
    "TestsFlextModelsCorePublic",
    "TestsFlextModelsCoreState",
    "TestsFlextModelsDomain",
    "TestsFlextModelsFixtureDicts",
    "TestsFlextModelsFixturePayloads",
    "TestsFlextModelsFixtureSuite",
    "TestsFlextModelsGuardsMapper",
    "TestsFlextModelsSchemaOwner",
    "TestsFlextModelsSchemaOwnerHeir",
    "TestsFlextModelsServiceCaseCore",
    "TestsFlextModelsServiceCaseReliability",
    "TestsFlextModelsServiceCaseValidation",
    "TestsFlextModelsServiceCases",
    "TestsFlextModelsTestData",
    "TestsFlextModelsTestDataIdentity",
    "TestsFlextModelsTestDataValues",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "TestsFlextModelsContainer": ".container",
        "TestsFlextModelsCore": ".core",
        "TestsFlextModelsCoreErrors": ".core_errors",
        "TestsFlextModelsCorePublic": ".core_public",
        "TestsFlextModelsCoreState": ".core_state",
        "TestsFlextModelsDomain": ".domain",
        "TestsFlextModelsFixtureDicts": ".fixtures",
        "TestsFlextModelsFixturePayloads": ".fixture_payloads",
        "TestsFlextModelsFixtureSuite": ".fixture_suite",
        "TestsFlextModelsGuardsMapper": ".guards_mapper",
        "TestsFlextModelsSchemaOwner": ".schema_owner",
        "TestsFlextModelsSchemaOwnerHeir": ".schema_owner_heir",
        "TestsFlextModelsServiceCaseCore": ".service_case_core",
        "TestsFlextModelsServiceCaseReliability": ".service_case_reliability",
        "TestsFlextModelsServiceCaseValidation": ".service_case_validation",
        "TestsFlextModelsServiceCases": ".service_cases",
        "TestsFlextModelsTestData": ".test_data",
        "TestsFlextModelsTestDataIdentity": ".test_data_identity",
        "TestsFlextModelsTestDataValues": ".test_data_values",
    },
    public_exports=__all__,
)
