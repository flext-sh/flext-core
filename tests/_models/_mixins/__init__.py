# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models. Mixins package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from tests._models._mixins.container import TestsFlextModelsContainerMixin
    from tests._models._mixins.core import TestsFlextModelsCoreMixin
    from tests._models._mixins.core_errors import TestsFlextModelsCoreErrorsMixin
    from tests._models._mixins.core_public import TestsFlextModelsCorePublicMixin
    from tests._models._mixins.core_state import TestsFlextModelsCoreStateMixin
    from tests._models._mixins.domain import TestsFlextModelsDomainMixin
    from tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin,
    )
    from tests._models._mixins.fixture_suite import TestsFlextModelsFixtureSuiteMixin
    from tests._models._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
    from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
    from tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin,
    )
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin,
    )
    from tests._models._mixins.service_cases import TestsFlextModelsServiceCasesMixin
    from tests._models._mixins.test_data import TestsFlextModelsTestDataMixin
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin,
    )
    from tests._models._mixins.test_data_values import (
        TestsFlextModelsTestDataValuesMixin,
    )


__all__: tuple[str, ...] = (
    "TestsFlextModelsContainerMixin",
    "TestsFlextModelsCoreErrorsMixin",
    "TestsFlextModelsCoreMixin",
    "TestsFlextModelsCorePublicMixin",
    "TestsFlextModelsCoreStateMixin",
    "TestsFlextModelsDomainMixin",
    "TestsFlextModelsFixtureDictsMixin",
    "TestsFlextModelsFixturePayloadsMixin",
    "TestsFlextModelsFixtureSuiteMixin",
    "TestsFlextModelsGuardsMapperMixin",
    "TestsFlextModelsServiceCaseCoreMixin",
    "TestsFlextModelsServiceCaseReliabilityMixin",
    "TestsFlextModelsServiceCaseValidationMixin",
    "TestsFlextModelsServiceCasesMixin",
    "TestsFlextModelsTestDataIdentityMixin",
    "TestsFlextModelsTestDataMixin",
    "TestsFlextModelsTestDataValuesMixin",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextModelsContainerMixin": ".container",
        "TestsFlextModelsCoreErrorsMixin": ".core_errors",
        "TestsFlextModelsCoreMixin": ".core",
        "TestsFlextModelsCorePublicMixin": ".core_public",
        "TestsFlextModelsCoreStateMixin": ".core_state",
        "TestsFlextModelsDomainMixin": ".domain",
        "TestsFlextModelsFixtureDictsMixin": ".fixtures",
        "TestsFlextModelsFixturePayloadsMixin": ".fixture_payloads",
        "TestsFlextModelsFixtureSuiteMixin": ".fixture_suite",
        "TestsFlextModelsGuardsMapperMixin": ".guards_mapper",
        "TestsFlextModelsServiceCaseCoreMixin": ".service_case_core",
        "TestsFlextModelsServiceCaseReliabilityMixin": ".service_case_reliability",
        "TestsFlextModelsServiceCaseValidationMixin": ".service_case_validation",
        "TestsFlextModelsServiceCasesMixin": ".service_cases",
        "TestsFlextModelsTestDataIdentityMixin": ".test_data_identity",
        "TestsFlextModelsTestDataMixin": ".test_data",
        "TestsFlextModelsTestDataValuesMixin": ".test_data_values",
    }),
    public_exports=__all__,
)
