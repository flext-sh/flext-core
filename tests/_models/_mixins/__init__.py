# AUTO-GENERATED FILE — Regenerate with: make gen
"""Mixins package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core.tests._models._mixins.container import (
        TestsFlextModelsContainerMixin as TestsFlextModelsContainerMixin,
    )
    from flext_core.tests._models._mixins.core import (
        TestsFlextModelsCoreMixin as TestsFlextModelsCoreMixin,
    )
    from flext_core.tests._models._mixins.core_errors import (
        TestsFlextModelsCoreErrorsMixin as TestsFlextModelsCoreErrorsMixin,
    )
    from flext_core.tests._models._mixins.core_public import (
        TestsFlextModelsCorePublicMixin as TestsFlextModelsCorePublicMixin,
    )
    from flext_core.tests._models._mixins.core_state import (
        TestsFlextModelsCoreStateMixin as TestsFlextModelsCoreStateMixin,
    )
    from flext_core.tests._models._mixins.domain import (
        TestsFlextModelsDomainMixin as TestsFlextModelsDomainMixin,
    )
    from flext_core.tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin as TestsFlextModelsFixturePayloadsMixin,
    )
    from flext_core.tests._models._mixins.fixture_suite import (
        TestsFlextModelsFixtureSuiteMixin as TestsFlextModelsFixtureSuiteMixin,
    )
    from flext_core.tests._models._mixins.fixtures import (
        TestsFlextModelsFixtureDictsMixin as TestsFlextModelsFixtureDictsMixin,
    )
    from flext_core.tests._models._mixins.guards_mapper import (
        TestsFlextModelsGuardsMapperMixin as TestsFlextModelsGuardsMapperMixin,
    )
    from flext_core.tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin as TestsFlextModelsServiceCaseCoreMixin,
    )
    from flext_core.tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin as TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from flext_core.tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin as TestsFlextModelsServiceCaseValidationMixin,
    )
    from flext_core.tests._models._mixins.service_cases import (
        TestsFlextModelsServiceCasesMixin as TestsFlextModelsServiceCasesMixin,
    )
    from flext_core.tests._models._mixins.test_data import (
        TestsFlextModelsTestDataMixin as TestsFlextModelsTestDataMixin,
    )
    from flext_core.tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin as TestsFlextModelsTestDataIdentityMixin,
    )
    from flext_core.tests._models._mixins.test_data_values import (
        TestsFlextModelsTestDataValuesMixin as TestsFlextModelsTestDataValuesMixin,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".container": ("TestsFlextModelsContainerMixin",),
        ".core": ("TestsFlextModelsCoreMixin",),
        ".core_errors": ("TestsFlextModelsCoreErrorsMixin",),
        ".core_public": ("TestsFlextModelsCorePublicMixin",),
        ".core_state": ("TestsFlextModelsCoreStateMixin",),
        ".domain": ("TestsFlextModelsDomainMixin",),
        ".fixture_payloads": ("TestsFlextModelsFixturePayloadsMixin",),
        ".fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
        ".fixtures": ("TestsFlextModelsFixtureDictsMixin",),
        ".guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
        ".service_case_core": ("TestsFlextModelsServiceCaseCoreMixin",),
        ".service_case_reliability": ("TestsFlextModelsServiceCaseReliabilityMixin",),
        ".service_case_validation": ("TestsFlextModelsServiceCaseValidationMixin",),
        ".service_cases": ("TestsFlextModelsServiceCasesMixin",),
        ".test_data": ("TestsFlextModelsTestDataMixin",),
        ".test_data_identity": ("TestsFlextModelsTestDataIdentityMixin",),
        ".test_data_values": ("TestsFlextModelsTestDataValuesMixin",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
