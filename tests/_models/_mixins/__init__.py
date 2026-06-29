# AUTO-GENERATED FILE — Regenerate with: make gen
"""Mixins package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        u as u,
        x as x,
    )

    from tests._models._mixins.container import (
        TestsFlextModelsContainerMixin as TestsFlextModelsContainerMixin,
    )
    from tests._models._mixins.core import (
        TestsFlextModelsCoreMixin as TestsFlextModelsCoreMixin,
    )
    from tests._models._mixins.core_errors import (
        TestsFlextModelsCoreErrorsMixin as TestsFlextModelsCoreErrorsMixin,
    )
    from tests._models._mixins.core_public import (
        TestsFlextModelsCorePublicMixin as TestsFlextModelsCorePublicMixin,
    )
    from tests._models._mixins.core_state import (
        TestsFlextModelsCoreStateMixin as TestsFlextModelsCoreStateMixin,
    )
    from tests._models._mixins.domain import (
        TestsFlextModelsDomainMixin as TestsFlextModelsDomainMixin,
    )
    from tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin as TestsFlextModelsFixturePayloadsMixin,
    )
    from tests._models._mixins.fixture_suite import (
        TestsFlextModelsFixtureSuiteMixin as TestsFlextModelsFixtureSuiteMixin,
    )
    from tests._models._mixins.fixtures import (
        TestsFlextModelsFixtureDictsMixin as TestsFlextModelsFixtureDictsMixin,
    )
    from tests._models._mixins.guards_mapper import (
        TestsFlextModelsGuardsMapperMixin as TestsFlextModelsGuardsMapperMixin,
    )
    from tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin as TestsFlextModelsServiceCaseCoreMixin,
    )
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin as TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin as TestsFlextModelsServiceCaseValidationMixin,
    )
    from tests._models._mixins.service_cases import (
        TestsFlextModelsServiceCasesMixin as TestsFlextModelsServiceCasesMixin,
    )
    from tests._models._mixins.test_data import (
        TestsFlextModelsTestDataMixin as TestsFlextModelsTestDataMixin,
    )
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin as TestsFlextModelsTestDataIdentityMixin,
    )
    from tests._models._mixins.test_data_values import (
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
        "flext_tests": (
            "c",
            "d",
            "e",
            "h",
            "m",
            "p",
            "r",
            "s",
            "t",
            "td",
            "tf",
            "tk",
            "tm",
            "tv",
            "u",
            "x",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
