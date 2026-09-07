# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models. Mixins package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .container import TestsFlextModelsContainerMixin
    from .core import TestsFlextModelsCoreMixin
    from .core_errors import TestsFlextModelsCoreErrorsMixin
    from .core_public import TestsFlextModelsCorePublicMixin
    from .core_state import TestsFlextModelsCoreStateMixin
    from .domain import TestsFlextModelsDomainMixin
    from .fixture_payloads import TestsFlextModelsFixturePayloadsMixin
    from .fixture_suite import TestsFlextModelsFixtureSuiteMixin
    from .fixtures import TestsFlextModelsFixtureDictsMixin
    from .guards_mapper import TestsFlextModelsGuardsMapperMixin
    from .service_case_core import TestsFlextModelsServiceCaseCoreMixin
    from .service_case_reliability import TestsFlextModelsServiceCaseReliabilityMixin
    from .service_case_validation import TestsFlextModelsServiceCaseValidationMixin
    from .service_cases import TestsFlextModelsServiceCasesMixin
    from .test_data import TestsFlextModelsTestDataMixin
    from .test_data_identity import TestsFlextModelsTestDataIdentityMixin
    from .test_data_values import TestsFlextModelsTestDataValuesMixin
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
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
            ".service_case_reliability": (
                "TestsFlextModelsServiceCaseReliabilityMixin",
            ),
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
