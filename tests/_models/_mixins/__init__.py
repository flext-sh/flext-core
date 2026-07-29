# AUTO-GENERATED FILE — Regenerate with: make gen
"""Mixins package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map({
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
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
