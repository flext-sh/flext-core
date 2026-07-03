"""Lazy export map part 01."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_01 = build_lazy_import_map({
    "._constants.domain": ("TestsFlextConstantsDomain",),
    "._constants.errors": ("TestsFlextConstantsErrors",),
    "._constants.fixtures": ("TestsFlextConstantsFixtures",),
    "._constants.loggings": ("TestsFlextConstantsLoggings",),
    "._constants.other": ("TestsFlextConstantsOther",),
    "._constants.result": ("TestsFlextConstantsResult",),
    "._constants.services": ("TestsFlextConstantsServices",),
    "._constants.settings": ("TestsFlextConstantsSettings",),
    "._models._mixins.container": ("TestsFlextModelsContainerMixin",),
    "._models._mixins.core": ("TestsFlextModelsCoreMixin",),
    "._models._mixins.core_errors": ("TestsFlextModelsCoreErrorsMixin",),
    "._models._mixins.core_public": ("TestsFlextModelsCorePublicMixin",),
    "._models._mixins.core_state": ("TestsFlextModelsCoreStateMixin",),
    "._models._mixins.domain": ("TestsFlextModelsDomainMixin",),
    "._models._mixins.fixture_payloads": ("TestsFlextModelsFixturePayloadsMixin",),
    "._models._mixins.fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
    "._models._mixins.fixtures": ("TestsFlextModelsFixtureDictsMixin",),
    "._models._mixins.guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
    "._models._mixins.service_case_core": ("TestsFlextModelsServiceCaseCoreMixin",),
    "._models._mixins.service_case_reliability": (
        "TestsFlextModelsServiceCaseReliabilityMixin",
    ),
    "._models._mixins.service_case_validation": (
        "TestsFlextModelsServiceCaseValidationMixin",
    ),
    "._models._mixins.service_cases": ("TestsFlextModelsServiceCasesMixin",),
    "._models._mixins.test_data": ("TestsFlextModelsTestDataMixin",),
    "._models._mixins.test_data_identity": ("TestsFlextModelsTestDataIdentityMixin",),
    "._models._mixins.test_data_values": ("TestsFlextModelsTestDataValuesMixin",),
    "._models.mixins": ("TestsFlextModelsMixins",),
    "._utilities.case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin",),
    "._utilities.case_generators": ("TestsFlextUtilitiesCaseGeneratorsMixin",),
    "._utilities.case_service_factories": (
        "TestsFlextUtilitiesCaseServiceFactoriesMixin",
    ),
    "._utilities.contracts": ("TestsFlextUtilitiesContractsMixin",),
    "._utilities.dispatch": ("TestsFlextUtilitiesDispatchMixin",),
    "._utilities.parser_reliability": ("TestsFlextUtilitiesParserReliabilityMixin",),
    "._utilities.parser_scenarios": ("TestsFlextUtilitiesParserScenariosMixin",),
    "._utilities.railway": ("TestsFlextUtilitiesRailwayMixin",),
    "._utilities.railway_cases": ("TestsFlextUtilitiesRailwayCasesMixin",),
    "._utilities.railway_pipelines": ("TestsFlextUtilitiesRailwayPipelinesMixin",),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_01"]
