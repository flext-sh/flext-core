# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin",),
        ".case_generators": ("TestsFlextUtilitiesCaseGeneratorsMixin",),
        ".case_service_factories": ("TestsFlextUtilitiesCaseServiceFactoriesMixin",),
        ".contracts": ("TestsFlextUtilitiesContractsMixin",),
        ".dispatch": ("TestsFlextUtilitiesDispatchMixin",),
        ".parser_reliability": ("TestsFlextUtilitiesParserReliabilityMixin",),
        ".parser_scenarios": ("TestsFlextUtilitiesParserScenariosMixin",),
        ".railway": ("TestsFlextUtilitiesRailwayMixin",),
        ".railway_cases": ("TestsFlextUtilitiesRailwayCasesMixin",),
        ".railway_pipelines": ("TestsFlextUtilitiesRailwayPipelinesMixin",),
        ".railway_services": ("TestsFlextUtilitiesRailwayServicesMixin",),
        ".reliability_scenarios": ("TestsFlextUtilitiesReliabilityScenariosMixin",),
        ".service_factories": ("TestsFlextUtilitiesServiceFactoriesMixin",),
        ".services": ("TestsFlextUtilitiesServicesMixin",),
        ".user_factories": ("TestsFlextUtilitiesUserFactoriesMixin",),
        ".validation_factories": ("TestsFlextUtilitiesValidationFactoriesMixin",),
        ".validation_network": ("TestsFlextUtilitiesValidationNetworkScenarios",),
        ".validation_numeric": ("TestsFlextUtilitiesValidationNumericScenarios",),
        ".validation_pattern": ("TestsFlextUtilitiesValidationPatternScenarios",),
        ".validation_scenarios": ("TestsFlextUtilitiesValidationScenariosMixin",),
        ".validation_string": ("TestsFlextUtilitiesValidationStringScenarios",),
        ".validation_uri": ("TestsFlextUtilitiesValidationUriScenarios",),
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
