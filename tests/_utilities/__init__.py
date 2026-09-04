# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .case_factories import TestsFlextUtilitiesCaseFactoriesMixin
    from .case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
    from .case_service_factories import TestsFlextUtilitiesCaseServiceFactoriesMixin
    from .contracts import TestsFlextUtilitiesContractsMixin
    from .dispatch import TestsFlextUtilitiesDispatchMixin
    from .parser_reliability import TestsFlextUtilitiesParserReliabilityMixin
    from .parser_scenarios import TestsFlextUtilitiesParserScenariosMixin
    from .railway import TestsFlextUtilitiesRailwayMixin
    from .railway_cases import TestsFlextUtilitiesRailwayCasesMixin
    from .railway_pipelines import TestsFlextUtilitiesRailwayPipelinesMixin
    from .railway_services import TestsFlextUtilitiesRailwayServicesMixin
    from .reliability_scenarios import TestsFlextUtilitiesReliabilityScenariosMixin
    from .service_factories import TestsFlextUtilitiesServiceFactoriesMixin
    from .services import TestsFlextUtilitiesServicesMixin
    from .user_factories import TestsFlextUtilitiesUserFactoriesMixin
    from .validation_factories import TestsFlextUtilitiesValidationFactoriesMixin
    from .validation_network import TestsFlextUtilitiesValidationNetworkScenarios
    from .validation_numeric import TestsFlextUtilitiesValidationNumericScenarios
    from .validation_pattern import TestsFlextUtilitiesValidationPatternScenarios
    from .validation_scenarios import TestsFlextUtilitiesValidationScenariosMixin
    from .validation_string import TestsFlextUtilitiesValidationStringScenarios
    from .validation_uri import TestsFlextUtilitiesValidationUriScenarios
__all__: tuple[str, ...] = (
    "TestsFlextUtilitiesCaseFactoriesMixin",
    "TestsFlextUtilitiesCaseGeneratorsMixin",
    "TestsFlextUtilitiesCaseServiceFactoriesMixin",
    "TestsFlextUtilitiesContractsMixin",
    "TestsFlextUtilitiesDispatchMixin",
    "TestsFlextUtilitiesParserReliabilityMixin",
    "TestsFlextUtilitiesParserScenariosMixin",
    "TestsFlextUtilitiesRailwayCasesMixin",
    "TestsFlextUtilitiesRailwayMixin",
    "TestsFlextUtilitiesRailwayPipelinesMixin",
    "TestsFlextUtilitiesRailwayServicesMixin",
    "TestsFlextUtilitiesReliabilityScenariosMixin",
    "TestsFlextUtilitiesServiceFactoriesMixin",
    "TestsFlextUtilitiesServicesMixin",
    "TestsFlextUtilitiesUserFactoriesMixin",
    "TestsFlextUtilitiesValidationFactoriesMixin",
    "TestsFlextUtilitiesValidationNetworkScenarios",
    "TestsFlextUtilitiesValidationNumericScenarios",
    "TestsFlextUtilitiesValidationPatternScenarios",
    "TestsFlextUtilitiesValidationScenariosMixin",
    "TestsFlextUtilitiesValidationStringScenarios",
    "TestsFlextUtilitiesValidationUriScenarios",
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
            ".case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin",),
            ".case_generators": ("TestsFlextUtilitiesCaseGeneratorsMixin",),
            ".case_service_factories": (
                "TestsFlextUtilitiesCaseServiceFactoriesMixin",
            ),
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
