# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin, u
    from tests._utilities.case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
    from tests._utilities.case_service_factories import (
        TestsFlextUtilitiesCaseServiceFactoriesMixin,
    )
    from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
    from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
    from tests._utilities.parser_reliability import (
        TestsFlextUtilitiesParserReliabilityMixin,
    )
    from tests._utilities.parser_scenarios import (
        TestsFlextUtilitiesParserScenariosMixin,
    )
    from tests._utilities.railway import TestsFlextUtilitiesRailwayMixin
    from tests._utilities.railway_cases import TestsFlextUtilitiesRailwayCasesMixin
    from tests._utilities.railway_pipelines import (
        TestsFlextUtilitiesRailwayPipelinesMixin,
    )
    from tests._utilities.railway_services import (
        TestsFlextUtilitiesRailwayServicesMixin,
    )
    from tests._utilities.reliability_scenarios import (
        TestsFlextUtilitiesReliabilityScenariosMixin,
    )
    from tests._utilities.service_factories import (
        TestsFlextUtilitiesServiceFactoriesMixin,
    )
    from tests._utilities.services import TestsFlextUtilitiesServicesMixin
    from tests._utilities.user_factories import TestsFlextUtilitiesUserFactoriesMixin
    from tests._utilities.validation_factories import (
        TestsFlextUtilitiesValidationFactoriesMixin,
    )
    from tests._utilities.validation_network import (
        TestsFlextUtilitiesValidationNetworkScenarios,
    )
    from tests._utilities.validation_numeric import (
        TestsFlextUtilitiesValidationNumericScenarios,
    )
    from tests._utilities.validation_pattern import (
        TestsFlextUtilitiesValidationPatternScenarios,
    )
    from tests._utilities.validation_string import (
        TestsFlextUtilitiesValidationStringScenarios,
    )
    from tests._utilities.validation_uri import (
        TestsFlextUtilitiesValidationUriScenarios,
    )


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
    "TestsFlextUtilitiesValidationStringScenarios",
    "TestsFlextUtilitiesValidationUriScenarios",
    "u",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin", "u"),
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
            ".validation_string": ("TestsFlextUtilitiesValidationStringScenarios",),
            ".validation_uri": ("TestsFlextUtilitiesValidationUriScenarios",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
