# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from tests._utilities.beartype_engine import TestsFlextBeartypeEngine
    from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin, u
    from tests._utilities.case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
    from tests._utilities.case_service_factories import (
        TestsFlextUtilitiesCaseServiceFactoriesMixin,
    )
    from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
    from tests._utilities.decorators import TestsFlextDecoratorsLegacy
    from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
    from tests._utilities.handlers import TestsFlextFlextHandlers
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
    from tests._utilities.result_combinators import TestsFlextUtilitiesResultCombinators
    from tests._utilities.result_exception_carrying import (
        TestsFlextResultExceptionCarrying,
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
    "TestsFlextBeartypeEngine",
    "TestsFlextDecoratorsLegacy",
    "TestsFlextFlextHandlers",
    "TestsFlextResultExceptionCarrying",
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
    "TestsFlextUtilitiesResultCombinators",
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

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextBeartypeEngine": ".beartype_engine",
        "TestsFlextDecoratorsLegacy": ".decorators",
        "TestsFlextFlextHandlers": ".handlers",
        "TestsFlextResultExceptionCarrying": ".result_exception_carrying",
        "TestsFlextUtilitiesCaseFactoriesMixin": ".case_factories",
        "TestsFlextUtilitiesCaseGeneratorsMixin": ".case_generators",
        "TestsFlextUtilitiesCaseServiceFactoriesMixin": ".case_service_factories",
        "TestsFlextUtilitiesContractsMixin": ".contracts",
        "TestsFlextUtilitiesDispatchMixin": ".dispatch",
        "TestsFlextUtilitiesParserReliabilityMixin": ".parser_reliability",
        "TestsFlextUtilitiesParserScenariosMixin": ".parser_scenarios",
        "TestsFlextUtilitiesRailwayCasesMixin": ".railway_cases",
        "TestsFlextUtilitiesRailwayMixin": ".railway",
        "TestsFlextUtilitiesRailwayPipelinesMixin": ".railway_pipelines",
        "TestsFlextUtilitiesRailwayServicesMixin": ".railway_services",
        "TestsFlextUtilitiesReliabilityScenariosMixin": ".reliability_scenarios",
        "TestsFlextUtilitiesResultCombinators": ".result_combinators",
        "TestsFlextUtilitiesServiceFactoriesMixin": ".service_factories",
        "TestsFlextUtilitiesServicesMixin": ".services",
        "TestsFlextUtilitiesUserFactoriesMixin": ".user_factories",
        "TestsFlextUtilitiesValidationFactoriesMixin": ".validation_factories",
        "TestsFlextUtilitiesValidationNetworkScenarios": ".validation_network",
        "TestsFlextUtilitiesValidationNumericScenarios": ".validation_numeric",
        "TestsFlextUtilitiesValidationPatternScenarios": ".validation_pattern",
        "TestsFlextUtilitiesValidationStringScenarios": ".validation_string",
        "TestsFlextUtilitiesValidationUriScenarios": ".validation_uri",
        "u": ".case_factories",
    }),
    public_exports=__all__,
)[0]
