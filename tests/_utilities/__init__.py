# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._utilities.beartype_engine import TestsFlextBeartypeEngine
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
    from tests._utilities.services import TestsFlextUtilitiesServicesMixin
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
    "TestsFlextUtilitiesServicesMixin",
    "TestsFlextUtilitiesValidationFactoriesMixin",
    "TestsFlextUtilitiesValidationNetworkScenarios",
    "TestsFlextUtilitiesValidationNumericScenarios",
    "TestsFlextUtilitiesValidationPatternScenarios",
    "TestsFlextUtilitiesValidationStringScenarios",
    "TestsFlextUtilitiesValidationUriScenarios",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextBeartypeEngine": ".beartype_engine",
        "TestsFlextDecoratorsLegacy": ".decorators",
        "TestsFlextFlextHandlers": ".handlers",
        "TestsFlextResultExceptionCarrying": ".result_exception_carrying",
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
        "TestsFlextUtilitiesServicesMixin": ".services",
        "TestsFlextUtilitiesValidationFactoriesMixin": ".validation_factories",
        "TestsFlextUtilitiesValidationNetworkScenarios": ".validation_network",
        "TestsFlextUtilitiesValidationNumericScenarios": ".validation_numeric",
        "TestsFlextUtilitiesValidationPatternScenarios": ".validation_pattern",
        "TestsFlextUtilitiesValidationStringScenarios": ".validation_string",
        "TestsFlextUtilitiesValidationUriScenarios": ".validation_uri",
    }),
    public_exports=__all__,
)
