# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._utilities.contracts import TestsFlextUtilitiesContracts
    from tests._utilities.decorators import TestsFlextDecoratorsLegacy
    from tests._utilities.dispatch import TestsFlextUtilitiesDispatch
    from tests._utilities.handlers import TestsFlextFlextHandlers
    from tests._utilities.parser_reliability import TestsFlextUtilitiesParserReliability
    from tests._utilities.parser_scenarios import TestsFlextUtilitiesParserScenarios
    from tests._utilities.railway import TestsFlextUtilitiesRailway
    from tests._utilities.railway_cases import TestsFlextUtilitiesRailwayCases
    from tests._utilities.railway_pipelines import TestsFlextUtilitiesRailwayPipelines
    from tests._utilities.railway_services import TestsFlextUtilitiesRailwayServices
    from tests._utilities.reliability_scenarios import (
        TestsFlextUtilitiesReliabilityScenarios,
    )
    from tests._utilities.result_combinators import TestsFlextUtilitiesResultCombinators
    from tests._utilities.result_exception_carrying import (
        TestsFlextResultExceptionCarrying,
    )
    from tests._utilities.services import TestsFlextUtilitiesServices
    from tests._utilities.validation_factories import (
        TestsFlextUtilitiesValidationFactories,
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
    "TestsFlextDecoratorsLegacy",
    "TestsFlextFlextHandlers",
    "TestsFlextResultExceptionCarrying",
    "TestsFlextUtilitiesContracts",
    "TestsFlextUtilitiesDispatch",
    "TestsFlextUtilitiesParserReliability",
    "TestsFlextUtilitiesParserScenarios",
    "TestsFlextUtilitiesRailway",
    "TestsFlextUtilitiesRailwayCases",
    "TestsFlextUtilitiesRailwayPipelines",
    "TestsFlextUtilitiesRailwayServices",
    "TestsFlextUtilitiesReliabilityScenarios",
    "TestsFlextUtilitiesResultCombinators",
    "TestsFlextUtilitiesServices",
    "TestsFlextUtilitiesValidationFactories",
    "TestsFlextUtilitiesValidationNetworkScenarios",
    "TestsFlextUtilitiesValidationNumericScenarios",
    "TestsFlextUtilitiesValidationPatternScenarios",
    "TestsFlextUtilitiesValidationStringScenarios",
    "TestsFlextUtilitiesValidationUriScenarios",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "TestsFlextDecoratorsLegacy": ".decorators",
        "TestsFlextFlextHandlers": ".handlers",
        "TestsFlextResultExceptionCarrying": ".result_exception_carrying",
        "TestsFlextUtilitiesContracts": ".contracts",
        "TestsFlextUtilitiesDispatch": ".dispatch",
        "TestsFlextUtilitiesParserReliability": ".parser_reliability",
        "TestsFlextUtilitiesParserScenarios": ".parser_scenarios",
        "TestsFlextUtilitiesRailway": ".railway",
        "TestsFlextUtilitiesRailwayCases": ".railway_cases",
        "TestsFlextUtilitiesRailwayPipelines": ".railway_pipelines",
        "TestsFlextUtilitiesRailwayServices": ".railway_services",
        "TestsFlextUtilitiesReliabilityScenarios": ".reliability_scenarios",
        "TestsFlextUtilitiesResultCombinators": ".result_combinators",
        "TestsFlextUtilitiesServices": ".services",
        "TestsFlextUtilitiesValidationFactories": ".validation_factories",
        "TestsFlextUtilitiesValidationNetworkScenarios": ".validation_network",
        "TestsFlextUtilitiesValidationNumericScenarios": ".validation_numeric",
        "TestsFlextUtilitiesValidationPatternScenarios": ".validation_pattern",
        "TestsFlextUtilitiesValidationStringScenarios": ".validation_string",
        "TestsFlextUtilitiesValidationUriScenarios": ".validation_uri",
    },
    public_exports=__all__,
)
