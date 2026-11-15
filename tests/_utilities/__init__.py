# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

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
    deferred_members={
        "TestsFlextUtilitiesContracts": (
            "Contract",
            "assert_safe_string_valid",
            "assert_format_app_id",
        ),
        "TestsFlextUtilitiesDispatch": (
            "handler_module",
            "HandlerService",
            "InheritedHandlerService",
            "PrivateHandlerService",
            "DescriptorHandlerService",
            "CapabilityHandler",
            "AutoSubscriber",
            "Handler",
            "Middleware",
        ),
        "TestsFlextUtilitiesParserReliability": (),
        "TestsFlextUtilitiesParserScenarios": (
            "PUBLIC_PARSE_CASES",
            "LDIF_PARSE_SCENARIOS",
        ),
        "TestsFlextUtilitiesRailway": (),
        "TestsFlextUtilitiesRailwayCases": (
            "success_cases",
            "failure_cases",
            "railway_success_cases",
            "multi_operation_cases",
        ),
        "TestsFlextUtilitiesRailwayPipelines": (
            "_v1_email_view",
            "_v1_send_email",
            "_v1_identity_response",
            "_v1_status_view",
            "execute_v1_pipeline",
            "execute_v2_pipeline",
        ),
        "TestsFlextUtilitiesRailwayServices": (
            "GetUserService",
            "SendEmailService",
            "value_lt_100",
            "make",
            "create_user_service",
        ),
        "TestsFlextUtilitiesReliabilityScenarios": (
            "_RETRY_BASE_SETTINGS",
            "_RETRY_EXHAUSTED_SETTINGS",
            "RETRY_SCENARIOS",
            "CIRCUIT_BREAKER_SCENARIOS",
        ),
        "TestsFlextUtilitiesResultCombinators": (
            "double",
            "is_even",
            "is_odd",
            "increment",
            "shout",
            "forty_two",
            "zero",
        ),
        "TestsFlextUtilitiesServices": (
            "MemoryCounter",
            "CountingService",
            "ValidatingService",
            "GetUserServiceAuto",
            "ValidatingServiceAuto",
        ),
        "TestsFlextUtilitiesValidationFactories": (
            "input_reader",
            "validated_label",
        ),
        "TestsFlextUtilitiesValidationNetworkScenarios": (
            "PORT_SCENARIOS",
            "HOSTNAME_SCENARIOS",
        ),
        "TestsFlextUtilitiesValidationNumericScenarios": (
            "NON_NEGATIVE_SCENARIOS",
            "POSITIVE_SCENARIOS",
            "RANGE_SCENARIOS",
        ),
        "TestsFlextUtilitiesValidationPatternScenarios": (
            "LENGTH_SCENARIOS",
            "PATTERN_SCENARIOS",
        ),
        "TestsFlextUtilitiesValidationStringScenarios": (
            "REQUIRED_SCENARIOS",
            "CHOICE_SCENARIOS",
        ),
        "TestsFlextUtilitiesValidationUriScenarios": ("URI_SCENARIOS",),
    },
)

defer_lazy_bases(
    __name__,
    {
        "TestsFlextUtilitiesParserReliability": (
            (".", "TestsFlextUtilitiesParserScenarios"),
            (".", "TestsFlextUtilitiesReliabilityScenarios"),
        ),
        "TestsFlextUtilitiesRailway": (
            (".", "TestsFlextUtilitiesRailwayPipelines"),
            (".", "TestsFlextUtilitiesRailwayServices"),
            (".", "TestsFlextUtilitiesRailwayCases"),
        ),
        "TestsFlextUtilitiesRailwayPipelines": (
            (".", "TestsFlextUtilitiesRailwayServices"),
        ),
        "TestsFlextUtilitiesValidationFactories": (
            (".", "TestsFlextUtilitiesServices"),
        ),
    },
)
