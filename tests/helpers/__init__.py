# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Helpers package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
    from tests.helpers import _scenarios_impl, factories, factories_impl, scenarios
    from tests.helpers._scenarios_impl import (
        ParserScenario,
        ParserScenarios,
        ReliabilityScenario,
        ReliabilityScenarios,
        ValidationScenario,
        ValidationScenarios,
    )
    from tests.helpers.factories import TestHelperFactories
    from tests.helpers.factories_impl import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from tests.helpers.scenarios import TestHelperScenarios

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "FailingService": "tests.helpers.factories_impl",
    "FailingServiceAuto": "tests.helpers.factories_impl",
    "FailingServiceAutoFactory": "tests.helpers.factories_impl",
    "FailingServiceFactory": "tests.helpers.factories_impl",
    "GenericModelFactory": "tests.helpers.factories_impl",
    "GetUserService": "tests.helpers.factories_impl",
    "GetUserServiceAuto": "tests.helpers.factories_impl",
    "GetUserServiceAutoFactory": "tests.helpers.factories_impl",
    "GetUserServiceFactory": "tests.helpers.factories_impl",
    "ParserScenario": "tests.helpers._scenarios_impl",
    "ParserScenarios": "tests.helpers._scenarios_impl",
    "ReliabilityScenario": "tests.helpers._scenarios_impl",
    "ReliabilityScenarios": "tests.helpers._scenarios_impl",
    "ServiceFactoryRegistry": "tests.helpers.factories_impl",
    "ServiceTestCaseFactory": "tests.helpers.factories_impl",
    "ServiceTestCases": "tests.helpers.factories_impl",
    "TestDataGenerators": "tests.helpers.factories_impl",
    "TestHelperFactories": "tests.helpers.factories",
    "TestHelperScenarios": "tests.helpers.scenarios",
    "UserFactory": "tests.helpers.factories_impl",
    "ValidatingService": "tests.helpers.factories_impl",
    "ValidatingServiceAuto": "tests.helpers.factories_impl",
    "ValidatingServiceAutoFactory": "tests.helpers.factories_impl",
    "ValidatingServiceFactory": "tests.helpers.factories_impl",
    "ValidationScenario": "tests.helpers._scenarios_impl",
    "ValidationScenarios": "tests.helpers._scenarios_impl",
    "_scenarios_impl": "tests.helpers._scenarios_impl",
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "factories": "tests.helpers.factories",
    "factories_impl": "tests.helpers.factories_impl",
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
    "reset_all_factories": "tests.helpers.factories_impl",
    "s": ("flext_core.service", "FlextService"),
    "scenarios": "tests.helpers.scenarios",
    "t": ("flext_core.typings", "FlextTypes"),
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
