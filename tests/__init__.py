# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import tests.base as _tests_base

    base = _tests_base
    import tests.benchmark as _tests_benchmark
    from tests.base import TestsFlextCoreServiceBase, TestsFlextCoreServiceBase as s

    benchmark = _tests_benchmark
    import tests.conftest as _tests_conftest

    conftest = _tests_conftest
    import tests.constants as _tests_constants

    constants = _tests_constants
    import tests.helpers as _tests_helpers
    from tests.constants import TestsFlextCoreConstants, TestsFlextCoreConstants as c

    helpers = _tests_helpers
    import tests.integration as _tests_integration

    integration = _tests_integration
    import tests.models as _tests_models

    models = _tests_models
    import tests.protocols as _tests_protocols
    from tests.models import TestsFlextCoreModels, TestsFlextCoreModels as m

    protocols = _tests_protocols
    import tests.test_documented_patterns as _tests_test_documented_patterns
    from tests.protocols import TestsFlextCoreProtocols, TestsFlextCoreProtocols as p

    test_documented_patterns = _tests_test_documented_patterns
    import tests.test_examples_execution as _tests_test_examples_execution

    test_examples_execution = _tests_test_examples_execution
    import tests.test_service_result_property as _tests_test_service_result_property

    test_service_result_property = _tests_test_service_result_property
    import tests.test_utils as _tests_test_utils

    test_utils = _tests_test_utils
    import tests.typings as _tests_typings

    typings = _tests_typings
    import tests.unit as _tests_unit
    from tests.typings import (
        T,
        T_co,
        T_contra,
        TestsFlextCoreTypes,
        TestsFlextCoreTypes as t,
    )

    unit = _tests_unit
    import tests.utilities as _tests_utilities
    from tests.unit import TestsFlextUnitProtocols
    from tests.unit._utilities import TestFlextUtilitiesGuards

    utilities = _tests_utilities
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.result import FlextResult as r
    from tests.utilities import TestsFlextCoreUtilities, TestsFlextCoreUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "tests.benchmark",
        "tests.helpers",
        "tests.integration",
        "tests.unit",
    ),
    {
        "T": ("tests.typings", "T"),
        "T_co": ("tests.typings", "T_co"),
        "T_contra": ("tests.typings", "T_contra"),
        "TestsFlextCoreConstants": ("tests.constants", "TestsFlextCoreConstants"),
        "TestsFlextCoreModels": ("tests.models", "TestsFlextCoreModels"),
        "TestsFlextCoreProtocols": ("tests.protocols", "TestsFlextCoreProtocols"),
        "TestsFlextCoreServiceBase": ("tests.base", "TestsFlextCoreServiceBase"),
        "TestsFlextCoreTypes": ("tests.typings", "TestsFlextCoreTypes"),
        "TestsFlextCoreUtilities": ("tests.utilities", "TestsFlextCoreUtilities"),
        "base": "tests.base",
        "benchmark": "tests.benchmark",
        "c": ("tests.constants", "TestsFlextCoreConstants"),
        "conftest": "tests.conftest",
        "constants": "tests.constants",
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "h": ("flext_core.handlers", "FlextHandlers"),
        "helpers": "tests.helpers",
        "integration": "tests.integration",
        "m": ("tests.models", "TestsFlextCoreModels"),
        "models": "tests.models",
        "p": ("tests.protocols", "TestsFlextCoreProtocols"),
        "protocols": "tests.protocols",
        "r": ("flext_core.result", "FlextResult"),
        "s": ("tests.base", "TestsFlextCoreServiceBase"),
        "t": ("tests.typings", "TestsFlextCoreTypes"),
        "test_documented_patterns": "tests.test_documented_patterns",
        "test_examples_execution": "tests.test_examples_execution",
        "test_service_result_property": "tests.test_service_result_property",
        "test_utils": "tests.test_utils",
        "typings": "tests.typings",
        "u": ("tests.utilities", "TestsFlextCoreUtilities"),
        "unit": "tests.unit",
        "utilities": "tests.utilities",
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("logger", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "T",
    "T_co",
    "T_contra",
    "TestFlextUtilitiesGuards",
    "TestsFlextCoreConstants",
    "TestsFlextCoreModels",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreServiceBase",
    "TestsFlextCoreTypes",
    "TestsFlextCoreUtilities",
    "TestsFlextUnitProtocols",
    "base",
    "benchmark",
    "c",
    "conftest",
    "constants",
    "d",
    "e",
    "h",
    "helpers",
    "integration",
    "m",
    "models",
    "p",
    "protocols",
    "r",
    "s",
    "t",
    "test_documented_patterns",
    "test_examples_execution",
    "test_service_result_property",
    "test_utils",
    "typings",
    "u",
    "unit",
    "utilities",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
