# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Tests package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports
from tests.benchmark import _LAZY_IMPORTS as _CHILD_LAZY_0
from tests.helpers import _LAZY_IMPORTS as _CHILD_LAZY_1
from tests.integration import _LAZY_IMPORTS as _CHILD_LAZY_2
from tests.unit import _LAZY_IMPORTS as _CHILD_LAZY_3

if TYPE_CHECKING:
    from tests.base import *
    from tests.benchmark import *
    from tests.conftest import *
    from tests.constants import *
    from tests.fixtures.namespace_validator import *
    from tests.helpers import *
    from tests.integration import *
    from tests.integration.patterns import *
    from tests.models import *
    from tests.protocols import *
    from tests.test_documented_patterns import *
    from tests.test_service_result_property import *
    from tests.test_utils import *
    from tests.typings import *
    from tests.unit import *
    from tests.unit.contracts import *
    from tests.unit.flext_tests import *
    from tests.utilities import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    **_CHILD_LAZY_0,
    **_CHILD_LAZY_1,
    **_CHILD_LAZY_2,
    **_CHILD_LAZY_3,
    "DEFAULT_TIMEOUT": "tests.fixtures.namespace_validator.rule1_loose_constant",
    "FlextCoreTestConstants": "tests.constants",
    "FlextCoreTestModels": "tests.models",
    "FlextCoreTestProtocols": "tests.protocols",
    "FlextCoreTestTypes": "tests.typings",
    "FlextCoreTestUtilities": "tests.utilities",
    "FlextTestConstants": "tests.fixtures.namespace_validator.rule0_multiple_classes",
    "FlextTestModels": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "FlextTestResult": "tests.test_utils",
    "FlextTestResultCo": "tests.test_utils",
    "FlextTestTypes": "tests.fixtures.namespace_validator.rule2_protocol_in_types",
    "FlextTestUtilities": "tests.fixtures.namespace_validator.rule1_magic_number",
    "FunctionalExternalService": "tests.conftest",
    "LooseTypeAlias": "tests.fixtures.namespace_validator.typings",
    "MAX_RETRIES": "tests.fixtures.namespace_validator.rule1_loose_constant",
    "MAX_VALUE": "tests.fixtures.namespace_validator.rule0_no_class",
    "RandomConstants": "tests.fixtures.namespace_validator.rule0_wrong_prefix",
    "Rule0LooseItemsFixture": "tests.fixtures.namespace_validator.rule0_loose_items",
    "Rule0MultipleClassesFixture": "tests.fixtures.namespace_validator.rule0_multiple_classes",
    "Rule1LooseEnumFixture": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "Status": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "T_co": "tests.typings",
    "T_contra": "tests.typings",
    "TestDocumentedPatterns": "tests.test_documented_patterns",
    "TestServiceResultProperty": "tests.test_service_result_property",
    "TestUtils": "tests.test_utils",
    "TestsFlextServiceBase": "tests.base",
    "assert_rejects": "tests.conftest",
    "assert_validates": "tests.conftest",
    "assertion_helpers": "tests.test_utils",
    "base": "tests.base",
    "benchmark": "tests.benchmark",
    "c": ["tests.constants", "FlextCoreTestConstants"],
    "clean_container": "tests.conftest",
    "conftest": "tests.conftest",
    "constants": "tests.constants",
    "d": "flext_tests",
    "e": "flext_tests",
    "empty_strings": "tests.conftest",
    "fixture_factory": "tests.test_utils",
    "flext_result_failure": "tests.conftest",
    "flext_result_success": "tests.conftest",
    "h": "flext_tests",
    "helper": "tests.fixtures.namespace_validator.rule0_no_class",
    "helpers": "tests.helpers",
    "integration": "tests.integration",
    "invalid_hostnames": "tests.conftest",
    "invalid_port_numbers": "tests.conftest",
    "invalid_uris": "tests.conftest",
    "m": ["tests.models", "FlextCoreTestModels"],
    "mock_external_service": "tests.conftest",
    "models": "tests.models",
    "out_of_range": "tests.conftest",
    "p": ["tests.protocols", "FlextCoreTestProtocols"],
    "parser_scenarios": "tests.conftest",
    "protocols": "tests.protocols",
    "r": "flext_tests",
    "reliability_scenarios": "tests.conftest",
    "reset_global_container": "tests.conftest",
    "s": "flext_tests",
    "sample_data": "tests.conftest",
    "t": ["tests.typings", "FlextCoreTestTypes"],
    "temp_dir": "tests.conftest",
    "temp_directory": "tests.conftest",
    "temp_file": "tests.conftest",
    "test_context": "tests.conftest",
    "test_data_factory": "tests.test_utils",
    "test_documented_patterns": "tests.test_documented_patterns",
    "test_service_result_property": "tests.test_service_result_property",
    "test_utils": "tests.test_utils",
    "typings": "tests.typings",
    "u": ["tests.utilities", "FlextCoreTestUtilities"],
    "unit": "tests.unit",
    "utilities": "tests.utilities",
    "valid_hostnames": "tests.conftest",
    "valid_port_numbers": "tests.conftest",
    "valid_ranges": "tests.conftest",
    "valid_strings": "tests.conftest",
    "valid_uris": "tests.conftest",
    "validation_scenarios": "tests.conftest",
    "whitespace_strings": "tests.conftest",
    "x": "flext_tests",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
