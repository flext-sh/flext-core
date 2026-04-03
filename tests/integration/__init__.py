# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Integration package."""

from __future__ import annotations

import typing as _t

from flext_core.constants import FlextConstants as c
from flext_core.decorators import FlextDecorators as d
from flext_core.exceptions import FlextExceptions as e
from flext_core.handlers import FlextHandlers as h
from flext_core.lazy import install_lazy_exports, merge_lazy_imports
from flext_core.mixins import FlextMixins as x
from flext_core.models import FlextModels as m
from flext_core.protocols import FlextProtocols as p
from flext_core.result import FlextResult as r
from flext_core.service import FlextService as s
from flext_core.typings import FlextTypes as t
from flext_core.utilities import FlextUtilities as u
from tests.integration.patterns.test_advanced_patterns import (
    TestAdvancedPatterns,
    TestFunction,
)
from tests.integration.patterns.test_architectural_patterns import (
    TestArchitecturalPatterns,
)
from tests.integration.patterns.test_patterns_commands import TestPatternsCommands
from tests.integration.patterns.test_patterns_logging import (
    EXPECTED_BULK_SIZE,
    TestPatternsLogging,
)
from tests.integration.patterns.test_patterns_testing import (
    P,
    R,
    TestPatternsTesting,
)
from tests.integration.test_architecture import TestAutomatedArchitecture
from tests.integration.test_config_integration import (
    TestFlextSettingsSingletonIntegration,
)
from tests.integration.test_infra_integration import TestInfraIntegration
from tests.integration.test_integration import TestLibraryIntegration
from tests.integration.test_migration_validation import TestMigrationValidation
from tests.integration.test_refactor_nesting_file import (
    pytestmark,
    test_class_nesting_refactor_single_file_end_to_end,
)
from tests.integration.test_refactor_nesting_idempotency import TestIdempotency
from tests.integration.test_refactor_nesting_project import TestProjectLevelRefactor
from tests.integration.test_refactor_nesting_workspace import (
    TestWorkspaceLevelRefactor,
)
from tests.integration.test_refactor_policy_mro import TestRefactorPolicyMRO
from tests.integration.test_service import TestService
from tests.integration.test_system import TestCompleteFlextSystemIntegration

if _t.TYPE_CHECKING:
    import tests.integration.patterns as _tests_integration_patterns

    patterns = _tests_integration_patterns
    import tests.integration.patterns.test_advanced_patterns as _tests_integration_patterns_test_advanced_patterns

    test_advanced_patterns = _tests_integration_patterns_test_advanced_patterns
    import tests.integration.patterns.test_architectural_patterns as _tests_integration_patterns_test_architectural_patterns

    test_architectural_patterns = (
        _tests_integration_patterns_test_architectural_patterns
    )
    import tests.integration.patterns.test_patterns_commands as _tests_integration_patterns_test_patterns_commands

    test_patterns_commands = _tests_integration_patterns_test_patterns_commands
    import tests.integration.patterns.test_patterns_logging as _tests_integration_patterns_test_patterns_logging

    test_patterns_logging = _tests_integration_patterns_test_patterns_logging
    import tests.integration.patterns.test_patterns_testing as _tests_integration_patterns_test_patterns_testing

    test_patterns_testing = _tests_integration_patterns_test_patterns_testing
    import tests.integration.test_architecture as _tests_integration_test_architecture

    test_architecture = _tests_integration_test_architecture
    import tests.integration.test_config_integration as _tests_integration_test_config_integration

    test_config_integration = _tests_integration_test_config_integration
    import tests.integration.test_infra_integration as _tests_integration_test_infra_integration

    test_infra_integration = _tests_integration_test_infra_integration
    import tests.integration.test_integration as _tests_integration_test_integration

    test_integration = _tests_integration_test_integration
    import tests.integration.test_migration_validation as _tests_integration_test_migration_validation

    test_migration_validation = _tests_integration_test_migration_validation
    import tests.integration.test_refactor_nesting_file as _tests_integration_test_refactor_nesting_file

    test_refactor_nesting_file = _tests_integration_test_refactor_nesting_file
    import tests.integration.test_refactor_nesting_idempotency as _tests_integration_test_refactor_nesting_idempotency

    test_refactor_nesting_idempotency = (
        _tests_integration_test_refactor_nesting_idempotency
    )
    import tests.integration.test_refactor_nesting_project as _tests_integration_test_refactor_nesting_project

    test_refactor_nesting_project = _tests_integration_test_refactor_nesting_project
    import tests.integration.test_refactor_nesting_workspace as _tests_integration_test_refactor_nesting_workspace

    test_refactor_nesting_workspace = _tests_integration_test_refactor_nesting_workspace
    import tests.integration.test_refactor_policy_mro as _tests_integration_test_refactor_policy_mro

    test_refactor_policy_mro = _tests_integration_test_refactor_policy_mro
    import tests.integration.test_service as _tests_integration_test_service

    test_service = _tests_integration_test_service
    import tests.integration.test_system as _tests_integration_test_system

    test_system = _tests_integration_test_system

    _ = (
        EXPECTED_BULK_SIZE,
        P,
        R,
        TestAdvancedPatterns,
        TestArchitecturalPatterns,
        TestAutomatedArchitecture,
        TestCompleteFlextSystemIntegration,
        TestFlextSettingsSingletonIntegration,
        TestFunction,
        TestIdempotency,
        TestInfraIntegration,
        TestLibraryIntegration,
        TestMigrationValidation,
        TestPatternsCommands,
        TestPatternsLogging,
        TestPatternsTesting,
        TestProjectLevelRefactor,
        TestRefactorPolicyMRO,
        TestService,
        TestWorkspaceLevelRefactor,
        c,
        d,
        e,
        h,
        m,
        p,
        patterns,
        pytestmark,
        r,
        s,
        t,
        test_advanced_patterns,
        test_architectural_patterns,
        test_architecture,
        test_class_nesting_refactor_single_file_end_to_end,
        test_config_integration,
        test_infra_integration,
        test_integration,
        test_migration_validation,
        test_patterns_commands,
        test_patterns_logging,
        test_patterns_testing,
        test_refactor_nesting_file,
        test_refactor_nesting_idempotency,
        test_refactor_nesting_project,
        test_refactor_nesting_workspace,
        test_refactor_policy_mro,
        test_service,
        test_system,
        u,
        x,
    )
_LAZY_IMPORTS = merge_lazy_imports(
    ("tests.integration.patterns",),
    {
        "TestAutomatedArchitecture": "tests.integration.test_architecture",
        "TestCompleteFlextSystemIntegration": "tests.integration.test_system",
        "TestFlextSettingsSingletonIntegration": "tests.integration.test_config_integration",
        "TestIdempotency": "tests.integration.test_refactor_nesting_idempotency",
        "TestInfraIntegration": "tests.integration.test_infra_integration",
        "TestLibraryIntegration": "tests.integration.test_integration",
        "TestMigrationValidation": "tests.integration.test_migration_validation",
        "TestProjectLevelRefactor": "tests.integration.test_refactor_nesting_project",
        "TestRefactorPolicyMRO": "tests.integration.test_refactor_policy_mro",
        "TestService": "tests.integration.test_service",
        "TestWorkspaceLevelRefactor": "tests.integration.test_refactor_nesting_workspace",
        "c": ("flext_core.constants", "FlextConstants"),
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "h": ("flext_core.handlers", "FlextHandlers"),
        "m": ("flext_core.models", "FlextModels"),
        "p": ("flext_core.protocols", "FlextProtocols"),
        "patterns": "tests.integration.patterns",
        "pytestmark": "tests.integration.test_refactor_nesting_file",
        "r": ("flext_core.result", "FlextResult"),
        "s": ("flext_core.service", "FlextService"),
        "t": ("flext_core.typings", "FlextTypes"),
        "test_architecture": "tests.integration.test_architecture",
        "test_class_nesting_refactor_single_file_end_to_end": "tests.integration.test_refactor_nesting_file",
        "test_config_integration": "tests.integration.test_config_integration",
        "test_infra_integration": "tests.integration.test_infra_integration",
        "test_integration": "tests.integration.test_integration",
        "test_migration_validation": "tests.integration.test_migration_validation",
        "test_refactor_nesting_file": "tests.integration.test_refactor_nesting_file",
        "test_refactor_nesting_idempotency": "tests.integration.test_refactor_nesting_idempotency",
        "test_refactor_nesting_project": "tests.integration.test_refactor_nesting_project",
        "test_refactor_nesting_workspace": "tests.integration.test_refactor_nesting_workspace",
        "test_refactor_policy_mro": "tests.integration.test_refactor_policy_mro",
        "test_service": "tests.integration.test_service",
        "test_system": "tests.integration.test_system",
        "u": ("flext_core.utilities", "FlextUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)

__all__ = [
    "EXPECTED_BULK_SIZE",
    "P",
    "R",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestAutomatedArchitecture",
    "TestCompleteFlextSystemIntegration",
    "TestFlextSettingsSingletonIntegration",
    "TestFunction",
    "TestIdempotency",
    "TestInfraIntegration",
    "TestLibraryIntegration",
    "TestMigrationValidation",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestProjectLevelRefactor",
    "TestRefactorPolicyMRO",
    "TestService",
    "TestWorkspaceLevelRefactor",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "patterns",
    "pytestmark",
    "r",
    "s",
    "t",
    "test_advanced_patterns",
    "test_architectural_patterns",
    "test_architecture",
    "test_class_nesting_refactor_single_file_end_to_end",
    "test_config_integration",
    "test_infra_integration",
    "test_integration",
    "test_migration_validation",
    "test_patterns_commands",
    "test_patterns_logging",
    "test_patterns_testing",
    "test_refactor_nesting_file",
    "test_refactor_nesting_idempotency",
    "test_refactor_nesting_project",
    "test_refactor_nesting_workspace",
    "test_refactor_policy_mro",
    "test_service",
    "test_system",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
