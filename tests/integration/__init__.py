# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Integration package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.integration import (
        patterns as patterns,
        test_architecture as test_architecture,
        test_config_integration as test_config_integration,
        test_infra_integration as test_infra_integration,
        test_integration as test_integration,
        test_migration_validation as test_migration_validation,
        test_refactor_nesting_file as test_refactor_nesting_file,
        test_refactor_nesting_idempotency as test_refactor_nesting_idempotency,
        test_refactor_nesting_project as test_refactor_nesting_project,
        test_refactor_nesting_workspace as test_refactor_nesting_workspace,
        test_refactor_policy_mro as test_refactor_policy_mro,
        test_service as test_service,
        test_system as test_system,
    )
    from tests.integration.patterns import (
        test_advanced_patterns as test_advanced_patterns,
        test_architectural_patterns as test_architectural_patterns,
        test_patterns_commands as test_patterns_commands,
        test_patterns_logging as test_patterns_logging,
        test_patterns_testing as test_patterns_testing,
    )
    from tests.integration.patterns.test_advanced_patterns import (
        TestAdvancedPatterns as TestAdvancedPatterns,
        TestFunction as TestFunction,
    )
    from tests.integration.patterns.test_architectural_patterns import (
        TestArchitecturalPatterns as TestArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        TestPatternsCommands as TestPatternsCommands,
    )
    from tests.integration.patterns.test_patterns_logging import (
        EXPECTED_BULK_SIZE as EXPECTED_BULK_SIZE,
        TestPatternsLogging as TestPatternsLogging,
    )
    from tests.integration.patterns.test_patterns_testing import (
        TestPatternsTesting as TestPatternsTesting,
    )
    from tests.integration.test_architecture import (
        TestAutomatedArchitecture as TestAutomatedArchitecture,
    )
    from tests.integration.test_config_integration import (
        TestFlextSettingsSingletonIntegration as TestFlextSettingsSingletonIntegration,
    )
    from tests.integration.test_infra_integration import (
        TestInfraIntegration as TestInfraIntegration,
    )
    from tests.integration.test_integration import (
        TestLibraryIntegration as TestLibraryIntegration,
    )
    from tests.integration.test_migration_validation import (
        TestMigrationValidation as TestMigrationValidation,
    )
    from tests.integration.test_refactor_nesting_file import (
        pytestmark as pytestmark,
        test_class_nesting_refactor_single_file_end_to_end as test_class_nesting_refactor_single_file_end_to_end,
    )
    from tests.integration.test_refactor_nesting_idempotency import (
        TestIdempotency as TestIdempotency,
    )
    from tests.integration.test_refactor_nesting_project import (
        TestProjectLevelRefactor as TestProjectLevelRefactor,
    )
    from tests.integration.test_refactor_nesting_workspace import (
        TestWorkspaceLevelRefactor as TestWorkspaceLevelRefactor,
    )
    from tests.integration.test_refactor_policy_mro import (
        TestRefactorPolicyMRO as TestRefactorPolicyMRO,
    )
    from tests.integration.test_service import TestService as TestService
    from tests.integration.test_system import (
        TestCompleteFlextSystemIntegration as TestCompleteFlextSystemIntegration,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "EXPECTED_BULK_SIZE": [
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ],
    "TestAdvancedPatterns": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ],
    "TestArchitecturalPatterns": [
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ],
    "TestAutomatedArchitecture": [
        "tests.integration.test_architecture",
        "TestAutomatedArchitecture",
    ],
    "TestCompleteFlextSystemIntegration": [
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ],
    "TestFlextSettingsSingletonIntegration": [
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
    ],
    "TestFunction": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ],
    "TestIdempotency": [
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ],
    "TestInfraIntegration": [
        "tests.integration.test_infra_integration",
        "TestInfraIntegration",
    ],
    "TestLibraryIntegration": [
        "tests.integration.test_integration",
        "TestLibraryIntegration",
    ],
    "TestMigrationValidation": [
        "tests.integration.test_migration_validation",
        "TestMigrationValidation",
    ],
    "TestPatternsCommands": [
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ],
    "TestPatternsLogging": [
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ],
    "TestPatternsTesting": [
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ],
    "TestProjectLevelRefactor": [
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ],
    "TestRefactorPolicyMRO": [
        "tests.integration.test_refactor_policy_mro",
        "TestRefactorPolicyMRO",
    ],
    "TestService": ["tests.integration.test_service", "TestService"],
    "TestWorkspaceLevelRefactor": [
        "tests.integration.test_refactor_nesting_workspace",
        "TestWorkspaceLevelRefactor",
    ],
    "patterns": ["tests.integration.patterns", ""],
    "pytestmark": ["tests.integration.test_refactor_nesting_file", "pytestmark"],
    "test_advanced_patterns": ["tests.integration.patterns.test_advanced_patterns", ""],
    "test_architectural_patterns": [
        "tests.integration.patterns.test_architectural_patterns",
        "",
    ],
    "test_architecture": ["tests.integration.test_architecture", ""],
    "test_class_nesting_refactor_single_file_end_to_end": [
        "tests.integration.test_refactor_nesting_file",
        "test_class_nesting_refactor_single_file_end_to_end",
    ],
    "test_config_integration": ["tests.integration.test_config_integration", ""],
    "test_infra_integration": ["tests.integration.test_infra_integration", ""],
    "test_integration": ["tests.integration.test_integration", ""],
    "test_migration_validation": ["tests.integration.test_migration_validation", ""],
    "test_patterns_commands": ["tests.integration.patterns.test_patterns_commands", ""],
    "test_patterns_logging": ["tests.integration.patterns.test_patterns_logging", ""],
    "test_patterns_testing": ["tests.integration.patterns.test_patterns_testing", ""],
    "test_refactor_nesting_file": ["tests.integration.test_refactor_nesting_file", ""],
    "test_refactor_nesting_idempotency": [
        "tests.integration.test_refactor_nesting_idempotency",
        "",
    ],
    "test_refactor_nesting_project": [
        "tests.integration.test_refactor_nesting_project",
        "",
    ],
    "test_refactor_nesting_workspace": [
        "tests.integration.test_refactor_nesting_workspace",
        "",
    ],
    "test_refactor_policy_mro": ["tests.integration.test_refactor_policy_mro", ""],
    "test_service": ["tests.integration.test_service", ""],
    "test_system": ["tests.integration.test_system", ""],
}

_EXPORTS: Sequence[str] = [
    "EXPECTED_BULK_SIZE",
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
    "patterns",
    "pytestmark",
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
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
