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
        test_architecture,
        test_config_integration,
        test_infra_integration,
        test_integration,
        test_migration_validation,
        test_refactor_nesting_file,
        test_refactor_nesting_idempotency,
        test_refactor_nesting_project,
        test_refactor_nesting_workspace,
        test_refactor_policy_mro,
        test_service,
        test_system,
    )
    from tests.integration.patterns import *
    from tests.integration.test_architecture import *
    from tests.integration.test_config_integration import *
    from tests.integration.test_infra_integration import *
    from tests.integration.test_integration import *
    from tests.integration.test_migration_validation import *
    from tests.integration.test_refactor_nesting_file import *
    from tests.integration.test_refactor_nesting_idempotency import *
    from tests.integration.test_refactor_nesting_project import *
    from tests.integration.test_refactor_nesting_workspace import *
    from tests.integration.test_refactor_policy_mro import *
    from tests.integration.test_service import *
    from tests.integration.test_system import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "EXPECTED_BULK_SIZE": "tests.integration.patterns.test_patterns_logging",
    "TestAdvancedPatterns": "tests.integration.patterns.test_advanced_patterns",
    "TestArchitecturalPatterns": "tests.integration.patterns.test_architectural_patterns",
    "TestAutomatedArchitecture": "tests.integration.test_architecture",
    "TestCompleteFlextSystemIntegration": "tests.integration.test_system",
    "TestFlextSettingsSingletonIntegration": "tests.integration.test_config_integration",
    "TestFunction": "tests.integration.patterns.test_advanced_patterns",
    "TestIdempotency": "tests.integration.test_refactor_nesting_idempotency",
    "TestInfraIntegration": "tests.integration.test_infra_integration",
    "TestLibraryIntegration": "tests.integration.test_integration",
    "TestMigrationValidation": "tests.integration.test_migration_validation",
    "TestPatternsCommands": "tests.integration.patterns.test_patterns_commands",
    "TestPatternsLogging": "tests.integration.patterns.test_patterns_logging",
    "TestPatternsTesting": "tests.integration.patterns.test_patterns_testing",
    "TestProjectLevelRefactor": "tests.integration.test_refactor_nesting_project",
    "TestRefactorPolicyMRO": "tests.integration.test_refactor_policy_mro",
    "TestService": "tests.integration.test_service",
    "TestWorkspaceLevelRefactor": "tests.integration.test_refactor_nesting_workspace",
    "patterns": "tests.integration.patterns",
    "pytestmark": "tests.integration.test_refactor_nesting_file",
    "test_advanced_patterns": "tests.integration.patterns.test_advanced_patterns",
    "test_architectural_patterns": "tests.integration.patterns.test_architectural_patterns",
    "test_architecture": "tests.integration.test_architecture",
    "test_class_nesting_refactor_single_file_end_to_end": "tests.integration.test_refactor_nesting_file",
    "test_config_integration": "tests.integration.test_config_integration",
    "test_infra_integration": "tests.integration.test_infra_integration",
    "test_integration": "tests.integration.test_integration",
    "test_migration_validation": "tests.integration.test_migration_validation",
    "test_patterns_commands": "tests.integration.patterns.test_patterns_commands",
    "test_patterns_logging": "tests.integration.patterns.test_patterns_logging",
    "test_patterns_testing": "tests.integration.patterns.test_patterns_testing",
    "test_refactor_nesting_file": "tests.integration.test_refactor_nesting_file",
    "test_refactor_nesting_idempotency": "tests.integration.test_refactor_nesting_idempotency",
    "test_refactor_nesting_project": "tests.integration.test_refactor_nesting_project",
    "test_refactor_nesting_workspace": "tests.integration.test_refactor_nesting_workspace",
    "test_refactor_policy_mro": "tests.integration.test_refactor_policy_mro",
    "test_service": "tests.integration.test_service",
    "test_system": "tests.integration.test_system",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
