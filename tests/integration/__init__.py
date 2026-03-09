# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Integration tests for FLEXT Core components.

Integration tests focus on testing component interactions and
how different modules work together in realistic scenarios using
advanced Python 3.13 patterns and comprehensive edge case testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.integration.patterns.test_advanced_patterns import TestFunction
    from tests.integration.patterns.test_architectural_patterns import (
        TestEnterprisePatterns,
        TestEventDrivenPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        CreateUserCommand,
        CreateUserCommandHandler,
        FailingCommand,
        FailingCommandHandler,
        FlextCommandId,
        FlextCommandType,
        TestFlextCommand,
        TestFlextCommandHandler,
        TestFlextCommandResults,
        UpdateUserCommand,
        UpdateUserCommandHandler,
    )
    from tests.integration.patterns.test_patterns_logging import (
        TestFlextContext,
        TestFlextLogger,
        TestFlextLoggerIntegration,
        TestFlextLoggerUsage,
        TestFlextLogLevel,
        make_result_logger,
    )
    from tests.integration.patterns.test_patterns_testing import (
        AssertionBuilder,
        FixtureBuilder,
        FlextTestBuilder,
        GivenWhenThenBuilder,
        MockScenario,
        ParameterizedTestBuilder,
        SuiteBuilder,
        TestAdvancedPatterns,
        TestComprehensiveIntegration,
        TestPerformanceAnalysis,
        TestPropertyBasedPatterns,
        TestRealWorldScenarios,
        arrange_act_assert,
        mark_test_pattern,
    )
    from tests.integration.test_config_integration import (
        ConfigTestCase,
        ConfigTestFactories,
        TestFlextSettingsSingletonIntegration,
        ThreadSafetyTest,
    )
    from tests.integration.test_infra_integration import (
        TestBaseMkGenerationFlow,
        TestContainerIntegration,
        TestCrossModuleIntegration,
        TestIntegrationWithRealCommandServices,
        TestOutputSingletonConsistency,
        TestPathResolverDiscoveryFlow,
        TestServiceFlextResultChaining,
        TestWorkspaceDetectionOrchestrationFlow,
    )
    from tests.integration.test_integration import TestLibraryIntegration
    from tests.integration.test_migration_validation import (
        TestBackwardCompatibility,
        TestMigrationComplexity,
        TestMigrationScenario1,
        TestMigrationScenario2,
        TestMigrationScenario4,
        TestMigrationScenario5,
    )
    from tests.integration.test_refactor_nesting_file import (
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from tests.integration.test_refactor_nesting_idempotency import TestIdempotency
    from tests.integration.test_refactor_nesting_project import TestProjectLevelRefactor
    from tests.integration.test_refactor_nesting_workspace import (
        TestWorkspaceLevelRefactor,
    )
    from tests.integration.test_refactor_policy_mro import (
        AlgarOudMigConstants,
        AlgarOudMigModels,
        AlgarOudMigProtocols,
        AlgarOudMigTypes,
        AlgarOudMigUtilities,
        FlextCliConstants,
        FlextCliModels,
        FlextCliProtocols,
        FlextCliTypes,
        FlextCliUtilities,
        FlextLdapConstants,
        FlextLdapConstants as c,
        FlextLdapModels,
        FlextLdapModels as m,
        FlextLdapProtocols,
        FlextLdapProtocols as p,
        FlextLdapTypes,
        FlextLdapTypes as t,
        FlextLdapUtilities,
        FlextLdapUtilities as u,
        test_mro_resolver_accepts_expected_order,
        test_mro_resolver_rejects_wrong_order,
    )
    from tests.integration.test_service import (
        LifecycleService,
        NotificationService,
        ServiceConfig,
        TestFlextServiceIntegration,
        UserQueryService,
        UserQueryService as s,
        UserServiceEntity,
        pytestmark,
    )
    from tests.integration.test_system import TestCompleteFlextSystemIntegration

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AlgarOudMigConstants": ("tests.integration.test_refactor_policy_mro", "AlgarOudMigConstants"),
    "AlgarOudMigModels": ("tests.integration.test_refactor_policy_mro", "AlgarOudMigModels"),
    "AlgarOudMigProtocols": ("tests.integration.test_refactor_policy_mro", "AlgarOudMigProtocols"),
    "AlgarOudMigTypes": ("tests.integration.test_refactor_policy_mro", "AlgarOudMigTypes"),
    "AlgarOudMigUtilities": ("tests.integration.test_refactor_policy_mro", "AlgarOudMigUtilities"),
    "AssertionBuilder": ("tests.integration.patterns.test_patterns_testing", "AssertionBuilder"),
    "ConfigTestCase": ("tests.integration.test_config_integration", "ConfigTestCase"),
    "ConfigTestFactories": ("tests.integration.test_config_integration", "ConfigTestFactories"),
    "CreateUserCommand": ("tests.integration.patterns.test_patterns_commands", "CreateUserCommand"),
    "CreateUserCommandHandler": ("tests.integration.patterns.test_patterns_commands", "CreateUserCommandHandler"),
    "FailingCommand": ("tests.integration.patterns.test_patterns_commands", "FailingCommand"),
    "FailingCommandHandler": ("tests.integration.patterns.test_patterns_commands", "FailingCommandHandler"),
    "FixtureBuilder": ("tests.integration.patterns.test_patterns_testing", "FixtureBuilder"),
    "FlextCliConstants": ("tests.integration.test_refactor_policy_mro", "FlextCliConstants"),
    "FlextCliModels": ("tests.integration.test_refactor_policy_mro", "FlextCliModels"),
    "FlextCliProtocols": ("tests.integration.test_refactor_policy_mro", "FlextCliProtocols"),
    "FlextCliTypes": ("tests.integration.test_refactor_policy_mro", "FlextCliTypes"),
    "FlextCliUtilities": ("tests.integration.test_refactor_policy_mro", "FlextCliUtilities"),
    "FlextCommandId": ("tests.integration.patterns.test_patterns_commands", "FlextCommandId"),
    "FlextCommandType": ("tests.integration.patterns.test_patterns_commands", "FlextCommandType"),
    "FlextLdapConstants": ("tests.integration.test_refactor_policy_mro", "FlextLdapConstants"),
    "FlextLdapModels": ("tests.integration.test_refactor_policy_mro", "FlextLdapModels"),
    "FlextLdapProtocols": ("tests.integration.test_refactor_policy_mro", "FlextLdapProtocols"),
    "FlextLdapTypes": ("tests.integration.test_refactor_policy_mro", "FlextLdapTypes"),
    "FlextLdapUtilities": ("tests.integration.test_refactor_policy_mro", "FlextLdapUtilities"),
    "FlextTestBuilder": ("tests.integration.patterns.test_patterns_testing", "FlextTestBuilder"),
    "GivenWhenThenBuilder": ("tests.integration.patterns.test_patterns_testing", "GivenWhenThenBuilder"),
    "LifecycleService": ("tests.integration.test_service", "LifecycleService"),
    "MockScenario": ("tests.integration.patterns.test_patterns_testing", "MockScenario"),
    "NotificationService": ("tests.integration.test_service", "NotificationService"),
    "ParameterizedTestBuilder": ("tests.integration.patterns.test_patterns_testing", "ParameterizedTestBuilder"),
    "ServiceConfig": ("tests.integration.test_service", "ServiceConfig"),
    "SuiteBuilder": ("tests.integration.patterns.test_patterns_testing", "SuiteBuilder"),
    "TestAdvancedPatterns": ("tests.integration.patterns.test_patterns_testing", "TestAdvancedPatterns"),
    "TestBackwardCompatibility": ("tests.integration.test_migration_validation", "TestBackwardCompatibility"),
    "TestBaseMkGenerationFlow": ("tests.integration.test_infra_integration", "TestBaseMkGenerationFlow"),
    "TestCompleteFlextSystemIntegration": ("tests.integration.test_system", "TestCompleteFlextSystemIntegration"),
    "TestComprehensiveIntegration": ("tests.integration.patterns.test_patterns_testing", "TestComprehensiveIntegration"),
    "TestContainerIntegration": ("tests.integration.test_infra_integration", "TestContainerIntegration"),
    "TestCrossModuleIntegration": ("tests.integration.test_infra_integration", "TestCrossModuleIntegration"),
    "TestEnterprisePatterns": ("tests.integration.patterns.test_architectural_patterns", "TestEnterprisePatterns"),
    "TestEventDrivenPatterns": ("tests.integration.patterns.test_architectural_patterns", "TestEventDrivenPatterns"),
    "TestFlextCommand": ("tests.integration.patterns.test_patterns_commands", "TestFlextCommand"),
    "TestFlextCommandHandler": ("tests.integration.patterns.test_patterns_commands", "TestFlextCommandHandler"),
    "TestFlextCommandResults": ("tests.integration.patterns.test_patterns_commands", "TestFlextCommandResults"),
    "TestFlextContext": ("tests.integration.patterns.test_patterns_logging", "TestFlextContext"),
    "TestFlextLogLevel": ("tests.integration.patterns.test_patterns_logging", "TestFlextLogLevel"),
    "TestFlextLogger": ("tests.integration.patterns.test_patterns_logging", "TestFlextLogger"),
    "TestFlextLoggerIntegration": ("tests.integration.patterns.test_patterns_logging", "TestFlextLoggerIntegration"),
    "TestFlextLoggerUsage": ("tests.integration.patterns.test_patterns_logging", "TestFlextLoggerUsage"),
    "TestFlextServiceIntegration": ("tests.integration.test_service", "TestFlextServiceIntegration"),
    "TestFlextSettingsSingletonIntegration": ("tests.integration.test_config_integration", "TestFlextSettingsSingletonIntegration"),
    "TestFunction": ("tests.integration.patterns.test_advanced_patterns", "TestFunction"),
    "TestIdempotency": ("tests.integration.test_refactor_nesting_idempotency", "TestIdempotency"),
    "TestIntegrationWithRealCommandServices": ("tests.integration.test_infra_integration", "TestIntegrationWithRealCommandServices"),
    "TestLibraryIntegration": ("tests.integration.test_integration", "TestLibraryIntegration"),
    "TestMigrationComplexity": ("tests.integration.test_migration_validation", "TestMigrationComplexity"),
    "TestMigrationScenario1": ("tests.integration.test_migration_validation", "TestMigrationScenario1"),
    "TestMigrationScenario2": ("tests.integration.test_migration_validation", "TestMigrationScenario2"),
    "TestMigrationScenario4": ("tests.integration.test_migration_validation", "TestMigrationScenario4"),
    "TestMigrationScenario5": ("tests.integration.test_migration_validation", "TestMigrationScenario5"),
    "TestOutputSingletonConsistency": ("tests.integration.test_infra_integration", "TestOutputSingletonConsistency"),
    "TestPathResolverDiscoveryFlow": ("tests.integration.test_infra_integration", "TestPathResolverDiscoveryFlow"),
    "TestPerformanceAnalysis": ("tests.integration.patterns.test_patterns_testing", "TestPerformanceAnalysis"),
    "TestProjectLevelRefactor": ("tests.integration.test_refactor_nesting_project", "TestProjectLevelRefactor"),
    "TestPropertyBasedPatterns": ("tests.integration.patterns.test_patterns_testing", "TestPropertyBasedPatterns"),
    "TestRealWorldScenarios": ("tests.integration.patterns.test_patterns_testing", "TestRealWorldScenarios"),
    "TestServiceFlextResultChaining": ("tests.integration.test_infra_integration", "TestServiceFlextResultChaining"),
    "TestWorkspaceDetectionOrchestrationFlow": ("tests.integration.test_infra_integration", "TestWorkspaceDetectionOrchestrationFlow"),
    "TestWorkspaceLevelRefactor": ("tests.integration.test_refactor_nesting_workspace", "TestWorkspaceLevelRefactor"),
    "ThreadSafetyTest": ("tests.integration.test_config_integration", "ThreadSafetyTest"),
    "UpdateUserCommand": ("tests.integration.patterns.test_patterns_commands", "UpdateUserCommand"),
    "UpdateUserCommandHandler": ("tests.integration.patterns.test_patterns_commands", "UpdateUserCommandHandler"),
    "UserQueryService": ("tests.integration.test_service", "UserQueryService"),
    "UserServiceEntity": ("tests.integration.test_service", "UserServiceEntity"),
    "arrange_act_assert": ("tests.integration.patterns.test_patterns_testing", "arrange_act_assert"),
    "c": ("tests.integration.test_refactor_policy_mro", "FlextLdapConstants"),
    "m": ("tests.integration.test_refactor_policy_mro", "FlextLdapModels"),
    "make_result_logger": ("tests.integration.patterns.test_patterns_logging", "make_result_logger"),
    "mark_test_pattern": ("tests.integration.patterns.test_patterns_testing", "mark_test_pattern"),
    "p": ("tests.integration.test_refactor_policy_mro", "FlextLdapProtocols"),
    "pytestmark": ("tests.integration.test_service", "pytestmark"),
    "s": ("tests.integration.test_service", "UserQueryService"),
    "t": ("tests.integration.test_refactor_policy_mro", "FlextLdapTypes"),
    "test_class_nesting_refactor_single_file_end_to_end": ("tests.integration.test_refactor_nesting_file", "test_class_nesting_refactor_single_file_end_to_end"),
    "test_mro_resolver_accepts_expected_order": ("tests.integration.test_refactor_policy_mro", "test_mro_resolver_accepts_expected_order"),
    "test_mro_resolver_rejects_wrong_order": ("tests.integration.test_refactor_policy_mro", "test_mro_resolver_rejects_wrong_order"),
    "u": ("tests.integration.test_refactor_policy_mro", "FlextLdapUtilities"),
}

__all__ = [
    "AlgarOudMigConstants",
    "AlgarOudMigModels",
    "AlgarOudMigProtocols",
    "AlgarOudMigTypes",
    "AlgarOudMigUtilities",
    "AssertionBuilder",
    "ConfigTestCase",
    "ConfigTestFactories",
    "CreateUserCommand",
    "CreateUserCommandHandler",
    "FailingCommand",
    "FailingCommandHandler",
    "FixtureBuilder",
    "FlextCliConstants",
    "FlextCliModels",
    "FlextCliProtocols",
    "FlextCliTypes",
    "FlextCliUtilities",
    "FlextCommandId",
    "FlextCommandType",
    "FlextLdapConstants",
    "FlextLdapModels",
    "FlextLdapProtocols",
    "FlextLdapTypes",
    "FlextLdapUtilities",
    "FlextTestBuilder",
    "GivenWhenThenBuilder",
    "LifecycleService",
    "MockScenario",
    "NotificationService",
    "ParameterizedTestBuilder",
    "ServiceConfig",
    "SuiteBuilder",
    "TestAdvancedPatterns",
    "TestBackwardCompatibility",
    "TestBaseMkGenerationFlow",
    "TestCompleteFlextSystemIntegration",
    "TestComprehensiveIntegration",
    "TestContainerIntegration",
    "TestCrossModuleIntegration",
    "TestEnterprisePatterns",
    "TestEventDrivenPatterns",
    "TestFlextCommand",
    "TestFlextCommandHandler",
    "TestFlextCommandResults",
    "TestFlextContext",
    "TestFlextLogLevel",
    "TestFlextLogger",
    "TestFlextLoggerIntegration",
    "TestFlextLoggerUsage",
    "TestFlextServiceIntegration",
    "TestFlextSettingsSingletonIntegration",
    "TestFunction",
    "TestIdempotency",
    "TestIntegrationWithRealCommandServices",
    "TestLibraryIntegration",
    "TestMigrationComplexity",
    "TestMigrationScenario1",
    "TestMigrationScenario2",
    "TestMigrationScenario4",
    "TestMigrationScenario5",
    "TestOutputSingletonConsistency",
    "TestPathResolverDiscoveryFlow",
    "TestPerformanceAnalysis",
    "TestProjectLevelRefactor",
    "TestPropertyBasedPatterns",
    "TestRealWorldScenarios",
    "TestServiceFlextResultChaining",
    "TestWorkspaceDetectionOrchestrationFlow",
    "TestWorkspaceLevelRefactor",
    "ThreadSafetyTest",
    "UpdateUserCommand",
    "UpdateUserCommandHandler",
    "UserQueryService",
    "UserServiceEntity",
    "arrange_act_assert",
    "c",
    "m",
    "make_result_logger",
    "mark_test_pattern",
    "p",
    "pytestmark",
    "s",
    "t",
    "test_class_nesting_refactor_single_file_end_to_end",
    "test_mro_resolver_accepts_expected_order",
    "test_mro_resolver_rejects_wrong_order",
    "u",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
