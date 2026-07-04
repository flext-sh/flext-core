"""Integration type-checking exports for tests."""

from __future__ import annotations

from tests.benchmark.test_container_memory import TestsFlextContainerMemory
from tests.benchmark.test_container_performance import TestsFlextContainerPerformance
from tests.benchmark.test_lazy_performance import TestsFlextLazyPerformance
from tests.fixtures.clean_module import (
    TestsFlextCleanConstants,
    TestsFlextCleanModels,
    TestsFlextCleanProtocols,
    TestsFlextCleanServiceBase,
)
from tests.integration.test_architecture import TestsFlextAutomatedArchitecture
from tests.integration.test_documented_patterns import TestsFlextDocumentedPatterns
from tests.integration.test_examples_execution import TestsFlextExamplesExecution
from tests.integration.test_integration import TestsFlextLibraryIntegration
from tests.integration.test_migration_validation import TestsFlextMigrationValidation
from tests.integration.test_service import TestsFlextServiceIntegration
from tests.integration.test_settings_integration import TestsFlextSettingsIntegration
from tests.integration.test_system import TestsFlextSystemIntegration

__all__: list[str] = [
    "TestsFlextAutomatedArchitecture",
    "TestsFlextCleanConstants",
    "TestsFlextCleanModels",
    "TestsFlextCleanProtocols",
    "TestsFlextCleanServiceBase",
    "TestsFlextContainerMemory",
    "TestsFlextContainerPerformance",
    "TestsFlextDocumentedPatterns",
    "TestsFlextExamplesExecution",
    "TestsFlextLazyPerformance",
    "TestsFlextLibraryIntegration",
    "TestsFlextMigrationValidation",
    "TestsFlextServiceIntegration",
    "TestsFlextSettingsIntegration",
    "TestsFlextSystemIntegration",
]
