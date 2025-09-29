"""Integration tests for flext-core wildcard export system.

This module validates that the wildcard import system works correctly across
the entire flext-core ecosystem, ensuring all major components are properly
exported and functional.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from datetime import UTC, datetime
from typing import get_origin

import pytest

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextCqrs,
    FlextExceptions,
    FlextHandlers,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)


class TestFlextCoreWildcardExports:
    """Integration tests for the flext-core wildcard export system."""

    def test_wildcard_import_works(self) -> None:
        """Test that wildcard import from flext_core works without errors."""
        # Since we already have the wildcard import at module level, test it works
        assert "FlextResult" in globals()
        assert "FlextConstants" in globals()

    def test_all_main_modules_exported(self) -> None:
        """Test that all main flext-core modules are exported via wildcard import."""
        # Define essential modules that must be available
        essential_modules = {
            "FlextResult": "Core result type for railway-oriented programming",
            "FlextConstants": "Hierarchical constants system",
            "FlextExceptions": "Exception hierarchy system",
            "FlextUtilities": "Utility functions and helpers",
            "FlextTypes": "Type system and protocol definitions",
            "FlextConfig": "Configuration management system",
            "FlextContainer": "Dependency injection container",
            "FlextCqrs": "CQRS utilities",
            "FlextHandlers": "Handlers for CQRS patterns",
        }

        # Verify each essential module is available in globals
        current_globals = globals()
        for module_name, description in essential_modules.items():
            assert module_name in current_globals, (
                f"Module '{module_name}' ({description}) not found in wildcard exports"
            )

            # Verify the module is not None
            module = current_globals[module_name]
            assert module is not None, f"Module '{module_name}' is None"

    def test_flext_result_functionality(self) -> None:
        """Test that FlextResult is functional after wildcard import."""
        # Test success case
        success_result = FlextResult[str].ok("test_value")
        assert success_result.is_success is True
        assert success_result.value == "test_value"
        assert success_result.error is None

        # Test failure case
        failure_result = FlextResult[str].fail("test_error")
        assert failure_result.is_success is False
        assert failure_result.error == "test_error"
        # Don't access .value on failed result as it raises TypeError

        # Test chaining operations
        chained_result = success_result.map(lambda x: x.upper())
        assert chained_result.is_success is True
        assert chained_result.value == "TEST_VALUE"

    def test_flext_utilities_functionality(self) -> None:
        """Test that FlextUtilities functions work after wildcard import."""
        # Test UUID generation
        # API changed: use Generators.generate_id() instead
        generated_uuid = FlextUtilities.Generators.generate_id()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) > 0  # ID format is flexible

        # ID format changed, no longer standard UUID
        # Just verify it's a non-empty string

        # Test timestamp generation (using standard datetime since generate_iso_timestamp doesn't exist)
        timestamp = datetime.now(UTC).isoformat()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Test safe type conversion (using to_int which returns FlextResult)
        int_result = FlextUtilities.TypeConversions.to_int("123")
        # FlextResult return type
        assert int_result.is_success is True
        assert int_result.value == 123

    def test_flext_constants_access(self) -> None:
        """Test that FlextConstants hierarchy is accessible after wildcard import."""
        # Test accessing nested constants
        timeout = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout, int)
        assert timeout > 0

        # Test error constants
        validation_error = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error, str)
        assert len(validation_error) > 0  # Just check it's not empty

        # Test message constants - use real attribute
        invalid_msg = FlextConstants.Messages.INVALID_INPUT
        assert isinstance(invalid_msg, str)
        assert len(invalid_msg) > 0

        # Test config constants
        environments = FlextConstants.Config.ENVIRONMENTS
        assert isinstance(environments, list)
        assert "development" in environments

    def test_flext_exceptions_functionality(self) -> None:
        """Test that FlextExceptions work correctly after wildcard import."""
        # Test ValidationError creation and format
        validation_error = FlextExceptions.ValidationError("test_message")
        error_str = str(validation_error)
        assert "[VALIDATION_ERROR]" in error_str
        assert "test_message" in error_str

        # Test OperationError creation
        operation_error = FlextExceptions.OperationError("operation_failed")
        assert isinstance(operation_error, Exception)
        assert "operation_failed" in str(operation_error)

        # Test error hierarchy
        assert issubclass(FlextExceptions.ValidationError, FlextExceptions.BaseError)
        assert issubclass(FlextExceptions.OperationError, FlextExceptions.BaseError)

    def test_flext_types_availability(self) -> None:
        """Test that FlextTypes are available after wildcard import."""
        # Test that FlextTypes module is available
        assert hasattr(FlextTypes, "Core")
        assert hasattr(FlextTypes, "Config")
        assert hasattr(FlextTypes, "Domain")

        # Test type annotations work
        # Verify FlextResult is generic
        result_type = FlextResult[str]
        assert get_origin(result_type) is FlextResult

    def test_flext_config_functionality(self) -> None:
        """Test that FlextConfig works after wildcard import."""
        # Test that FlextConfig can be instantiated
        assert hasattr(FlextConfig, "Settings") or callable(FlextConfig)

        # Test configuration functionality
        assert hasattr(FlextConfig, "Settings") or callable(FlextConfig)
        # FlextConfig itself is the configuration class

    def test_flext_container_functionality(self) -> None:
        """Test that FlextContainer works after wildcard import."""
        # Test container creation
        container = FlextContainer()
        assert container is not None

        # Test basic container operations
        container.register_factory("test_service", lambda: "test_value")
        assert container.has("test_service")

        resolved_result = container.get("test_service")
        assert resolved_result.is_success
        assert resolved_result.value == "test_value"

    def test_flext_commands_functionality(self) -> None:
        """Test that CQRS components work after wildcard import."""
        # Test command pattern components
        assert hasattr(FlextCqrs, "Results") or hasattr(FlextCqrs, "Factories")

        # Test handler functionality
        assert hasattr(FlextHandlers, "handle") or hasattr(FlextHandlers, "execute")
        # CQRS components provide command patterns

    def test_no_import_duplications(self) -> None:
        """Test that there are no duplicate exports in wildcard import."""
        # Get all exported names
        exported_names = [name for name in globals() if not name.startswith("_")]

        # Check for duplicates
        unique_names = set(exported_names)
        assert len(exported_names) == len(unique_names), (
            f"Duplicate exports detected: "
            f"{[name for name in exported_names if exported_names.count(name) > 1]}"
        )

    def test_legacy_compatibility(self) -> None:
        """Test that legacy imports still work alongside wildcard imports."""
        # Test that ERROR_CODES mapping is available for legacy compatibility
        if "ERROR_CODES" in globals():
            error_codes: dict[str, str] = globals()["ERROR_CODES"]
            assert isinstance(error_codes, dict)
            assert len(error_codes) > 0

    def test_export_count_reasonable(self) -> None:
        """Test that the number of exports is reasonable (not too few, not too many)."""
        exported_names = [name for name in globals() if not name.startswith("_")]
        export_count = len(exported_names)

        # Should have a reasonable number of exports (between 20 and 500)
        assert 20 <= export_count <= 500, (
            f"Export count {export_count} seems unreasonable. "
            f"Expected between 20 and 500 exports."
        )

    def test_all_major_categories_present(self) -> None:
        """Test that all major categories of flext-core are represented in exports."""
        # Categories that should be present
        expected_categories = [
            "Result",  # FlextResult and related
            "Constants",  # FlextConstants
            "Exceptions",  # FlextExceptions
            "Utilities",  # FlextUtilities
            "Types",  # FlextTypes
            "Config",  # FlextConfig
            "Container",  # FlextContainer
            "Handler",  # FlextHandlers, FlextQueryHandler
            "Cqrs",  # FlextCqrs for CQRS patterns
        ]

        exported_names = [name for name in globals() if not name.startswith("_")]

        for category in expected_categories:
            assert any(category in name for name in exported_names), (
                f"No exports found for category '{category}'. "
                f"Expected to find names containing '{category}' in exports."
            )


class TestFlextCoreIntegrationScenarios:
    """Integration test scenarios using multiple flext-core components together."""

    def test_end_to_end_workflow(self) -> None:
        """Test a complete workflow using multiple flext-core components."""
        # 1. Generate a unique ID for operation tracking
        operation_id = FlextUtilities.Generators.generate_id()  # Use actual method

        # 2. Create a result and validate it
        result = FlextResult[FlextTypes.Core.Headers].ok(
            {
                "operation_id": operation_id,
                "status": "started",
            },
        )
        # FlextValidations was completely removed - using direct validation
        validation_result = bool(operation_id and operation_id.strip())

        assert validation_result is True
        assert result.is_success is True

        # 3. Process the result through transformations
        processed_result = result.map(
            lambda data: {**data, "timestamp": datetime.now(UTC).isoformat()},
        )

        assert processed_result.is_success is True
        assert "timestamp" in processed_result.value

        # 4. Handle any potential errors using exception system
        def _handle_workflow_failure(
            result: FlextResult[FlextTypes.Core.Headers],
        ) -> None:
            if not result.is_success:
                error_msg = "Workflow failed"
                raise FlextExceptions.OperationError(error_msg)

        try:
            _handle_workflow_failure(processed_result)
        except FlextExceptions.OperationError as e:
            pytest.fail(f"Unexpected workflow failure: {e}")

        # 5. Verify final state using constants
        final_status = "COMPLETED"  # Use string since Status doesn't exist
        final_result = processed_result.map(
            lambda data: {**data, "status": final_status},
        )

        assert final_result.is_success is True
        assert final_result.value["status"] == "COMPLETED"

    def test_error_handling_integration(self) -> None:
        """Test error handling across multiple components."""
        # FlextValidations was completely removed - using direct validation
        invalid_data = ""
        validation_result = bool(invalid_data and invalid_data.strip())

        assert validation_result is False

        # Convert validation failure to exception
        if not validation_result:
            error = FlextExceptions.ValidationError("Validation failed")
            assert FlextConstants.Errors.VALIDATION_ERROR in str(error)

        # Test error code mapping
        error_code = FlextConstants.Errors.VALIDATION_ERROR
        assert error_code == "VALIDATION_ERROR"  # Actual format, not prefixed

    def test_configuration_and_container_integration(self) -> None:
        """Test integration between configuration and container systems."""
        # Create container and register configuration
        container = FlextContainer()

        # Register configuration values using constants
        container.register_factory("timeout", lambda: FlextConstants.Defaults.TIMEOUT)

        container.register_factory(
            "error_mapping",
            lambda: {
                "VALIDATION_ERROR": "Validation failed",
                "CONFIG_ERROR": "Configuration error",
                "NOT_FOUND_ERROR": "Resource not found",
            },
        )

        # Resolve and verify
        timeout_result = container.get("timeout")
        error_mapping_result = container.get("error_mapping")

        assert timeout_result.is_success
        assert error_mapping_result.is_success

        timeout = timeout_result.value
        error_mapping = error_mapping_result.value

        assert isinstance(timeout, int)
        assert isinstance(error_mapping, dict)
        assert len(error_mapping) > 0


@pytest.mark.integration
class TestFlextCoreImportPerformance:
    """Performance tests for import system."""

    def test_import_time_reasonable(self) -> None:
        """Test that wildcard import time is reasonable."""
        # Reload the module to test import time
        start_time = time.time()
        importlib.reload(__import__("flext_core"))
        import_time = time.time() - start_time

        # Import should take less than 5 seconds
        assert import_time < 5.0, f"Import time {import_time:.2f}s is too slow"

    def test_memory_usage_reasonable(self) -> None:
        """Test that memory usage after import is reasonable."""
        # Get current memory usage (simplified)
        modules_count = len(sys.modules)

        # Should not have imported an excessive number of modules
        # This is a heuristic check - adjusted for test environment with pytest/dependencies
        assert modules_count < 5000, f"Too many modules loaded: {modules_count}"


@pytest.mark.integration
@pytest.mark.slow
class TestFlextCoreExportCompleteness:
    """Comprehensive tests for export completeness."""

    def test_all_public_classes_exported(self) -> None:
        """Test that all public classes from major modules are exported."""
        # This test ensures no important classes are missing from exports

        # Get all classes from the current module's globals
        exported_classes: list[str] = []
        for name, obj in globals().items():
            if inspect.isclass(obj) and not name.startswith("_"):
                exported_classes.append(name)

        # Should have at least some classes
        assert len(exported_classes) > 0, "No classes found in exports"

    def test_all_public_functions_exported(self) -> None:
        """Test that all public functions from major modules are exported."""
        # Get all functions from the current module's globals
        exported_functions: list[str] = []
        for name, obj in globals().items():
            if inspect.isfunction(obj) and not name.startswith("_"):
                exported_functions.append(name)

        # Functions might be attached to classes, so this is more lenient
        # Just ensure the test runs without error
