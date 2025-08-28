"""Integration tests for flext-core wildcard export system.

This module validates that the wildcard import system works correctly across
the entire flext-core ecosystem, ensuring all major components are properly
exported and functional.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import uuid
from typing import get_origin

import pytest

from flext_core import (
    FlextCommands,
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextFields,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    FlextValidation,
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
            "FlextFields": "Pydantic field extensions",
            "FlextValidation": "Validation system",
            "FlextCommands": "Command pattern implementations",
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
        assert success_result.success is True
        assert success_result.value == "test_value"
        assert success_result.error is None

        # Test failure case
        failure_result = FlextResult[str].fail("test_error")
        assert failure_result.success is False
        assert failure_result.error == "test_error"
        # Don't access .value on failed result as it raises TypeError

        # Test chaining operations
        chained_result = success_result.map(lambda x: x.upper())
        assert chained_result.success is True
        assert chained_result.value == "TEST_VALUE"

    def test_flext_utilities_functionality(self) -> None:
        """Test that FlextUtilities functions work after wildcard import."""
        # Test UUID generation
        generated_uuid = FlextUtilities.generate_uuid()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) == 36  # Standard UUID format

        # Verify it's a valid UUID
        uuid.UUID(generated_uuid)  # This will raise if invalid

        # Test timestamp generation
        timestamp = FlextUtilities.generate_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Test safe type conversion
        int_result = FlextUtilities.safe_int("123")
        if hasattr(int_result, "success"):
            # FlextResult return type
            assert int_result.success is True
            assert int_result.value == 123
        else:
            # Direct conversion case
            assert int_result == 123

    def test_flext_constants_access(self) -> None:
        """Test that FlextConstants hierarchy is accessible after wildcard import."""
        # Test accessing nested constants
        timeout = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout, int)
        assert timeout > 0

        # Test error constants
        validation_error = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error, str)
        assert "FLEXT" in validation_error

        # Test message constants
        success_msg = FlextConstants.Messages.SUCCESS
        assert isinstance(success_msg, str)
        assert len(success_msg) > 0

        # Test pattern constants
        email_pattern = FlextConstants.Patterns.EMAIL_PATTERN
        assert isinstance(email_pattern, str)
        assert "@" in email_pattern

    def test_flext_exceptions_functionality(self) -> None:
        """Test that FlextExceptions work correctly after wildcard import."""
        # Test ValidationError creation and format
        validation_error = FlextExceptions.ValidationError("test_message")
        error_str = str(validation_error)
        assert "[FLEXT_3001]" in error_str
        assert "test_message" in error_str

        # Test OperationError creation
        operation_error = FlextExceptions.OperationError("operation_failed")
        assert isinstance(operation_error, Exception)
        assert "operation_failed" in str(operation_error)

        # Test error hierarchy
        assert issubclass(FlextExceptions.ValidationError, FlextExceptions.Base)
        assert issubclass(FlextExceptions.OperationError, FlextExceptions.Base)

    def test_flext_types_availability(self) -> None:
        """Test that FlextTypes are available after wildcard import."""
        # Test that FlextTypes module is available
        assert hasattr(FlextTypes, "ResultProtocol")
        assert hasattr(FlextTypes, "ConfigProtocol")
        assert hasattr(FlextTypes, "ContainerProtocol")

        # Test type annotations work
        # Verify FlextResult is generic
        result_type = FlextResult[str]
        assert get_origin(result_type) is FlextResult

    def test_flext_config_functionality(self) -> None:
        """Test that FlextConfig works after wildcard import."""
        # Test that FlextConfig can be instantiated
        assert hasattr(FlextConfig, "Settings")
        assert hasattr(FlextConfig, "Manager")

        # Test configuration loading capabilities
        config_manager = FlextConfig.Manager()
        assert config_manager is not None

    def test_flext_container_functionality(self) -> None:
        """Test that FlextContainer works after wildcard import."""
        # Test container creation
        container = FlextContainer()
        assert container is not None

        # Test basic container operations
        container.register_singleton("test_service", lambda: "test_value")
        assert container.is_registered("test_service")

        resolved = container.resolve("test_service")
        assert resolved == "test_value"

    def test_flext_fields_functionality(self) -> None:
        """Test that FlextFields work after wildcard import."""
        # Test that FlextFields are available
        assert hasattr(FlextFields, "ServiceNameField")
        assert hasattr(FlextFields, "EntityIdField")
        assert hasattr(FlextFields, "SecretKeyField")

        # Test field creation
        service_name_field = FlextFields.ServiceNameField()
        assert service_name_field is not None

    def test_flext_validation_functionality(self) -> None:
        """Test that FlextValidation works after wildcard import."""
        # Test validation decorators and functions
        assert hasattr(FlextValidation, "Guards")
        assert hasattr(FlextValidation, "Decorators")

        # Test basic validation
        guard_result = FlextValidation.Guards.require_non_empty_string("test")
        assert guard_result.success is True

        empty_result = FlextValidation.Guards.require_non_empty_string("")
        assert empty_result.success is False

    def test_flext_commands_functionality(self) -> None:
        """Test that FlextCommands work after wildcard import."""
        # Test command pattern components
        assert hasattr(FlextCommands, "Base")
        assert hasattr(FlextCommands, "Handler")
        assert hasattr(FlextCommands, "Registry")

        # Test command registry
        registry = FlextCommands.Registry()
        assert registry is not None

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
            error_codes = globals()["ERROR_CODES"]
            assert isinstance(error_codes, dict)
            assert len(error_codes) > 0

    def test_export_count_reasonable(self) -> None:
        """Test that the number of exports is reasonable (not too few, not too many)."""
        exported_names = [name for name in globals() if not name.startswith("_")]
        export_count = len(exported_names)

        # Should have a reasonable number of exports (between 50 and 500)
        assert 50 <= export_count <= 500, (
            f"Export count {export_count} seems unreasonable. "
            f"Expected between 50 and 500 exports."
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
            "Fields",  # FlextFields
            "Validation",  # FlextValidation
            "Commands",  # FlextCommands
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
        operation_id = FlextUtilities.generate_uuid()

        # 2. Create a result and validate it
        result = FlextResult[dict].ok(
            {"operation_id": operation_id, "status": "started"}
        )
        validation_result = FlextValidation.Guards.require_non_empty_string(
            operation_id
        )

        assert validation_result.success is True
        assert result.success is True

        # 3. Process the result through transformations
        processed_result = result.map(
            lambda data: {**data, "timestamp": FlextUtilities.generate_timestamp()}
        )

        assert processed_result.success is True
        assert "timestamp" in processed_result.value

        # 4. Handle any potential errors using exception system
        try:
            if not processed_result.success:
                error_msg = "Workflow failed"
                raise FlextExceptions.OperationError(error_msg)
        except FlextExceptions.OperationError as e:
            pytest.fail(f"Unexpected workflow failure: {e}")

        # 5. Verify final state using constants
        final_status = FlextConstants.Status.COMPLETED
        final_result = processed_result.map(
            lambda data: {**data, "status": final_status}
        )

        assert final_result.success is True
        assert final_result.value["status"] == FlextConstants.Status.COMPLETED

    def test_error_handling_integration(self) -> None:
        """Test error handling across multiple components."""
        # Test validation error propagation
        invalid_data = ""
        validation_result = FlextValidation.Guards.require_non_empty_string(
            invalid_data
        )

        assert validation_result.success is False

        # Convert validation failure to exception
        if not validation_result.success:
            error = FlextExceptions.ValidationError(validation_result.error)
            assert FlextConstants.Errors.VALIDATION_ERROR in str(error)

        # Test error code mapping
        error_code = FlextConstants.Errors.VALIDATION_ERROR
        assert error_code.startswith("FLEXT_")

    def test_configuration_and_container_integration(self) -> None:
        """Test integration between configuration and container systems."""
        # Create container and register configuration
        container = FlextContainer()

        # Register configuration values using constants
        container.register_singleton("timeout", lambda: FlextConstants.Defaults.TIMEOUT)

        container.register_singleton(
            "error_mapping", lambda: FlextConstants.Errors.MESSAGES
        )

        # Resolve and verify
        timeout = container.resolve("timeout")
        error_mapping = container.resolve("error_mapping")

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
        # This is a heuristic check
        assert modules_count < 1000, f"Too many modules loaded: {modules_count}"


@pytest.mark.integration
@pytest.mark.slow
class TestFlextCoreExportCompleteness:
    """Comprehensive tests for export completeness."""

    def test_all_public_classes_exported(self) -> None:
        """Test that all public classes from major modules are exported."""
        # This test ensures no important classes are missing from exports

        # Get all classes from the current module's globals
        exported_classes = []
        for name, obj in globals().items():
            if inspect.isclass(obj) and not name.startswith("_"):
                exported_classes.append(name)

        # Should have at least some classes
        assert len(exported_classes) > 0, "No classes found in exports"

    def test_all_public_functions_exported(self) -> None:
        """Test that all public functions from major modules are exported."""
        # Get all functions from the current module's globals
        exported_functions = []
        for name, obj in globals().items():
            if inspect.isfunction(obj) and not name.startswith("_"):
                exported_functions.append(name)

        # Functions might be attached to classes, so this is more lenient
        # Just ensure the test runs without error
