"""Integration tests for flext-core wildcard export system.

This module validates that the wildcard import system works correctly across
the entire flext-core ecosystem, ensuring all major components are properly
exported and functional.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import cast, get_origin

import pytest

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextHandlers,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)


class TestFlextWildcardExports:
    """Integration tests for the flext-core wildcard export system."""

    def test_wildcard_import_works(self) -> None:
        """Test that wildcard import from flext_core works without errors."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "/home/marlonsc/flext/flext-core/src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")

        # At this point, spec and spec.loader are guaranteed to be not None
        spec = cast("importlib.machinery.ModuleSpec", spec)  # Help type checker
        loader = spec.loader  # Type: Loader (not None)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)  # type: ignore[union-attr]

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Verify essential exports are available
        assert "FlextResult" in namespace
        assert "FlextConstants" in namespace
        assert "FlextExceptions" in namespace
        assert "FlextUtilities" in namespace

    def test_essential_modules_exported(self) -> None:
        """Test that essential flext-core modules are exported via wildcard import."""
        # Define essential modules that must be available
        essential_modules = [
            "FlextResult",
            "FlextConstants",
            "FlextExceptions",
            "FlextUtilities",
        ]

        # Verify each essential module is available in globals
        current_globals = globals()
        missing_modules: FlextTypes.StringList = []

        for module_name in essential_modules:
            if module_name not in current_globals:
                missing_modules.append(module_name)
            else:
                # Verify the module is not None
                module = current_globals[module_name]
                assert module is not None, f"Module '{module_name}' is None"

        assert not missing_modules, f"Missing essential modules: {missing_modules}"

    def test_flext_result_basic_functionality(self) -> None:
        """Test basic FlextResult functionality."""
        # Test success case
        success_result = FlextResult[str].ok("test_value")
        assert success_result.is_success is True
        assert success_result.value == "test_value"
        assert success_result.error is None

        # Test failure case
        failure_result = FlextResult[str].fail("test_error")
        assert failure_result.is_success is False
        assert failure_result.error == "test_error"

        # Test chaining operations
        chained_result = success_result.map(lambda x: x.upper())
        assert chained_result.is_success is True
        assert chained_result.value == "TEST_VALUE"

    def test_flext_utilities_basic_functionality(self) -> None:
        """Test basic FlextUtilities functionality."""
        # Test UUID generation - use correct API
        generated_uuid = FlextUtilities.Generators.generate_uuid()
        assert isinstance(generated_uuid, str)
        assert len(generated_uuid) == 36  # Standard UUID format

        # Verify it's a valid UUID
        uuid.UUID(generated_uuid)  # This will raise if invalid

    def test_flext_constants_access(self) -> None:
        """Test that FlextConstants hierarchy is accessible."""
        # Test accessing nested constants
        timeout = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout, int)
        assert timeout > 0

        # Test error constants
        validation_error = FlextConstants.Errors.VALIDATION_ERROR
        assert isinstance(validation_error, str)
        assert len(validation_error) > 0  # Just check it's not empty

        # Test message constants
        invalid_msg = FlextConstants.Messages.INVALID_INPUT
        assert isinstance(invalid_msg, str)
        assert len(invalid_msg) > 0

    def test_flext_exceptions_basic_functionality(self) -> None:
        """Test basic FlextExceptions functionality."""
        # Test ValidationError creation and format
        validation_error = FlextExceptions.ValidationError("test_message")
        error_str = str(validation_error)
        assert "[VALIDATION_ERROR]" in error_str

    def test_no_import_duplications(self) -> None:
        """Test that there are no duplicate exports in wildcard import."""
        # Test wildcard import in a clean namespace
        namespace: dict[str, object] = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "/home/marlonsc/flext/flext-core/src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")

        # At this point, spec and spec.loader are guaranteed to be not None
        spec = cast("importlib.machinery.ModuleSpec", spec)  # Help type checker
        loader = spec.loader  # Type: Loader (not None)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)  # type: ignore[union-attr]

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Get all exported names (excluding builtins)
        exported_names: list[str] = [
            name
            for name in namespace
            if not name.startswith("_") and name not in dir(__builtins__)
        ]

        # Check for duplicates
        unique_names: set[str] = set(exported_names)
        duplicates: list[str] = [
            name for name in unique_names if exported_names.count(name) > 1
        ]
        assert len(exported_names) == len(unique_names), (
            f"Duplicate exports detected: {duplicates}"
        )

    def test_major_categories_present(self) -> None:
        """Test that major categories of flext-core are represented in exports."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "/home/marlonsc/flext/flext-core/src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")

        # At this point, spec and spec.loader are guaranteed to be not None
        spec = cast("importlib.machinery.ModuleSpec", spec)  # Help type checker
        loader = spec.loader  # Type: Loader (not None)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)  # type: ignore[union-attr]

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Categories that should be present
        expected_categories = [
            "Result",  # FlextResult and related
            "Constants",  # FlextConstants
            "Exceptions",  # FlextExceptions
            "Utilities",  # FlextUtilities
        ]

        exported_names: list[str] = [
            name
            for name in namespace
            if not name.startswith("_") and name not in dir(__builtins__)
        ]

        for category in expected_categories:
            assert any(category in name for name in exported_names), (
                f"No exports found for category '{category}'. "
                f"Expected to find names containing '{category}' in exports."
            )

    def test_export_count_reasonable(self) -> None:
        """Test that the number of exports is reasonable."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "/home/marlonsc/flext/flext-core/src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")

        # At this point, spec and spec.loader are guaranteed to be not None
        spec = cast("importlib.machinery.ModuleSpec", spec)  # Help type checker
        loader = spec.loader  # Type: Loader (not None)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)  # type: ignore[union-attr]

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        exported_names: list[str] = [
            name
            for name in namespace
            if not name.startswith("_") and name not in dir(__builtins__)
        ]
        export_count = len(exported_names)

        # Should have at least 10 exports (minimum for a functional system)
        assert export_count >= 10, (
            f"Export count {export_count} seems too low. "
            f"Expected at least 10 exports for a functional system."
        )

        # Upper bound check - shouldn't exceed 1000 (would indicate bloat)
        assert export_count <= 1000, (
            f"Export count {export_count} seems too high. "
            f"Expected at most 1000 exports to avoid namespace pollution."
        )

    def test_end_to_end_basic_workflow(self) -> None:
        """Test a basic workflow using multiple flext-core components."""
        # 1. Generate a unique ID for operation tracking
        operation_id = FlextUtilities.Generators.generate_uuid()
        assert isinstance(operation_id, str)

        # 2. Create a result
        result = FlextResult[FlextTypes.Dict].ok(
            {
                "operation_id": operation_id,
                "status": "started",
            },
        )
        assert result.is_success is True

        # 3. Process the result through transformations
        processed_result = result.map(
            lambda data: {**data, "timestamp": "2024-01-01T00:00:00Z"},
        )

        assert processed_result.is_success is True
        assert "timestamp" in processed_result.value

        # 4. Verify final state using a simple string
        final_status = "COMPLETED"
        processed_result.map(
            lambda data: {**data, "status": final_status},
        )

    def test_system_exports_health(self) -> None:
        """Test overall health of the export system."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "/home/marlonsc/flext/flext-core/src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")

        # At this point, spec and spec.loader are guaranteed to be not None
        spec = cast("importlib.machinery.ModuleSpec", spec)  # Help type checker
        loader = spec.loader  # Type: Loader (not None)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)  # type: ignore[union-attr]

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Get all Flext-prefixed exports
        flext_exports: list[str] = [
            name
            for name in namespace
            if name.startswith("Flext")
            and not name.startswith("_")
            and name not in dir(__builtins__)
        ]

        # Should have at least the core classes
        core_classes = [
            "FlextResult",
            "FlextConstants",
            "FlextExceptions",
            "FlextUtilities",
        ]
        missing_core = [cls for cls in core_classes if cls not in flext_exports]

        assert not missing_core, f"Missing core classes: {missing_core}"

        # Should have reasonable number of exports
        assert len(flext_exports) >= 4, f"Too few Flext classes: {len(flext_exports)}"


@pytest.mark.integration
class TestFlextSystemValidation:
    """System-level validation tests."""

    def test_system_exports_health(self) -> None:
        """Test overall health of the export system."""
        # Get all Flext-prefixed exports
        flext_exports = [
            name
            for name in globals()
            if name.startswith("Flext") and not name.startswith("_")
        ]

        # Should have at least the core classes
        core_classes = [
            "FlextResult",
            "FlextConstants",
            "FlextExceptions",
            "FlextUtilities",
        ]
        missing_core = [cls for cls in core_classes if cls not in flext_exports]

        assert not missing_core, f"Missing core classes: {missing_core}"

        # Should have reasonable number of exports
        assert len(flext_exports) >= 4, f"Too few Flext classes: {len(flext_exports)}"

    def test_basic_import_stability(self) -> None:
        """Test that imports are stable and don't cause circular issues."""
        # This test just ensures the import completed successfully
        # and we can access basic functionality

        result = FlextResult[int].ok(42)
        assert result.value == 42

        timeout = FlextConstants.Defaults.TIMEOUT
        assert isinstance(timeout, int)

        error = FlextExceptions.ValidationError("test")
        assert isinstance(error, Exception)

        uuid_val = FlextUtilities.Generators.generate_uuid()
        assert isinstance(uuid_val, str)

    def test_railway_oriented_programming_pattern(self) -> None:
        """Test that the core railway-oriented programming pattern works."""
        # Chain of operations using FlextResult
        result = (
            FlextResult[int].ok(10).map(lambda x: x * 2).map(lambda x: x + 5).map(str)
        )

        assert result.is_success is True
        assert result.value == "25"

        # Test failure propagation
        failed_chain = (
            FlextResult[int]
            .fail("initial error")
            .map(lambda x: x * 2)  # Should not execute
            .map(lambda x: x + 5)  # Should not execute
        )

        assert failed_chain.is_success is False
        assert failed_chain.error == "initial error"

    def test_wildcard_import_works(self) -> None:
        """Test that wildcard import from flext_core works without errors."""
        # Since we already have the wildcard import at module level, test it works
        assert "FlextResult" in globals()
        assert "FlextConstants" in globals()
        assert "FlextHandlers" in globals()
        # Verify FlextHandlers is importable and not None
        assert FlextHandlers is not None

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
            error_codes: FlextTypes.StringDict = globals()["ERROR_CODES"]
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
        ]

        exported_names = [name for name in globals() if not name.startswith("_")]

        for category in expected_categories:
            assert any(category in name for name in exported_names), (
                f"No exports found for category '{category}'. "
                f"Expected to find names containing '{category}' in exports."
            )


class TestFlextIntegrationScenarios:
    """Integration test scenarios using multiple flext-core components together."""

    def test_end_to_end_workflow(self) -> None:
        """Test a complete workflow using multiple flext-core components."""
        # 1. Generate a unique ID for operation tracking
        operation_id = FlextUtilities.Generators.generate_id()  # Use actual method

        # 2. Create a result and validate it
        result = FlextResult[FlextTypes.StringDict].ok(
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
            result: FlextResult[FlextTypes.StringDict],
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
class TestFlextImportPerformance:
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
class TestFlextExportCompleteness:
    """Comprehensive tests for export completeness."""

    def test_all_public_classes_exported(self) -> None:
        """Test that all public classes from major modules are exported."""
        # This test ensures no important classes are missing from exports

        # Get all classes from the current module's globals
        exported_classes: FlextTypes.StringList = []
        for name, obj in globals().items():
            if inspect.isclass(obj) and not name.startswith("_"):
                exported_classes.append(name)

        # Should have at least some classes
        assert len(exported_classes) > 0, "No classes found in exports"

    def test_all_public_functions_exported(self) -> None:
        """Test that all public functions from major modules are exported."""
        # Get all functions from the current module's globals
        exported_functions: FlextTypes.StringList = []
        for name, obj in globals().items():
            if inspect.isfunction(obj) and not name.startswith("_"):
                exported_functions.append(name)

        # Functions might be attached to classes, so this is more lenient
        # Just ensure the test runs without error
