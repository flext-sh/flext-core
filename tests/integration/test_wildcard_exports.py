"""Integration tests for flext-core wildcard export system.

This module validates that the wildcard import system works correctly across
the entire flext-core ecosystem, ensuring all major components are properly
exported and functional.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import uuid
from typing import cast

import pytest

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)


class TestFlextCoreWildcardExports:
    """Integration tests for the flext-core wildcard export system."""

    def test_wildcard_import_works(self) -> None:
        """Test that wildcard import from flext_core works without errors."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        spec = cast("importlib.machinery.ModuleSpec", spec)
        assert spec.loader is not None  # Type guard for pyrefly
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

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
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        spec = cast("importlib.machinery.ModuleSpec", spec)
        assert spec.loader is not None  # Type guard for pyrefly
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Get all exported names (excluding builtins)
        exported_names = [
            name
            for name in namespace
            if not name.startswith("_") and name not in dir(__builtins__)
        ]

        # Check for duplicates
        unique_names = set(exported_names)
        duplicates = [name for name in unique_names if exported_names.count(name) > 1]
        assert len(exported_names) == len(unique_names), (
            f"Duplicate exports detected: {duplicates}"
        )

    def test_major_categories_present(self) -> None:
        """Test that major categories of flext-core are represented in exports."""
        # Test wildcard import in a clean namespace
        namespace = {}
        spec = importlib.util.spec_from_file_location(
            "flext_core",
            "src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        spec = cast("importlib.machinery.ModuleSpec", spec)
        assert spec.loader is not None  # Type guard for pyrefly
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

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

        exported_names = [
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
            "src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        spec = cast("importlib.machinery.ModuleSpec", spec)
        assert spec.loader is not None  # Type guard for pyrefly
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        exported_names = [
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


class TestFlextCoreIntegrationScenarios:
    """Integration test scenarios using multiple flext-core components together."""

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
            "src/flext_core/__init__.py",
        )
        if spec is None or spec.loader is None:
            pytest.fail("Failed to create module spec")
        spec = cast("importlib.machinery.ModuleSpec", spec)
        assert spec.loader is not None  # Type guard for pyrefly
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Copy the exports to our test namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)

        # Get all Flext-prefixed exports
        flext_exports = [
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
class TestFlextCoreSystemValidation:
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
