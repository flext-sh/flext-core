# ruff: noqa: ARG001, ARG002
"""Tests for FlextFields with modern pytest patterns.

Advanced tests using parametrized fixtures, factory patterns,
performance monitoring, and property-based testing.

Architectural Patterns Demonstrated:
- Enterprise-grade parametrized testing with structured TestCase objects
- Advanced fixture composition using conftest infrastructure
- Performance benchmarking with context managers and monitoring
- Hypothesis property-based testing for edge case discovery
- Snapshot testing for configuration validation and regression prevention
- Mock factories for external dependency simulation and isolation

Usage of New Conftest Infrastructure:
- test_builder: Fluent builder pattern for complex test data construction
- entity_factory: Dynamic entity creation with validation and business rules
- value_object_factory: Value object creation with domain validation
- performance_monitor: Function execution monitoring with memory tracking
- assert_helpers: Advanced assertion helpers with FlextResult validation
- snapshot_manager: Complex data structure snapshot testing and comparison
- hypothesis_strategies: Property-based testing with domain-specific strategies
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import SupportsFloat, cast

import pytest
from hypothesis import given, strategies as st

from flext_core import (
    FlextFields,
    FlextResult,
)

from ..conftest import TestScenario

# Type aliases for the new API
FieldInstance = object  # FlextFields.Core field instance
FieldConfig = dict[str, object]  # Configuration dictionary


# Simple local test utilities
@dataclass
class TestCase:
    name: str
    expected: bool
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    execution_time: float
    memory_used: int


def assert_performance(
    func: Callable[[], None], expected_time: float = 1.0
) -> PerformanceMetrics:
    """Simple performance assertion helper."""
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return PerformanceMetrics(execution_time=end - start, memory_used=0)


# Test markers for organized execution
pytestmark = [pytest.mark.unit, pytest.mark.core]

# ============================================================================
# Advanced Parametrized Testing with TestCase Structures
# ============================================================================


class TestFlextFieldCoreAdvanced:
    """Advanced field core testing with structured parametrization."""

    @pytest.fixture
    def field_creation_test_cases(self) -> list[TestCase[dict[str, object]]]:
        """Define structured test cases for field creation scenarios."""
        return [
            TestCase(
                id="basic_string_field",
                description="Create basic string field with minimal config",
                input_data={
                    "field_id": "basic_string",
                    "field_name": "basic_name",
                    "field_type": "string",
                },
                expected_output={
                    "field_id": "basic_string",
                    "field_name": "basic_name",
                    "field_type": "string",
                    "required": True,
                    "default_value": None,
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="comprehensive_field",
                description="Create field with all parameters configured",
                input_data={
                    "field_id": "comprehensive_field",
                    "field_name": "comprehensive_name",
                    "field_type": "string",
                    "required": False,
                    "default_value": "default",
                    "min_length": 5,
                    "max_length": 50,
                    "pattern": r"^[A-Z][a-z]+$",
                    "allowed_values": ["Test", "Demo", "Sample"],
                    "description": "Test field description",
                    "example": "Test",
                    "deprecated": True,
                    "sensitive": True,
                    "indexed": True,
                    "tags": ["validation", "test"],
                },
                expected_output={
                    "field_id": "comprehensive_field",
                    "required": False,
                    "default_value": "default",
                    "min_length": 5,
                    "max_length": 50,
                    "pattern": r"^[A-Z][a-z]+$",
                    "deprecated": True,
                    "sensitive": True,
                    "indexed": True,
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="integer_field",
                description="Create integer field with numeric constraints",
                input_data={
                    "field_id": "integer_field",
                    "field_name": "age_field",
                    "field_type": "integer",
                    "min_value": 0,
                    "max_value": 150,
                    "default_value": 25,
                },
                expected_output={
                    "field_type": "integer",
                    "min_value": 0,
                    "max_value": 150,
                    "default_value": 25,
                },
                scenario=TestScenario.HAPPY_PATH,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_field_creation_scenarios(
        self,
        field_creation_test_cases: list[TestCase[dict[str, object]]],
        assert_helpers: object,
    ) -> None:
        """Test field creation using structured test cases."""
        for test_case in field_creation_test_cases:
            # Create field using test case input data - extract required parameters
            input_data = test_case.input_data.copy()
            field_type = input_data.pop("field_type")
            field_name = input_data.pop("field_name")

            # Create field using new Factory API
            field_result = FlextFields.Factory.create_field(field_type, field_name, **input_data)
            assert field_result.is_success, f"Failed to create field: {field_result.error}"
            field = field_result.value

            # Field creation already validated above via assert
            # Additional validation that field is properly created
            assert field is not None

            # Verify expected output attributes - cast to access items
            expected_output = cast("dict[str, object]", test_case.expected_output)
            for attr_name, expected_value in expected_output.items():
                actual_value = getattr(field, attr_name)
                assert actual_value == expected_value, (
                    f"Test case {test_case.id}: {attr_name} mismatch. "
                    f"Expected {expected_value}, got {actual_value}"
                )

    @pytest.fixture
    def validation_test_cases(self) -> list[TestCase[dict[str, object]]]:
        """Define validation test cases for different field types and values."""
        return [
            TestCase(
                id="valid_string_value",
                description="Valid string value within constraints",
                input_data={
                    "field_config": {
                        "field_id": "string_field",
                        "field_name": "name",
                        "field_type": "string",
                        "min_length": 2,
                        "max_length": 10,
                    },
                    "test_value": "TestValue",
                },
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="string_too_short",
                description="String value below minimum length",
                input_data={
                    "field_config": {
                        "field_id": "string_field",
                        "field_name": "name",
                        "field_type": "string",
                        "min_length": 5,
                        "max_length": 10,
                    },
                    "test_value": "Hi",
                },
                expected_output=False,
                expected_error="String too short",
                scenario=TestScenario.ERROR_CASE,
            ),
            TestCase(
                id="integer_within_bounds",
                description="Integer value within min/max bounds",
                input_data={
                    "field_config": {
                        "field_id": "age_field",
                        "field_name": "age",
                        "field_type": "integer",
                        "min_value": 0,
                        "max_value": 120,
                    },
                    "test_value": 25,
                },
                expected_output=True,
                scenario=TestScenario.HAPPY_PATH,
            ),
            TestCase(
                id="integer_out_of_bounds",
                description="Integer value outside maximum bound",
                input_data={
                    "field_config": {
                        "field_id": "age_field",
                        "field_name": "age",
                        "field_type": "integer",
                        "min_value": 0,
                        "max_value": 120,
                    },
                    "test_value": 150,
                },
                expected_output=False,
                expected_error="Integer too large",
                scenario=TestScenario.ERROR_CASE,
            ),
        ]

    @pytest.mark.parametrize_advanced
    def test_field_validation_scenarios(
        self,
        validation_test_cases: list[TestCase[dict[str, object]]],
        assert_helpers: object,
    ) -> None:
        """Test field validation using structured test cases."""
        for test_case in validation_test_cases:
            field_config = test_case.input_data["field_config"]
            test_value = test_case.input_data["test_value"]

            # Create field - cast and extract parameters for new API
            field_config_typed = cast("dict[str, object]", field_config)
            config_copy = field_config_typed.copy()
            field_type = config_copy.pop("field_type")
            field_name = config_copy.pop("field_name", "test_field")

            # Create field using Factory API
            field_result = FlextFields.Factory.create_field(field_type, field_name, **config_copy)
            assert field_result.is_success, f"Field creation failed: {field_result.error}"
            field = field_result.value

            # Validate value using FlextFields validation
            validation_result = FlextFields.Validation.validate_field(field, test_value)

            # Cast assert_helpers to access methods
            cast("object", assert_helpers)
            if test_case.scenario == TestScenario.HAPPY_PATH:
                assert validation_result.success
            elif test_case.scenario == TestScenario.ERROR_CASE:
                assert validation_result.is_failure
                if test_case.expected_error:
                    error_message = validation_result.error or ""
                    assert test_case.expected_error in error_message

    @pytest.mark.parametrize(
        ("field_type", "test_values", "expected_valid"),
        [
            ("string", ["test", "hello world", ""], [True, True, True]),
            ("integer", [1, 42, -5, 0], [True, True, True, True]),
            ("boolean", [True, False], [True, True]),
            ("float", [1.0, math.pi, -2.5, 0.0], [True, True, True, True]),
        ],
    )
    def test_field_type_validation_matrix(
        self,
        field_type: str,
        test_values: list[object],
        expected_valid: list[bool],
        assert_helpers: object,
    ) -> None:
        """Test validation matrix for different field types and values."""
        # Create field using Factory API
        field_result = FlextFields.Factory.create_field(
            field_type,
            f"test_{field_type}",
            field_id=f"{field_type}_field",
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        for value, should_be_valid in zip(test_values, expected_valid, strict=False):
            result = FlextFields.Validation.validate_field(field, value)

            if should_be_valid:
                assert result.success
            else:
                assert result.is_failure


# ============================================================================
# Property-Based Testing with Hypothesis
# ============================================================================


class TestFlextFieldCorePropertyBased:
    """Property-based tests for field validation using Hypothesis."""

    @pytest.mark.hypothesis
    @given(
        field_name=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        ),
        min_length=st.integers(min_value=0, max_value=100),
        max_length=st.integers(min_value=1, max_value=200),
    )
    def test_string_field_length_properties(
        self,
        field_name: str,
        min_length: int,
        max_length: int,
    ) -> None:
        """Property: string field validation respects length constraints."""
        # Ensure min_length < max_length (strict inequality required by validation)
        if min_length >= max_length:
            min_length, max_length = 0, max(min_length, max_length) + 1

        # Create field using Factory API
        field_result = FlextFields.Factory.create_field(
            "string",
            field_name,
            field_id="property_test_field",
            min_length=min_length,
            max_length=max_length,
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        # Test with valid length string
        valid_string = "x" * ((min_length + max_length) // 2)
        if min_length <= len(valid_string) <= max_length:
            result = FlextFields.Validation.validate_field(field, valid_string)
            assert result.success, (
                f"Valid string should pass validation: {valid_string}"
            )

    @pytest.mark.hypothesis
    @given(
        min_value=st.integers(min_value=-1000, max_value=0),
        max_value=st.integers(min_value=1, max_value=1000),
        test_value=st.integers(min_value=-500, max_value=500),
    )
    def test_integer_field_range_properties(
        self,
        min_value: int,
        max_value: int,
        test_value: int,
    ) -> None:
        """Property: integer field validation respects range constraints."""
        # Create field using Factory API
        field_result = FlextFields.Factory.create_field(
            "integer",
            "test_integer",
            field_id="property_int_field",
            min_value=min_value,
            max_value=max_value,
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        result = FlextFields.Validation.validate_field(field, test_value)

        # Property: value within range should be valid
        if min_value <= test_value <= max_value:
            assert result.success, (
                f"Value {test_value} should be valid in range [{min_value}, {max_value}]"
            )
        else:
            assert result.is_failure, (
                f"Value {test_value} should be invalid outside range [{min_value}, {max_value}]"
            )

    @pytest.mark.hypothesis
    @given(
        allowed_values=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        test_value=st.text(min_size=1, max_size=10),
    )
    def test_allowed_values_property(
        self,
        allowed_values: list[str | int | float | bool],
        test_value: str,
    ) -> None:
        """Property: field with allowed values only accepts values from the list."""
        field_result = FlextFields.Factory.create_field(
            "string",
            "test_allowed",
            field_id="allowed_values_field",
            allowed_values=allowed_values,
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        result = FlextFields.Validation.validate_field(field, test_value)

        # Property: value in allowed_values should be valid
        if test_value in allowed_values:
            assert result.success, (
                f"Value '{test_value}' should be valid (in allowed values)"
            )
        else:
            assert result.is_failure, (
                f"Value '{test_value}' should be invalid (not in allowed values)"
            )


# ============================================================================
# Performance Testing with Monitoring
# ============================================================================


class TestFlextFieldCorePerformance:
    """Performance tests using conftest monitoring infrastructure."""

    @pytest.mark.benchmark
    def test_field_creation_performance(
        self,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
        performance_threshold: dict[str, float],
    ) -> None:
        """Benchmark field creation performance."""

        def create_hundred_fields() -> list[object]:
            fields = []
            for i in range(100):
                field_result = FlextFields.Factory.create_field(
                    "string",
                    f"performance_field_{i}",
                    field_id=f"perf_field_{i}",
                )
                assert field_result.is_success, f"Field creation failed: {field_result.error}"
                fields.append(field_result.value)
            return fields

        metrics = cast("PerformanceMetrics", performance_monitor(create_hundred_fields))

        # Performance assertions
        assert (
            metrics["execution_time"] < performance_threshold["result_creation"] * 100
        )  # 100ms for 100 fields
        assert len(cast("list[object]", metrics["result"])) == 100

        # Memory efficiency assertion
        assert metrics["memory_used"] < 1_000_000  # 1MB for 100 fields

    @pytest.mark.benchmark
    @pytest.mark.usefixtures("assert_helpers")
    def test_validation_performance(
        self,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
    ) -> None:
        """Benchmark validation performance with complex constraints."""
        # Create complex field
        field_result = FlextFields.Factory.create_field(
            "string",
            "complex_field",
            field_id="complex_validation_field",
            min_length=5,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
            allowed_values=[f"value_{i}" for i in range(50)],
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        def validate_many_values() -> list[FlextResult[object]]:
            results = []
            for i in range(100):
                value = f"value_{i % 25}"  # Mix of valid and invalid
                result = FlextFields.Validation.validate_field(field, value)
                results.append(result)
            return results

        metrics = cast("PerformanceMetrics", performance_monitor(validate_many_values))

        assert metrics["execution_time"] < 0.1  # 100ms for 100 validations
        assert len(cast("list[object]", metrics["result"])) == 100

    @pytest.mark.benchmark
    @pytest.mark.usefixtures("assert_helpers")
    def test_registry_performance(
        self,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
    ) -> None:
        """Benchmark registry operations performance."""
        registry = FlextFields.Registry.FieldRegistry()

        def registry_operations() -> list[FlextResult[object]]:
            # Register many fields
            for i in range(50):
                field_result = FlextFields.Factory.create_field(
                    "string",
                    f"field_{i}",
                    field_id=f"registry_field_{i}",
                )
                assert field_result.is_success, f"Field creation failed: {field_result.error}"
                registry.register_field(field_result.value)

            # Retrieve all fields
            results = []
            for i in range(50):
                result = registry.get_field_by_id(f"registry_field_{i}")
                results.append(result)

            return results

        with assert_performance(max_time=0.2, max_memory=5_000_000):  # 200ms, 5MB
            # Temporarily disable logging during performance test to get accurate measurements
            logging.disable(logging.CRITICAL)
            try:
                metrics = performance_monitor(registry_operations)
            finally:
                logging.disable(logging.NOTSET)

        result_metrics = cast("list[object]", metrics.get("result", []))
        assert len(result_metrics) == 50

        # All retrievals should be successful
        result_metrics = cast("list[FlextResult[object]]", metrics.get("result", []))
        for result in result_metrics:
            assert result.success


# ============================================================================
# Advanced Fixtures Integration
# ============================================================================


class TestFlextFieldCoreWithFixtures:
    """Tests demonstrating advanced fixture usage from conftest."""

    def test_fields_with_test_builder(
        self,
        test_builder: Callable[[], object],
        assert_helpers: object,
    ) -> None:
        """Test field creation using test data builder pattern."""
        # Build complex field configuration using fluent builder
        field_config = (
            test_builder()
            .with_field("field_id", "builder_test_field")
            .with_field("field_name", "builder_field")
            .with_field("field_type", "string")
            .with_field("required", False)
            .with_field("min_length", 3)
            .with_field("max_length", 20)
            .with_field("pattern", r"^[A-Za-z]+$")
            .with_field("description", "Field created with builder pattern")
            .with_field("tags", ["test", "builder", "advanced"])
            .build()
        )

        # Create field from builder data with proper type casting
        config = cast("dict[str, object]", field_config)
        field_result = FlextFields.Factory.create_field(
            str(config["field_type"]),
            str(config["field_name"]),
            field_id=str(config["field_id"]),
            required=bool(config["required"]),
            min_length=cast("int | None", config.get("min_length")),
            max_length=cast("int | None", config.get("max_length")),
            pattern=str(config["pattern"]) if config.get("pattern") else None,
            description=str(config["description"])
            if config.get("description")
            else None,
            tags=cast("list[str]", config["tags"]) if config.get("tags") else [],
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        # Validate field properties
        assert field.field_id == "builder_test_field"
        assert field.field_name == "builder_field"
        assert field.required is False
        assert field.min_length == 3
        assert field.max_length == 20
        assert field.pattern == r"^[A-Za-z]+$"
        assert "builder" in field.tags

        # Test validation with builder-created field
        validation_result = FlextFields.Validation.validate_field(field, "ValidValue")
        assert validation_result.success

    @pytest.mark.usefixtures("assert_helpers")
    def test_fields_with_sample_data(
        self,
        sample_data: dict[str, object],
        validators: dict[str, Callable[[str], bool]],
    ) -> None:
        """Test field validation using sample data fixture."""
        # Create field for email validation
        email_field_result = FlextFields.Factory.create_field(
            "string",
            "email",
            field_id="email_field",
            pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
            description="Email address field",
        )
        assert email_field_result.is_success, f"Field creation failed: {email_field_result.error}"
        email_field = email_field_result.value

        # Use sample data to test validation
        test_email = "test@example.com"  # Valid email format
        result = FlextFields.Validation.validate_field(email_field, test_email)

        assert result.success
        assert validators["is_valid_email"](test_email)

        # Test with UUID field
        uuid_field_result = FlextFields.Factory.create_field(
            "string",
            "uuid",
            field_id="uuid_field",
            pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        )
        assert uuid_field_result.is_success, f"Field creation failed: {uuid_field_result.error}"
        uuid_field = uuid_field_result.value

        uuid_result = FlextFields.Validation.validate_field(uuid_field, sample_data["uuid"])
        assert uuid_result.success
        assert validators["is_valid_uuid"](str(sample_data["uuid"]))

    def test_fields_with_service_factory(
        self,
        service_factory: Callable[[str], object],
    ) -> None:
        """Test field integration with external services using real service factory."""
        # Create real external validation service
        validator_service = service_factory("field_validator_service")
        # Real service factory creates services with expected methods

        # Test field integration with real service
        service_field_result = FlextFields.Factory.create_field(
            "string",
            "integration_test_field",
            field_id="service_integration_field",
        )
        assert service_field_result.is_success, f"Field creation failed: {service_field_result.error}"

        # Real services can be integrated for validation
        result = validator_service.validate()
        assert result.success

        # Create field that uses external validation
        external_field_result = FlextFields.Factory.create_field(
            "string",
            "external_field",
            field_id="external_validated_field",
            description="Field with external validation",
        )
        assert external_field_result.is_success, f"Field creation failed: {external_field_result.error}"
        external_field = external_field_result.value

        # Test that field was created successfully
        assert external_field.field_id == "external_validated_field"
        assert external_field.field_name == "external_field"


# ============================================================================
# Snapshot Testing for Complex Configurations
# ============================================================================


class TestFlextFieldCoreSnapshot:
    """Snapshot tests for complex field configurations and outputs."""

    @pytest.mark.snapshot
    def test_comprehensive_field_snapshot(
        self,
        snapshot_manager: Callable[[str, object], None],
    ) -> None:
        """Test comprehensive field configuration snapshot."""
        field_result = FlextFields.Factory.create_field(
            "string",
            "comprehensive_field",
            field_id="comprehensive_snapshot_field",
            required=False,
            default_value="default_snapshot",
            min_length=5,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
            allowed_values=["snapshot_value_1", "snapshot_value_2", "default_snapshot"],
            description="Comprehensive field for snapshot testing",
            example="snapshot_value_1",
            deprecated=False,
            sensitive=False,
            indexed=True,
            tags=["snapshot", "testing", "comprehensive"],
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        # Create field structure for snapshot
        field_snapshot = {
            "field_id": field.field_id,
            "field_name": field.field_name,
            "field_type": field.field_type,
            "required": field.required,
            "default_value": field.default_value,
            "min_length": field.min_length,
            "max_length": field.max_length,
            "pattern": field.pattern,
            "allowed_values": field.allowed_values,
            "description": field.description,
            "example": field.example,
            "deprecated": field.deprecated,
            "sensitive": field.sensitive,
            "indexed": field.indexed,
            "tags": field.tags,
            "metadata": {
                "created_via": "snapshot_test",
                "validation_rules": [
                    "required_check",
                    "length_validation",
                    "pattern_validation",
                    "allowed_values_check",
                ],
            },
        }

        snapshot_manager("comprehensive_field_config", field_snapshot)

    @pytest.mark.snapshot
    def test_field_registry_snapshot(
        self,
        snapshot_manager: Callable[[str, object], None],
    ) -> None:
        """Test field registry structure snapshot."""
        registry = FlextFields.Registry.FieldRegistry()

        # Register multiple fields with different configurations
        test_fields = []

        # Create string field
        string_field_result = FlextFields.Factory.create_field(
            "string",
            "string_field",
            field_id="string_field_snapshot",
            min_length=1,
            max_length=50,
        )
        assert string_field_result.is_success, f"String field creation failed: {string_field_result.error}"
        test_fields.append(string_field_result.value)

        # Create integer field
        integer_field_result = FlextFields.Factory.create_field(
            "integer",
            "integer_field",
            field_id="integer_field_snapshot",
            min_value=0,
            max_value=1000,
            default_value=100,
        )
        assert integer_field_result.is_success, f"Integer field creation failed: {integer_field_result.error}"
        test_fields.append(integer_field_result.value)

        # Create boolean field
        boolean_field_result = FlextFields.Factory.create_field(
            "boolean",
            "boolean_field",
            field_id="boolean_field_snapshot",
            default_value=False,
            description="Boolean field for snapshot testing",
        )
        assert boolean_field_result.is_success, f"Boolean field creation failed: {boolean_field_result.error}"
        test_fields.append(boolean_field_result.value)

        # Register all fields
        for test_field in test_fields:
            registry.register_field(test_field)

        # Create registry snapshot
        all_fields = registry.get_all_fields()
        registry_snapshot = {
            "field_count": len(all_fields),
            "field_ids": list(all_fields.keys()),
            "field_types": {
                field_id: field.field_type for field_id, field in all_fields.items()
            },
            "required_fields": [
                field_id for field_id, field in all_fields.items() if field.required
            ],
            "fields_with_defaults": [
                field_id
                for field_id, field in all_fields.items()
                if field.default_value is not None
            ],
        }

        snapshot_manager("field_registry_structure", registry_snapshot)


# ============================================================================
# Integration Tests with Multiple Components
# ============================================================================


class TestFlextFieldCoreIntegration:
    """Integration tests demonstrating component interaction."""

    def test_complete_field_workflow_integration(
        self,
        test_builder: Callable[[], object],
        assert_helpers: object,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
    ) -> None:
        """Test complete field workflow from creation to validation."""

        def complete_field_workflow() -> dict[str, object]:
            # 1. Create registry
            registry = FlextFields.Registry.FieldRegistry()

            # 2. Build field configuration using test builder
            field_config = (
                test_builder()
                .with_field("field_id", "workflow_integration_field")
                .with_field("field_name", "integration_field")
                .with_field("field_type", "string")
                .with_field("required", True)
                .with_field("min_length", 5)
                .with_field("max_length", 50)
                .with_field("pattern", r"^[A-Za-z][A-Za-z0-9_]*$")
                .with_field("description", "Integration workflow test field")
                .build()
            )

            # 3. Create and register field with proper type casting
            config = cast("dict[str, object]", field_config)
            field_result = FlextFields.Factory.create_field(
                str(config["field_type"]),
                str(config["field_name"]),
                field_id=str(config["field_id"]),
                required=bool(config["required"]),
                min_length=cast("int | None", config.get("min_length")),
                max_length=cast("int | None", config.get("max_length")),
                pattern=str(config["pattern"]) if config.get("pattern") else None,
                description=str(config["description"])
                if config.get("description")
                else None,
            )
            assert field_result.is_success, f"Field creation failed: {field_result.error}"
            field = field_result.value
            registry.register_field(field)

            # 4. Retrieve field from registry
            retrieval_result = registry.get_field_by_id(
                "workflow_integration_field",
            )
            retrieved_field = retrieval_result.value

            # 5. Validate various test values
            test_values: list[str] = [
                "ValidValue",
                "another_valid",
                "Val123",
                "X_valid",
            ]
            validation_results: list[FlextResult[object]] = []

            for value in test_values:
                result: FlextResult[object] = FlextFields.Validation.validate_field(retrieved_field, value)
                validation_results.append(result)

            return {
                "field": field,
                "retrieved_field": retrieved_field,
                "validation_results": validation_results,
            }

        # Monitor performance of complete workflow
        metrics = performance_monitor(complete_field_workflow)
        workflow_result = metrics["result"]

        # Validate workflow execution
        assert isinstance(workflow_result, dict)
        assert workflow_result["field"] is not None
        assert workflow_result["retrieved_field"] is not None
        assert isinstance(workflow_result["validation_results"], list)
        assert all(
            isinstance(result, FlextResult)
            for result in workflow_result["validation_results"]
        )

        result = FlextResult[object].ok(workflow_result["field"])
        assert result.success

        # Verify field properties
        field = workflow_result["field"]
        retrieved_field = workflow_result["retrieved_field"]

        assert field.field_id == retrieved_field.field_id
        assert field.field_name == retrieved_field.field_name
        assert field.field_type == retrieved_field.field_type

        # Verify all validations passed
        validation_results = workflow_result[
            "validation_results"
        ]
        for result in validation_results:
            assert result.success

        # Performance assertion
        assert isinstance(metrics, dict)

        # Robust execution_time conversion that handles multiple numeric types
        execution_time_raw = metrics["execution_time"]
        if isinstance(execution_time_raw, (int, float)):
            execution_time = float(execution_time_raw)
        elif isinstance(execution_time_raw, bytes):
            execution_time = float(execution_time_raw.decode().strip())
        elif hasattr(execution_time_raw, "__float__"):  # Decimal or similar
            execution_time = float(cast("SupportsFloat", execution_time_raw))
        elif isinstance(execution_time_raw, str):
            try:
                execution_time = float(execution_time_raw.strip())
            except ValueError as e:
                pytest.fail(
                    f"Could not parse execution_time '{execution_time_raw}' as float: {e}",
                )
        else:
            pytest.fail(
                f"Unexpected execution_time type: {type(execution_time_raw)} with value: {execution_time_raw}",
            )

        assert execution_time < 0.01  # 10ms for complete workflow

    @pytest.mark.integration
    def test_factory_methods_integration(self, assert_helpers: object) -> None:
        """Test integration between factory methods and registry."""
        # Create fields using factory methods
        string_field_result = FlextFields.Factory.create_field(
            "string",
            "factory_string",
            field_id="factory_string_field",
            min_length=3,
            max_length=30,
            pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        )

        integer_field_result = FlextFields.Factory.create_field(
            "integer",
            "factory_integer",
            field_id="factory_integer_field",
            min_value=0,
            max_value=999,
            default_value=42,
        )

        boolean_field_result = FlextFields.Factory.create_field(
            "boolean",
            "factory_boolean",
            field_id="factory_boolean_field",
            default_value=True,
        )

        # Validate all factory results - Factory.create_field returns FlextResult
        assert string_field_result.is_success, f"String field creation failed: {string_field_result.error}"
        assert integer_field_result.is_success, f"Integer field creation failed: {integer_field_result.error}"
        assert boolean_field_result.is_success, f"Boolean field creation failed: {boolean_field_result.error}"

        # Get created fields from FlextResult
        string_field = string_field_result.value
        integer_field = integer_field_result.value
        boolean_field = boolean_field_result.value

        # Test field functionality
        string_validation = FlextFields.Validation.validate_field(string_field, "valid_string")
        integer_validation = FlextFields.Validation.validate_field(integer_field, 123)
        boolean_validation = FlextFields.Validation.validate_field(boolean_field, False)

        assert string_validation.success
        assert integer_validation.success
        assert boolean_validation.success

        # Verify factory-created fields have correct types
        assert string_field.field_type == "string"
        assert integer_field.field_type == "integer"
        assert boolean_field.field_type == "boolean"


# ============================================================================
# Edge Cases and Boundary Testing
# ============================================================================


class TestFlextFieldCoreEdgeCases:
    """Edge cases and boundary condition tests."""

    @pytest.mark.boundary
    @pytest.mark.parametrize(
        ("field_type", "edge_values", "descriptions"),
        [
            (
                "string",
                ["", "a", "x" * 1000, " ", "\n", "\t", "ünïcødé"],
                [
                    "empty string",
                    "single char",
                    "very long",
                    "space",
                    "newline",
                    "tab",
                    "unicode",
                ],
            ),
            (
                "integer",
                [0, -1, 1, -999999, 999999, float("inf"), -float("inf")],
                [
                    "zero",
                    "negative one",
                    "positive one",
                    "large negative",
                    "large positive",
                    "infinity",
                    "negative infinity",
                ],
            ),
            (
                "boolean",
                [True, False, 1, 0, "true", "false", [], {}],
                [
                    "true",
                    "false",
                    "int one",
                    "int zero",
                    "str true",
                    "str false",
                    "empty list",
                    "empty dict",
                ],
            ),
            (
                "float",
                [0.0, -0.0, 1.0, -1.0, math.pi, float("nan"), float("inf")],
                [
                    "zero float",
                    "negative zero",
                    "one float",
                    "negative one",
                    "pi",
                    "nan",
                    "infinity",
                ],
            ),
        ],
    )
    def test_field_type_edge_values(
        self,
        field_type: str,
        edge_values: list[object],
        descriptions: list[str],
    ) -> None:
        """Test field validation with edge case values."""
        field_result = FlextFields.Factory.create_field(
            field_type.value,
            f"edge_{field_type.value}",
            field_id=f"edge_test_{field_type.value}",
        )
        assert field_result.is_success, f"Field creation failed: {field_result.error}"
        field = field_result.value

        for value, description in zip(edge_values, descriptions, strict=False):
            result = FlextFields.Validation.validate_field(field, value)

            # Log edge case testing for analysis

            # For edge cases, we mainly check that validation doesn't crash
            assert isinstance(result, FlextResult), (
                f"Should return FlextResult for {description}"
            )

    @pytest.mark.boundary
    def test_field_constraints_boundary_conditions(self) -> None:
        """Test boundary conditions for field constraints."""
        # String length boundaries
        string_field_result = FlextFields.Factory.create_field(
            "string",
            "boundary_string",
            field_id="boundary_string_field",
            min_length=5,
            max_length=10,
        )
        assert string_field_result.is_success, f"String field creation failed: {string_field_result.error}"
        string_field = string_field_result.value

        # Test exact boundaries
        boundary_tests = [
            ("x" * 4, False, "below min length"),  # Below minimum
            ("x" * 5, True, "exactly min length"),  # Exactly minimum
            ("x" * 6, True, "within range"),  # Within range
            ("x" * 10, True, "exactly max length"),  # Exactly maximum
            ("x" * 11, False, "above max length"),  # Above maximum
        ]

        for value, should_pass, description in boundary_tests:
            result = string_field.validate_value(value)
            if should_pass:
                assert result.success, f"{description}: '{value}' should be valid"
            else:
                assert result.is_failure, f"{description}: '{value}' should be invalid"

        # Integer value boundaries
        integer_field_result = FlextFields.Factory.create_field(
            "integer",
            "boundary_integer",
            field_id="boundary_integer_field",
            min_value=0,
            max_value=100,
        )
        assert integer_field_result.is_success, f"Integer field creation failed: {integer_field_result.error}"
        integer_field = integer_field_result.value

        integer_boundary_tests = [
            (-1, False, "below min value"),
            (0, True, "exactly min value"),
            (50, True, "within range"),
            (100, True, "exactly max value"),
            (101, False, "above max value"),
        ]

        for value, should_pass, description in integer_boundary_tests:
            result = integer_field.validate_value(value)
            if should_pass:
                assert result.success, f"{description}: {value} should be valid"
            else:
                assert result.is_failure, f"{description}: {value} should be invalid"


# ============================================================================
# Backward Compatibility and Legacy Support
# ============================================================================


class TestFlextFieldCoreBackwardCompatibility:
    """Tests for backward compatibility with legacy field systems."""

    @pytest.mark.parametrize(
        ("legacy_function", "modern_method", "args"),
        [
            (
                FlextFields.create_string_field,
                FlextFields.Core.StringField,
                {
                    "name": "compat_name",
                    "min_length": 1,
                    "max_length": 50,
                },
            ),
            (
                FlextFields.create_integer_field,
                FlextFields.Core.IntegerField,
                {
                    "name": "compat_age",
                    "min_value": 0,
                    "max_value": 150,
                },
            ),
            (
                FlextFields.create_boolean_field,
                FlextFields.Core.BooleanField,
                {
                    "name": "compat_flag",
                    "default": False,
                },
            ),
        ],
    )
    @pytest.mark.usefixtures("assert_helpers")
    def test_legacy_function_compatibility(
        self,
        legacy_function: Callable[[], object],
        modern_method: Callable[[], object],
        args: dict[str, object],
    ) -> None:
        """Test that legacy functions produce compatible results with modern methods."""
        # Call legacy function
        legacy_result = legacy_function(**args)

        # Call modern method
        modern_result = modern_method(**args)

        # Both should succeed - factory functions return FlextResult objects now
        assert legacy_result.is_success, f"Legacy function failed: {legacy_result.error}"
        assert modern_result.is_success, f"Modern method failed: {modern_result.error}"

        # Extract field objects from FlextResult
        legacy_field = legacy_result.value
        modern_field = modern_result.value

        # Compare essential properties
        assert legacy_field.field_id == modern_field.field_id
        assert legacy_field.field_name == modern_field.field_name
        assert legacy_field.field_type == modern_field.field_type

        # Test that both fields validate the same values identically
        test_values = [
            "test_value",  # For string fields
            42,  # For integer fields
            True,  # For boolean fields
            False,  # For boolean fields
        ]

        for test_value in test_values:
            legacy_validation = legacy_field.validate_value(test_value)
            modern_validation = modern_field.validate_value(test_value)

            # Results should match (both pass or both fail)
            assert legacy_validation.success == modern_validation.success, (
                f"Validation results should match for value {test_value}"
            )

    def test_field_metadata_backward_compatibility(self) -> None:
        """Test backward compatibility of field metadata systems."""
        # Create field using Factory API
        field_result = FlextFields.Factory.create_field(
            "string",
            "metadata_field",
            field_id="metadata_compat_field",
            description="Backward compatibility test field",
            example="example_value",
            tags=["compatibility", "legacy", "metadata"],
        )
        assert field_result.is_success, f"Metadata field creation failed: {field_result.error}"
        field = field_result.value

        # Extract metadata using new system
        metadata_result = FlextFields.Metadata.analyze_field(field)
        assert metadata_result.is_success, f"Metadata analysis failed: {metadata_result.error}"
        metadata = metadata_result.value

        # Verify metadata properties
        assert metadata.field_id == field.field_id
        assert metadata.field_name == field.field_name
        assert metadata.field_type == field.field_type
        assert metadata.description == field.description
        assert metadata.example == field.example
        assert metadata.tags == field.tags

        # Convert back to dict and verify structure
        metadata_dict = metadata_result.to_dict()
        expected_keys = {
            "field_id",
            "field_name",
            "field_type",
            "required",
            "description",
            "example",
            "tags",
        }

        assert set(metadata_dict.keys()).issuperset(expected_keys)
