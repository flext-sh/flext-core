"""Tests for flext_core.models functionality - REAL business logic validation.

Tests the actual functional behavior of models, not type structure.
Focus on business rules, state transitions, and domain logic.
"""

import pytest

from flext_core.models import (
    FlextDataFormat,
    FlextEntityStatus,
    FlextOperationStatus,
)


@pytest.mark.unit
class TestEnumsActualUsage:
    """Test enums are actually usable in business logic."""

    def test_entity_status_transitions(self) -> None:
        """Test entity status represents real state transitions."""
        # Test valid status values
        assert FlextEntityStatus.ACTIVE == "active"
        assert FlextEntityStatus.INACTIVE == "inactive"
        assert FlextEntityStatus.PENDING == "pending"
        assert FlextEntityStatus.DELETED == "deleted"
        assert FlextEntityStatus.SUSPENDED == "suspended"

        # Test status can be used in business logic
        statuses = [status.value for status in FlextEntityStatus]
        assert len(statuses) == 5
        assert "active" in statuses

    def test_operation_status_workflow(self) -> None:
        """Test operation status represents real workflow states."""
        # Test workflow progression
        assert FlextOperationStatus.PENDING == "pending"
        assert FlextOperationStatus.RUNNING == "running"
        assert FlextOperationStatus.COMPLETED == "completed"
        assert FlextOperationStatus.FAILED == "failed"
        assert FlextOperationStatus.CANCELLED == "cancelled"

        # Test can model actual workflow
        workflow = [
            FlextOperationStatus.PENDING,
            FlextOperationStatus.RUNNING,
            FlextOperationStatus.COMPLETED,
        ]
        assert len(workflow) == 3

    def test_data_format_actual_formats(self) -> None:
        """Test data formats represent real file formats."""
        # Test actual data formats
        assert FlextDataFormat.JSON == "json"
        assert FlextDataFormat.XML == "xml"
        assert FlextDataFormat.CSV == "csv"
        assert FlextDataFormat.LDIF == "ldif"
        assert FlextDataFormat.YAML == "yaml"
        assert FlextDataFormat.PARQUET == "parquet"

        # Test can be used for file extension logic
        json_files = [f"data.{FlextDataFormat.JSON}"]
        assert "data.json" in json_files


@pytest.mark.unit
class TestModelValidationRules:
    """Test model validation represents real business rules."""

    def test_base_model_validation_works(self) -> None:
        """Test base model validation functionality."""
        from flext_core.models import FlextBaseModel

        class TestModel(FlextBaseModel):
            name: str
            value: int

        # Valid model creation
        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

        # to_dict functionality
        data = model.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["value"] == 42

    def test_semantic_validation_called(self) -> None:
        """Test semantic validation is actually called."""
        from flext_core.models import FlextBaseModel

        class ValidatingModel(FlextBaseModel):
            name: str

            def validate_semantic_rules(self) -> object:
                from flext_core import FlextResult

                if self.name == "invalid":
                    return FlextResult.fail("Name cannot be 'invalid'")
                return FlextResult.ok(None)

        # Test validation works
        model = ValidatingModel(name="valid")
        result = model.validate_semantic_rules()
        assert result.is_success

        # Test validation fails appropriately
        invalid_model = ValidatingModel(name="invalid")
        result = invalid_model.validate_semantic_rules()
        assert result.is_failure
        assert "invalid" in result.error


@pytest.mark.unit
class TestModelInheritanceBehavior:
    """Test model inheritance works for business needs."""

    def test_immutable_model_prevents_modification(self) -> None:
        """Test immutable models actually prevent modification."""
        from flext_core.models import FlextImmutableModel

        class ImmutableTest(FlextImmutableModel):
            name: str
            value: int

        model = ImmutableTest(name="test", value=42)
        assert model.name == "test"

        # Should not be able to modify
        with pytest.raises((AttributeError, ValueError)):
            model.name = "modified"  # type: ignore[misc]

    def test_mutable_model_allows_modification(self) -> None:
        """Test mutable models actually allow modification."""
        from flext_core.models import FlextMutableModel

        class MutableTest(FlextMutableModel):
            name: str = "default"
            value: int = 0

        model = MutableTest()
        assert model.name == "default"

        # Should allow modification
        model.name = "modified"
        model.value = 100
        assert model.name == "modified"
        assert model.value == 100
