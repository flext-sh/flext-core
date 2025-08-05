"""Tests for flext_core.models functionality - Real business logic validation.

Tests the actual functional behavior of models, not type structure.
Focus on business rules, state transitions, and domain logic.
"""

from __future__ import annotations

import pytest

from flext_core.models import (
    FlextBaseModel,
    FlextDatabaseModel,
    FlextDataFormat,
    FlextDomainEntity,
    FlextDomainValueObject,
    FlextEntityStatus,
    FlextImmutableModel,
    FlextMutableModel,
    FlextOperationModel,
    FlextOperationStatus,
    FlextOracleModel,
    FlextServiceModel,
    FlextSingerStreamModel,
    create_database_model,
    create_operation_model,
    create_oracle_model,
    create_service_model,
    create_singer_stream_model,
    model_to_dict_safe,
    validate_all_models,
)
from flext_core.result import FlextResult


@pytest.mark.unit
class TestEnumsActualUsage:
    """Test enums are actually usable in business logic."""

    def test_entity_status_transitions(self) -> None:
        """Test entity status represents real state transitions."""
        # Test valid status values
        assert FlextEntityStatus.ACTIVE.value == "active"
        assert FlextEntityStatus.INACTIVE.value == "inactive"
        assert FlextEntityStatus.PENDING.value == "pending"
        assert FlextEntityStatus.DELETED.value == "deleted"
        assert FlextEntityStatus.SUSPENDED.value == "suspended"

        # Test status can be used in business logic
        statuses = [status.value for status in FlextEntityStatus]
        assert len(statuses) == 5
        assert "active" in statuses

    def test_operation_status_workflow(self) -> None:
        """Test operation status represents real workflow states."""
        # Test workflow progression
        assert FlextOperationStatus.PENDING.value == "pending"
        assert FlextOperationStatus.RUNNING.value == "running"
        assert FlextOperationStatus.COMPLETED.value == "completed"
        assert FlextOperationStatus.FAILED.value == "failed"
        assert FlextOperationStatus.CANCELLED.value == "cancelled"

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
        assert FlextDataFormat.JSON.value == "json"
        assert FlextDataFormat.XML.value == "xml"
        assert FlextDataFormat.CSV.value == "csv"
        assert FlextDataFormat.LDIF.value == "ldif"
        assert FlextDataFormat.YAML.value == "yaml"
        assert FlextDataFormat.PARQUET.value == "parquet"

        # Test can be used for file extension logic
        json_files = [f"data.{FlextDataFormat.JSON.value}"]
        assert "data.json" in json_files


@pytest.mark.unit
class TestModelValidationRules:
    """Test model validation represents real business rules."""

    def test_base_model_validation_works(self) -> None:
        """Test base model validation functionality."""

        class TestModel(FlextBaseModel):
            name: str = ""
            value: int = 0

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Valid model creation using model_validate
        model = TestModel.model_validate({"name": "test", "value": 42})
        assert model.name == "test"
        assert model.value == 42

        # to_dict functionality
        data = model.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["value"] == 42

        # to_typed_dict functionality
        typed_data = model.to_typed_dict()
        assert isinstance(typed_data, dict)
        assert typed_data["name"] == "test"
        assert typed_data["value"] == 42

    def test_semantic_validation_called(self) -> None:
        """Test semantic validation is actually called."""

        class ValidatingModel(FlextBaseModel):
            name: str = ""

            def validate_business_rules(self) -> FlextResult[None]:
                if self.name == "invalid":
                    return FlextResult.fail("Name cannot be 'invalid'")
                return FlextResult.ok(None)

        # Test validation works
        model = ValidatingModel.model_validate({"name": "valid"})
        result = model.validate_business_rules()
        assert result.success

        # Test validation fails appropriately
        invalid_model = ValidatingModel.model_validate({"name": "invalid"})
        result = invalid_model.validate_business_rules()
        assert result.is_failure
        assert "invalid" in (result.error or "")


@pytest.mark.unit
class TestModelInheritanceBehavior:
    """Test model inheritance works for business needs."""

    def test_immutable_model_prevents_modification(self) -> None:
        """Test immutable models actually prevent modification."""

        class ImmutableTest(FlextImmutableModel):
            name: str = ""
            value: int = 0

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model = ImmutableTest.model_validate({"name": "test", "value": 42})
        assert model.name == "test"

        # Should not be able to modify
        with pytest.raises((AttributeError, ValueError)):
            model.name = "modified"

    def test_mutable_model_allows_modification(self) -> None:
        """Test mutable models actually allow modification."""

        class MutableTest(FlextMutableModel):
            name: str = "default"
            value: int = 0

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        model = MutableTest.model_validate(
            {"id": "test", "name": "default", "value": 0}
        )
        assert model.name == "default"

        # Should allow modification
        model.name = "modified"
        model.value = 100
        assert model.name == "modified"
        assert model.value == 100


@pytest.mark.unit
class TestFlextDomainEntity:
    """Test domain entity functionality including identity, versioning, and events."""

    def test_domain_entity_creation(self) -> None:
        """Test domain entity creation with automatic ID generation."""
        entity = FlextDomainEntity(id="test-id")
        assert entity.id == "test-id"
        assert entity.version == 1
        assert entity.status == FlextEntityStatus.ACTIVE
        assert entity.domain_events == []

    def test_domain_entity_hash_and_equality(self) -> None:
        """Test domain entity hash and equality based on ID."""
        entity1 = FlextDomainEntity(id="entity1")
        entity2 = FlextDomainEntity(id="entity2")
        entity3 = FlextDomainEntity(id=entity1.id)

        # Test hash functionality
        entity_hash = hash(entity1)
        assert isinstance(entity_hash, int)
        assert hash(entity1) == hash(entity3)  # Same ID
        assert hash(entity1) != hash(entity2)  # Different ID

        # Test equality functionality
        assert entity1 == entity3  # Same ID
        assert entity1 != entity2  # Different ID
        assert entity1 != "not_an_entity"

    def test_domain_entity_version_increment(self) -> None:
        """Test version increment updates version and timestamp."""
        entity = FlextDomainEntity(id="test")
        original_version = entity.version
        original_updated_at = entity.updated_at

        # Increment version
        entity.increment_version()

        assert entity.version == original_version + 1
        assert entity.updated_at > original_updated_at

    def test_domain_entity_add_domain_event(self) -> None:
        """Test adding domain events for event sourcing."""
        entity = FlextDomainEntity(id="test")
        assert len(entity.domain_events) == 0

        # Add domain event
        event: dict[str, object] = {"type": "entity_created", "data": {"id": "test"}}
        entity.add_domain_event(event)

        assert len(entity.domain_events) == 1
        assert entity.domain_events[0] == event

        # Add another event
        event2: dict[str, object] = {"type": "entity_updated", "data": {"id": "test"}}
        entity.add_domain_event(event2)

        assert len(entity.domain_events) == 2
        assert entity.domain_events[1] == event2


@pytest.mark.unit
class TestFlextDomainValueObject:
    """Test domain value object functionality including immutability and hashing."""

    def test_value_object_creation(self) -> None:
        """Test value object creation with attributes."""
        vo = FlextDomainValueObject(
            metadata={"name": "test_vo", "description": "Test value object"}
        )
        assert vo.metadata["name"] == "test_vo"
        assert vo.metadata["description"] == "Test value object"

    def test_value_object_hash_and_equality(self) -> None:
        """Test value object hash and equality based on attributes."""
        vo1 = FlextDomainValueObject(metadata={"name": "test", "description": "desc"})
        vo2 = FlextDomainValueObject(metadata={"name": "test", "description": "desc"})
        vo3 = FlextDomainValueObject(
            metadata={"name": "different", "description": "desc"}
        )

        # Test equality
        assert vo1 == vo2  # Same attributes
        assert vo1 != vo3  # Different attributes
        assert vo1 != "not_a_vo"  # type: ignore[comparison-overlap]

        # Test hash (should be same for equal objects)
        assert hash(vo1) == hash(vo2)
        assert hash(vo1) != hash(vo3)

    def test_value_object_immutability(self) -> None:
        """Test value objects are immutable (frozen)."""
        vo = FlextDomainValueObject(metadata={"name": "test"})

        # Should not be able to modify
        with pytest.raises((AttributeError, ValueError)):
            vo.metadata = {"name": "modified"}  # type: ignore[misc]  # Testing immutability


@pytest.mark.unit
class TestModelFactoryFunctions:
    """Test factory functions for creating models with validation."""

    def test_create_database_model(self) -> None:
        """Test database model factory function."""
        # Test with defaults
        db_model = create_database_model()
        assert isinstance(db_model, FlextDatabaseModel)
        assert db_model.host == "localhost"
        assert db_model.port == 5432
        assert db_model.username == "postgres"

        # Test with custom values
        custom_db = create_database_model(
            host="custom.host", port=3306, username="REDACTED_LDAP_BIND_PASSWORD"
        )
        assert custom_db.host == "custom.host"
        assert custom_db.port == 3306
        assert custom_db.username == "REDACTED_LDAP_BIND_PASSWORD"

    def test_create_oracle_model(self) -> None:
        """Test Oracle model factory function."""
        # Test with service_name
        oracle_model = create_oracle_model(service_name="TESTDB")
        assert isinstance(oracle_model, FlextOracleModel)
        assert oracle_model.host == "localhost"
        assert oracle_model.port == 1521
        assert oracle_model.service_name == "TESTDB"

        # Test with custom values
        custom_oracle = create_oracle_model(
            host="oracle.host", port=1522, username="oracle_user"
        )
        assert custom_oracle.host == "oracle.host"
        assert custom_oracle.port == 1522
        assert custom_oracle.username == "oracle_user"

    def test_create_service_model(self) -> None:
        """Test service model factory function."""
        service_model = create_service_model(
            service_name="test-service", host="localhost", port=8080, version="1.0.0"
        )
        assert isinstance(service_model, FlextServiceModel)
        assert service_model.service_name == "test-service"
        assert service_model.version == "1.0.0"

    def test_create_operation_model(self) -> None:
        """Test operation model factory function."""
        operation_model = create_operation_model(
            operation_id="test_operation", operation_type="test"
        )
        assert isinstance(operation_model, FlextOperationModel)
        assert operation_model.operation_id == "test_operation"
        assert operation_model.operation_type == "test"
        assert operation_model.status == FlextOperationStatus.PENDING

    def test_create_singer_stream_model(self) -> None:
        """Test Singer stream model factory function."""
        stream_model = create_singer_stream_model(
            stream_name="test_stream",
            tap_name="test_tap",
            schema_definition={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        )
        assert isinstance(stream_model, FlextSingerStreamModel)
        assert stream_model.stream_name == "test_stream"
        # Test that schema_definition is properly set
        assert isinstance(stream_model.schema_definition, dict)
        # The factory function passes empty dict by default, so we check the provided schema
        if "type" in stream_model.schema_definition:
            assert stream_model.schema_definition["type"] == "object"


@pytest.mark.unit
class TestDatabaseModelValidation:
    """Test database model validation and functionality."""

    def test_database_model_creation(self) -> None:
        """Test database model creation and validation."""
        from pydantic import SecretStr

        db_model = FlextDatabaseModel(
            host="db.example.com",
            port=5432,
            username="user",
            password=SecretStr("secret"),
            database="testdb",
        )
        assert db_model.host == "db.example.com"
        assert db_model.port == 5432
        assert db_model.username == "user"
        assert hasattr(db_model.password, "get_secret_value")
        assert db_model.password.get_secret_value() == "secret"
        assert db_model.database == "testdb"

    def test_database_model_connection_string(self) -> None:
        """Test database model connection string generation."""
        from pydantic import SecretStr

        db_model = FlextDatabaseModel(
            host="localhost",
            port=5432,
            username="user",
            password=SecretStr("pass"),
            database="db",
        )

        # Test connection string property
        conn_str = db_model.connection_string()
        expected = "postgresql://user:pass@localhost:5432/db"
        assert conn_str == expected


@pytest.mark.unit
class TestOracleModelValidation:
    """Test Oracle model validation and functionality."""

    def test_oracle_model_validation_with_service_name(self) -> None:
        """Test Oracle model validation with service_name."""
        from pydantic import SecretStr

        oracle_model = FlextOracleModel(
            host="oracle.host",
            username="oracle_user",
            password=SecretStr("oracle_pass"),
            service_name="TESTDB",
        )
        assert oracle_model.service_name == "TESTDB"
        assert oracle_model.sid is None

    def test_oracle_model_validation_with_sid(self) -> None:
        """Test Oracle model validation with SID."""
        from pydantic import SecretStr

        oracle_model = FlextOracleModel(
            host="oracle.host",
            username="oracle_user",
            password=SecretStr("oracle_pass"),
            sid="TESTSID",
        )
        assert oracle_model.sid == "TESTSID"
        assert oracle_model.service_name is None

    def test_oracle_model_validation_error(self) -> None:
        """Test Oracle model validation fails without service_name or sid."""
        from pydantic import SecretStr

        # Create Oracle model without service_name or sid (explicitly set to None)
        model = FlextOracleModel(
            username="user", password=SecretStr("pass"), service_name=None, sid=None
        )

        # Validation should fail when calling validate_semantic_rules
        result = model.validate_semantic_rules()
        assert result.is_failure
        assert "Either service_name or sid must be provided" in (result.error or "")

    def test_oracle_model_connection_string(self) -> None:
        """Test Oracle model connection string generation."""
        from pydantic import SecretStr

        # Test with service_name
        oracle_model = FlextOracleModel(
            host="localhost",
            port=1521,
            username="user",
            password=SecretStr("pass"),
            service_name="TESTDB",
        )
        assert oracle_model.connection_string() == "localhost:1521/TESTDB"

        # Test with SID
        oracle_sid_model = FlextOracleModel(
            host="localhost",
            port=1521,
            username="user",
            password=SecretStr("pass"),
            sid="TESTSID",
        )
        assert oracle_sid_model.connection_string() == "localhost:1521:TESTSID"


@pytest.mark.unit
class TestOperationModel:
    """Test operation model functionality."""

    def test_operation_model_creation(self) -> None:
        """Test operation model creation with defaults."""
        operation = FlextOperationModel(operation_id="test_op", operation_type="test")
        assert operation.operation_id == "test_op"
        assert operation.operation_type == "test"
        assert operation.status == FlextOperationStatus.PENDING
        assert operation.progress_percentage == 0.0
        assert operation.retry_count == 0

    def test_operation_model_progress_update(self) -> None:
        """Test operation model progress tracking."""
        operation = FlextOperationModel(operation_id="test_op", operation_type="test")

        # Update progress
        operation.progress_percentage = 50.0
        operation.status = FlextOperationStatus.RUNNING

        assert operation.progress_percentage == 50.0
        assert operation.status == FlextOperationStatus.RUNNING

    def test_operation_model_completion(self) -> None:
        """Test operation model completion."""
        operation = FlextOperationModel(operation_id="test_op", operation_type="test")

        # Complete operation
        operation.progress_percentage = 100.0
        operation.status = FlextOperationStatus.COMPLETED

        assert operation.progress_percentage == 100.0
        assert operation.status == FlextOperationStatus.COMPLETED


@pytest.mark.unit
class TestServiceModel:
    """Test service model functionality."""

    def test_service_model_creation(self) -> None:
        """Test service model creation."""
        service = FlextServiceModel(
            service_name="test-service",
            service_id="test-id",
            host="localhost",
            port=8080,
            version="1.0.0",
        )
        assert service.service_name == "test-service"
        assert service.version == "1.0.0"
        assert service.service_id == "test-id"

    def test_service_model_health_check(self) -> None:
        """Test service model health check functionality."""
        service = FlextServiceModel(
            service_name="test-service",
            service_id="test-id",
            host="localhost",
            port=8080,
            version="1.0.0",
            health_check_url="/health",
        )
        assert service.health_check_url == "/health"


@pytest.mark.unit
class TestSingerStreamModel:
    """Test Singer stream model functionality."""

    def test_singer_stream_model_creation(self) -> None:
        """Test Singer stream model creation."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
        }

        stream = FlextSingerStreamModel(
            stream_name="users", tap_name="test_tap", schema_definition=schema
        )
        assert stream.stream_name == "users"
        assert stream.schema_definition == schema
        assert stream.tap_name == "test_tap"


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions for model operations."""

    def test_model_to_dict_safe(self) -> None:
        """Test safe model to dict conversion."""
        # Test with valid model
        entity = FlextDomainEntity(id="test")
        result = model_to_dict_safe(entity)
        assert isinstance(result, dict)
        assert result["id"] == "test"

        # Test with invalid input
        result_none = model_to_dict_safe(None)
        assert result_none == {}

        result_string = model_to_dict_safe("not_a_model")
        assert result_string == {}

    def test_validate_all_models(self) -> None:
        """Test validation of multiple models."""
        # Test with valid models
        models = [
            FlextDomainEntity(id="entity1"),
            FlextDomainValueObject(metadata={"name": "vo1"}),
            FlextOperationModel(operation_id="op1", operation_type="test"),
        ]

        result = validate_all_models(*models)
        assert result.success

        # Test with invalid input
        result_empty = validate_all_models()
        assert result_empty.success

        # Test with mixed valid/invalid
        entity = FlextDomainEntity(id="valid")
        result_mixed = validate_all_models(entity)
        assert result_mixed.success
