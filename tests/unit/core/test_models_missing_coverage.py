"""Additional tests to cover missing lines in models.py - focused on 63% → 90%+ coverage."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from flext_core.models import (
    FlextDomainEntity,
    FlextDomainValueObject,
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


@pytest.mark.unit
class TestDomainEntityMethods:
    """Test domain entity methods that need coverage."""

    def test_domain_entity_hash(self) -> None:
        """Test domain entity hash method."""
        entity = FlextDomainEntity(id="test-id")
        hash_value = hash(entity)
        assert isinstance(hash_value, int)

    def test_domain_entity_equality(self) -> None:
        """Test domain entity equality method."""
        entity1 = FlextDomainEntity(id="test-id")
        entity2 = FlextDomainEntity(id="test-id")
        entity3 = FlextDomainEntity(id="different-id")

        # Same ID should be equal
        assert entity1 == entity2
        # Different ID should not be equal
        assert entity1 != entity3
        # Different type should not be equal
        assert entity1 != "not_an_entity"

    def test_domain_entity_increment_version(self) -> None:
        """Test domain entity version increment."""
        entity = FlextDomainEntity(id="test-id")
        original_version = entity.version
        original_updated_at = entity.updated_at

        entity.increment_version()

        assert entity.version == original_version + 1
        assert entity.updated_at > original_updated_at

    def test_domain_entity_add_domain_event(self) -> None:
        """Test adding domain events."""
        entity = FlextDomainEntity(id="test-id")
        event = {"type": "created", "data": {"id": "test-id"}}

        entity.add_domain_event(event)

        assert len(entity.domain_events) == 1
        assert entity.domain_events[0] == event


@pytest.mark.unit
class TestDomainValueObjectMethods:
    """Test domain value object methods that need coverage."""

    def test_value_object_hash(self) -> None:
        """Test value object hash method."""
        vo = FlextDomainValueObject()
        hash_value = hash(vo)
        assert isinstance(hash_value, int)

    def test_value_object_equality(self) -> None:
        """Test value object equality method."""
        vo1 = FlextDomainValueObject()
        vo2 = FlextDomainValueObject()

        # Same attributes should be equal
        assert vo1 == vo2
        # Different type should not be equal
        assert vo1 != "not_a_vo"


@pytest.mark.unit
class TestOracleModelValidation:
    """Test Oracle model validation methods."""

    def test_oracle_model_validator(self) -> None:
        """Test Oracle model field validator."""
        # Create valid Oracle model with service_name
        oracle = FlextOracleModel(
            username="user", password=SecretStr("pass"), service_name="TEST"
        )
        assert oracle.service_name == "TEST"

    def test_oracle_model_validation_method(self) -> None:
        """Test Oracle model validation method."""
        # Test the custom model_validate method
        data = {"username": "user", "password": "pass", "service_name": "TEST"}
        oracle = FlextOracleModel.model_validate(data)
        assert oracle.service_name == "TEST"

        # Test validation failure via validate_semantic_rules
        invalid_model = FlextOracleModel(
            username="user",
            password="pass",
            # No service_name or sid
        )
        validation_result = invalid_model.validate_semantic_rules()
        assert validation_result.is_failure
        assert "Either service_name or sid must be provided" in validation_result.error

    def test_oracle_model_connection_string(self) -> None:
        """Test Oracle model connection string property."""
        # Test with service_name
        oracle = FlextOracleModel(
            host="localhost",
            port=1521,
            username="user",
            password=SecretStr("pass"),
            service_name="TEST",
        )
        assert oracle.connection_string() == "localhost:1521/TEST"

        # Test with SID
        oracle_sid = FlextOracleModel(
            host="localhost",
            port=1521,
            username="user",
            password=SecretStr("pass"),
            sid="TESTSID",
        )
        assert oracle_sid.connection_string() == "localhost:1521:TESTSID"


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for model creation."""

    def test_create_database_model(self) -> None:
        """Test database model factory."""
        db_model = create_database_model()
        assert db_model.host == "localhost"
        assert db_model.port == 5432

        # Test with kwargs
        custom_db = create_database_model(host="custom", port=3306)
        assert custom_db.host == "custom"
        assert custom_db.port == 3306

    def test_create_oracle_model(self) -> None:
        """Test Oracle model factory."""
        oracle_model = create_oracle_model(service_name="TEST")
        assert oracle_model.host == "localhost"
        assert oracle_model.service_name == "TEST"

        # Test with kwargs
        custom_oracle = create_oracle_model(host="oracle.host", port=1522)
        assert custom_oracle.host == "oracle.host"
        assert custom_oracle.port == 1522

    def test_create_operation_model(self) -> None:
        """Test operation model factory."""
        operation = create_operation_model(
            operation_id="op-123", operation_type="test_operation"
        )
        assert operation.operation_id == "op-123"
        assert operation.operation_type == "test_operation"
        assert operation.status == FlextOperationStatus.PENDING

    def test_create_service_model(self) -> None:
        """Test service model factory."""
        service = create_service_model(
            service_name="test-service", host="localhost", port=8080
        )
        assert service.service_name == "test-service"
        assert service.host == "localhost"
        assert service.port == 8080

    def test_create_singer_stream_model(self) -> None:
        """Test Singer stream model factory."""
        stream = create_singer_stream_model(
            stream_name="users", tap_name="tap-postgres"
        )
        assert stream.stream_name == "users"
        assert stream.tap_name == "tap-postgres"


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""

    def test_model_to_dict_safe(self) -> None:
        """Test safe model to dict conversion."""
        # Test with valid model
        entity = FlextDomainEntity(id="test-id")

        # First test: check direct model conversion works
        direct_result = entity.to_dict()
        assert isinstance(direct_result, dict)
        assert len(direct_result) > 0

        # Now test the utility function
        result = model_to_dict_safe(entity)
        assert isinstance(result, dict)
        # The function should return the same data
        assert result == direct_result

        # Test with None
        result_none = model_to_dict_safe(None)
        assert result_none == {}

        # Test with string (invalid model)
        result_str = model_to_dict_safe("not_a_model")
        assert result_str == {}

    def test_validate_all_models(self) -> None:
        """Test validate all models function."""
        # Test with valid models (individual arguments)
        entity = FlextDomainEntity(id="entity1")
        value_object = FlextDomainValueObject()
        result = validate_all_models(entity, value_object)
        assert result.success

        # Test with empty call
        result_empty = validate_all_models()
        assert result_empty.success

        # Note: invalid models test removed since function expects FlextBaseModel types


@pytest.mark.unit
class TestModelSpecificBehavior:
    """Test specific model behaviors to increase coverage."""

    def test_operation_model_fields(self) -> None:
        """Test operation model with all fields."""
        operation = FlextOperationModel(
            operation_id="op-123", operation_type="data_sync"
        )

        # Test default values
        assert operation.status == FlextOperationStatus.PENDING
        assert operation.progress_percentage == 0.0
        assert operation.retry_count == 0
        assert operation.max_retries == 3

    def test_service_model_fields(self) -> None:
        """Test service model with required fields."""
        service = FlextServiceModel(
            service_name="test-service",
            service_id="svc-123",
            host="localhost",
            port=8080,
            version="1.0.0",
        )

        assert service.service_name == "test-service"
        assert service.service_id == "svc-123"
        assert service.host == "localhost"
        assert service.port == 8080

    def test_singer_stream_model_fields(self) -> None:
        """Test Singer stream model with required fields."""
        stream = FlextSingerStreamModel(stream_name="users", tap_name="tap-postgres")

        assert stream.stream_name == "users"
        assert stream.tap_name == "tap-postgres"
        # Test default values
        assert stream.batch_size == 1000
        assert stream.replication_method == "FULL_TABLE"
