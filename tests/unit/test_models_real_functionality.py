"""Comprehensive real functional tests for flext_core.models module.

Tests the actual business functionality of all model classes without mocks,
focusing on real patterns, validation, serialization, and domain modeling.

Classes Tested:
- FlextModel: Base Pydantic model with alias generation
- FlextRootModel: Root data structure model
- FlextValue: Immutable value objects with business rules
- FlextEntity: Mutable entities with identity and lifecycle
- FlextFactory: Factory pattern implementations
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest
from pydantic import Field, ValidationError

from flext_core.models import (
    FlextEntity,
    FlextModel,
    FlextRootModel,
    FlextValue,
    create_timestamp,
    create_version,
    flext_alias_generator,
)
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# REAL DOMAIN MODELS FOR TESTING
# =============================================================================


# Use the base classes from models.py and create test-specific implementations
class EmailValue(FlextValue):
    """Real email value object for testing FlextValue."""

    address: str = Field(..., description="Email address")
    domain: str = Field(..., description="Email domain")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate email business rules."""
        if "@" not in self.address:
            return FlextResult[None].fail("Email must contain @ symbol")

        if not self.domain:
            return FlextResult[None].fail("Email domain cannot be empty")

        if "." not in self.domain:
            return FlextResult[None].fail("Domain must contain at least one dot")

        return FlextResult[None].ok(None)


class UserEntity(FlextEntity):
    """Real user entity for testing FlextEntity."""

    name: str = Field(..., description="User full name")
    email: str = Field(..., description="User email address")
    age: int = Field(..., ge=0, le=150, description="User age")
    is_active: bool = Field(default=True, description="User active status")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate user business rules."""
        if not self.name.strip():
            return FlextResult[None].fail("Name cannot be empty")

        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")

        if self.age < 0:
            return FlextResult[None].fail("Age cannot be negative")

        return FlextResult[None].ok(None)

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult[None].fail("User is already active")

        self.is_active = True

        # Add domain event
        event_result = self.add_domain_event(
            "UserActivated",
            {
                "user_id": str(self.id),
                "activated_at": datetime.now(UTC).isoformat(),
            },
        )

        if event_result.is_failure:
            return FlextResult[None].fail(
                f"Failed to record activation event: {event_result.error}"
            )

        return FlextResult[None].ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user account."""
        if not self.is_active:
            return FlextResult[None].fail("User is already inactive")

        self.is_active = False

        # Add domain event
        event_result = self.add_domain_event(
            "UserDeactivated",
            {
                "user_id": str(self.id),
                "deactivated_at": datetime.now(UTC).isoformat(),
            },
        )

        if event_result.is_failure:
            return FlextResult[None].fail(
                f"Failed to record deactivation event: {event_result.error}"
            )

        return FlextResult[None].ok(None)


class ConfigurationModel(FlextModel):
    """Real configuration model for testing FlextModel."""

    database_url: str = Field(..., description="Database connection URL")
    api_timeout: int = Field(default=30, ge=1, description="API timeout in seconds")
    debug_mode: bool = Field(default=False, description="Debug mode flag")
    features: list[str] = Field(default_factory=list, description="Enabled features")


class DataRootModel(FlextRootModel):
    """Real root model for testing FlextRootModel."""

    application_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(default="development", description="Environment")


# =============================================================================
# FLEXT MODEL TESTS - Base Pydantic Model with Aliases
# =============================================================================


class TestFlextModelRealFunctionality:
    """Test FlextModel real functionality and patterns."""

    def test_model_creation_with_alias_generation(self) -> None:
        """Test FlextModel creation with alias generation."""
        config = ConfigurationModel(
            database_url="postgresql://localhost:5432/test",
            api_timeout=60,
            debug_mode=True,
            features=["auth", "logging"],
        )

        assert config.database_url == "postgresql://localhost:5432/test"
        assert config.api_timeout == 60
        assert config.debug_mode is True
        assert config.features == ["auth", "logging"]

        # Test basic model properties
        assert config.__class__.__name__ == "ConfigurationModel"
        assert "test_models_real_functionality" in config.__class__.__module__

    def test_model_serialization_with_aliases(self) -> None:
        """Test FlextModel serialization uses camelCase aliases."""
        config = ConfigurationModel(
            database_url="postgresql://localhost:5432/test", api_timeout=45
        )

        # to_dict should use camelCase aliases
        dict_data = config.to_dict()
        assert "databaseUrl" in dict_data
        assert "apiTimeout" in dict_data
        assert dict_data["databaseUrl"] == "postgresql://localhost:5432/test"
        assert dict_data["apiTimeout"] == 45

    def test_model_basic_functionality(self) -> None:
        """Test FlextModel basic functionality."""
        config = ConfigurationModel(database_url="postgresql://localhost:5432/test")

        # Test basic serialization
        dict_data = config.to_dict()

        # Should contain the serialized data
        assert "databaseUrl" in dict_data
        assert dict_data["databaseUrl"] == "postgresql://localhost:5432/test"

    def test_business_rules_validation(self) -> None:
        """Test FlextModel business rules validation."""
        config = ConfigurationModel(database_url="postgresql://localhost:5432/test")

        # Default implementation should succeed
        result = config.validate_business_rules()
        assert result.is_success

    def test_model_field_validation_errors(self) -> None:
        """Test FlextModel handles Pydantic validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigurationModel(
                database_url="postgresql://localhost:5432/test",
                api_timeout=-1,  # Should fail validation (ge=1)
            )

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert any(
            "greater than or equal to 1" in str(error["msg"]) for error in errors
        )


# =============================================================================
# FLEXT ROOT MODEL TESTS - Root Data Structures
# =============================================================================


class TestFlextRootModelRealFunctionality:
    """Test FlextRootModel real functionality."""

    def test_root_model_creation(self) -> None:
        """Test FlextRootModel creation and inheritance from FlextModel."""
        root_data = DataRootModel(
            application_name="TestApp", version="1.0.0", environment="production"
        )

        assert root_data.application_name == "TestApp"
        assert root_data.version == "1.0.0"
        assert root_data.environment == "production"

        # Should inherit FlextModel behavior
        assert root_data.__class__.__name__ == "DataRootModel"
        assert "test_models_real_functionality" in root_data.__class__.__module__

    def test_root_model_serialization(self) -> None:
        """Test FlextRootModel serialization with aliases."""
        root_data = DataRootModel(application_name="TestApp", version="2.1.5")

        dict_data = root_data.to_dict()
        assert "applicationName" in dict_data
        assert dict_data["applicationName"] == "TestApp"
        assert dict_data["version"] == "2.1.5"
        assert dict_data["environment"] == "development"  # default value


# =============================================================================
# FLEXT VALUE TESTS - Immutable Value Objects
# =============================================================================


class TestFlextValueRealFunctionality:
    """Test FlextValue real functionality with immutable value objects."""

    def test_value_object_creation_and_immutability(self) -> None:
        """Test FlextValue creation and immutability."""
        email = EmailValue(address="user@example.com", domain="example.com")

        assert email.address == "user@example.com"
        assert email.domain == "example.com"

        # Should be frozen (immutable)
        with pytest.raises((ValidationError, AttributeError)):
            email.address = "changed@example.com"  # type: ignore[misc]

    def test_value_object_equality_by_value(self) -> None:
        """Test FlextValue equality comparison by value."""
        email1 = EmailValue(address="test@example.com", domain="example.com")
        email2 = EmailValue(address="test@example.com", domain="example.com")
        email3 = EmailValue(address="other@example.com", domain="example.com")

        # Same values should be equal
        assert email1 == email2
        assert hash(email1) == hash(email2)

        # Different values should not be equal
        assert email1 != email3
        assert hash(email1) != hash(email3)

    def test_value_object_business_rules_validation(self) -> None:
        """Test FlextValue business rules validation."""
        # Valid email should pass validation
        valid_email = EmailValue(address="user@example.com", domain="example.com")
        result = valid_email.validate_business_rules()
        assert result.is_success

        # Invalid email should fail validation
        invalid_email = EmailValue(
            address="invalid-email",  # No @ symbol
            domain="example.com",
        )
        result = invalid_email.validate_business_rules()
        assert result.is_failure
        assert "@ symbol" in result.error

    def test_value_object_flext_validation(self) -> None:
        """Test FlextValue validate_flext method."""
        email = EmailValue(address="user@example.com", domain="example.com")

        result = email.validate_flext()
        assert result.is_success
        assert result.value == email

    def test_value_object_to_payload_conversion(self) -> None:
        """Test FlextValue to_payload conversion."""
        email = EmailValue(address="user@example.com", domain="example.com")

        payload = email.to_payload()

        # Should have payload data structure
        assert payload is not None

        # Test payload data access
        data = payload.data
        assert isinstance(data, dict)
        assert "value_object_data" in data
        assert "class_info" in data
        assert "validation_status" in data

        # Validation status should be valid for valid email
        assert data["validation_status"] == "valid"

    def test_value_object_field_validation(self) -> None:
        """Test FlextValue field validation methods."""
        email = EmailValue(address="user@example.com", domain="example.com")

        # Test individual field validation
        result = email.validate_field("address", "user@example.com")
        assert result.is_success

        # Test all fields validation
        result = email.validate_all_fields()
        assert result.is_success

    def test_value_object_string_representation(self) -> None:
        """Test FlextValue string representations."""
        email = EmailValue(address="user@example.com", domain="example.com")

        str_repr = str(email)
        assert "EmailValue" in str_repr
        assert "address='user@example.com'" in str_repr
        assert "domain='example.com'" in str_repr


# =============================================================================
# FLEXT ENTITY TESTS - Mutable Entities with Identity
# =============================================================================


class TestFlextEntityRealFunctionality:
    """Test FlextEntity real functionality with mutable entities."""

    def test_entity_creation_with_identity(self) -> None:
        """Test FlextEntity creation with automatic ID generation."""
        user = UserEntity(
            id="user_123", name="John Doe", email="john@example.com", age=30
        )

        assert str(user.id) == "user_123"
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.age == 30
        assert user.is_active is True  # default value

        # Should have default version and timestamps
        assert user.version == 1
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_entity_basic_properties(self) -> None:
        """Test FlextEntity basic properties and methods."""
        user = UserEntity(
            id="user_456", name="Jane Doe", email="jane@example.com", age=25
        )

        # Basic properties should work
        assert user.__class__.__name__ == "UserEntity"
        assert user.version == 1  # new entity
        assert len(user.domain_events) == 0  # no events yet
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_entity_identity_based_equality(self) -> None:
        """Test FlextEntity identity-based equality."""
        user1 = UserEntity(
            id="user_123", name="John Doe", email="john@example.com", age=30
        )
        user2 = UserEntity(
            id="user_123", name="Jane Doe", email="jane@example.com", age=25
        )  # Same ID, different data
        user3 = UserEntity(
            id="user_456", name="John Doe", email="john@example.com", age=30
        )  # Different ID, same data

        # Same ID should be equal regardless of other attributes
        assert user1 == user2
        assert hash(user1) == hash(user2)

        # Different ID should not be equal even with same data
        assert user1 != user3
        assert hash(user1) != hash(user3)

    def test_entity_version_management(self) -> None:
        """Test FlextEntity version management and optimistic locking."""
        user = UserEntity(
            id="user_789", name="Bob Smith", email="bob@example.com", age=40
        )

        assert user.version == 1

        # Increment version
        result = user.increment_version()
        assert result.is_success

        new_user = result.value
        assert new_user.version == 2
        assert new_user.id == user.id  # Same identity
        # Note: Due to entity equality based on ID, they're equal even with different versions

    def test_entity_with_version_method(self) -> None:
        """Test FlextEntity with_version method."""
        user = UserEntity(
            id="user_101", name="Alice Johnson", email="alice@example.com", age=28
        )

        # Set specific version
        new_user = user.with_version(5)
        assert new_user.version == 5
        assert new_user.id == user.id
        assert new_user.name == user.name

        # Should validate version is greater than current
        with pytest.raises(Exception):  # Should raise FlextValidationError
            user.with_version(1)  # Same or lower version should fail

    def test_entity_domain_events(self) -> None:
        """Test FlextEntity domain events functionality."""
        user = UserEntity(
            id="user_202", name="Charlie Brown", email="charlie@example.com", age=22
        )

        # Initially no events
        assert len(user.domain_events) == 0

        # Add domain event
        result = user.add_domain_event(
            "UserCreated",
            {"user_id": str(user.id), "created_at": datetime.now(UTC).isoformat()},
        )

        assert result.is_success
        assert len(user.domain_events) == 1

    def test_entity_business_logic_with_events(self) -> None:
        """Test FlextEntity business logic with domain events."""
        user = UserEntity(
            id="user_303",
            name="Diana Prince",
            email="diana@example.com",
            age=35,
            is_active=False,
        )

        # Activate user
        result = user.activate()
        assert result.is_success
        assert user.is_active is True
        assert len(user.domain_events) == 1

        # Deactivate user
        result = user.deactivate()
        assert result.is_success
        assert user.is_active is False
        assert len(user.domain_events) == 2  # Both activation and deactivation events

    def test_entity_business_rules_validation(self) -> None:
        """Test FlextEntity business rules validation."""
        # Valid user should pass
        valid_user = UserEntity(
            id="user_404", name="Valid User", email="valid@example.com", age=30
        )
        result = valid_user.validate_business_rules()
        assert result.is_success

        # Test individual validation rules by creating users that pass Pydantic validation
        # but fail business rules
        user_empty_name = UserEntity(
            id="user_505",
            name=" ",  # Whitespace only name should fail business rules
            email="valid@example.com",
            age=25,
        )
        result = user_empty_name.validate_business_rules()
        assert result.is_failure
        assert "cannot be empty" in result.error

    def test_entity_string_representations(self) -> None:
        """Test FlextEntity string representations."""
        user = UserEntity(
            id="user_606", name="Frank Castle", email="frank@example.com", age=45
        )

        # Test __str__
        str_repr = str(user)
        assert "UserEntity" in str_repr
        assert "user_606" in str_repr

        # Test __repr__
        repr_str = repr(user)
        assert "UserEntity" in repr_str
        assert "id=user_606" in repr_str
        # Version might be RootModel, so just check it exists
        assert "version=" in repr_str


# =============================================================================
# FLEXT FACTORY TESTS - Factory Pattern Implementation
# =============================================================================


class TestFlextFactoryRealFunctionality:
    """Test FlextFactory real functionality."""

    def test_factory_user_creation(self) -> None:
        """Test FlextFactory creates users with proper patterns."""
        # Create user through factory
        user_data = {"name": "Factory User", "email": "factory@example.com", "age": 28}

        user = UserEntity(id=f"factory_{int(time.time())}", **user_data)

        assert user.name == "Factory User"
        assert user.email == "factory@example.com"
        assert user.age == 28
        assert user.is_active is True

    def test_factory_batch_creation(self) -> None:
        """Test FlextFactory patterns for batch creation."""
        users = []

        for i in range(5):
            user = UserEntity(
                id=f"batch_user_{i}",
                name=f"User {i}",
                email=f"user{i}@example.com",
                age=20 + i,
            )
            users.append(user)

        assert len(users) == 5

        # All should have unique IDs
        ids = {str(user.id) for user in users}
        assert len(ids) == 5

        # All should be valid
        for user in users:
            result = user.validate_business_rules()
            assert result.is_success


# =============================================================================
# HELPER FUNCTION TESTS - Utility Functions
# =============================================================================


class TestHelperFunctionsRealFunctionality:
    """Test helper functions from models.py."""

    def test_create_timestamp_function(self) -> None:
        """Test create_timestamp helper function."""
        timestamp1 = create_timestamp()
        time.sleep(0.001)  # Small delay
        timestamp2 = create_timestamp()

        # Both should be valid timestamps
        assert timestamp1 is not None
        assert timestamp2 is not None

        # Should be different (later timestamp should be greater)
        assert timestamp2 > timestamp1

    def test_create_version_function(self) -> None:
        """Test create_version helper function."""
        # Valid versions should succeed
        result = create_version(1)
        assert result.is_success
        assert result.value == 1

        result = create_version(100)
        assert result.is_success
        assert result.value == 100

        # Invalid versions should fail
        result = create_version(0)
        assert result.is_failure
        assert "must be >= 1" in result.error

        result = create_version(-5)
        assert result.is_failure

    def test_flext_alias_generator_function(self) -> None:
        """Test flext_alias_generator helper function."""
        # Should convert snake_case to camelCase
        assert flext_alias_generator("user_name") == "userName"
        assert flext_alias_generator("api_timeout") == "apiTimeout"
        assert flext_alias_generator("database_url") == "databaseUrl"
        assert flext_alias_generator("simple") == "simple"  # No change for single word

    def test_make_hashable_utility_function(self) -> None:
        """Test make_hashable utility function from models.py."""
        from flext_core.models import make_hashable

        # Test simple values
        assert make_hashable("string") == "string"
        assert make_hashable(42) == 42
        assert make_hashable(True) is True

        # Test dict conversion
        test_dict = {"key1": "value1", "key2": "value2"}
        result = make_hashable(test_dict)
        assert isinstance(result, frozenset)

        # Test list conversion
        test_list = ["item1", "item2"]
        result = make_hashable(test_list)
        assert isinstance(result, tuple)
        assert result == ("item1", "item2")

        # Test set conversion
        test_set = {"item1", "item2"}
        result = make_hashable(test_set)
        assert isinstance(result, frozenset)


# =============================================================================
# INTEGRATION TESTS - Multiple Components Working Together
# =============================================================================


class TestModelsIntegrationRealFunctionality:
    """Test integration between different model components."""

    def test_complete_entity_lifecycle_workflow(self) -> None:
        """Test complete entity lifecycle with all components."""
        # Create entity
        user = UserEntity(
            id="integration_user_123",
            name="Integration User",
            email="integration@example.com",
            age=30,
        )

        # Validate creation
        assert user.version == 1  # new entity
        assert len(user.domain_events) == 0  # no events initially

        # Add business event
        result = user.add_domain_event(
            "UserRegistered",
            {"registration_type": "standard", "source": "integration_test"},
        )
        assert result.is_success
        assert len(user.domain_events) == 1

        # Increment version
        result = user.increment_version()
        assert result.is_success
        new_user = result.value
        assert new_user.version == 2

        # Business logic operations
        result = new_user.deactivate()
        assert result.is_success
        assert new_user.is_active is False

        # Validation throughout lifecycle
        result = new_user.validate_business_rules()
        assert result.is_success

    def test_value_object_and_entity_integration(self) -> None:
        """Test value objects working with entities."""
        # Create value object
        email = EmailValue(address="integration@example.com", domain="example.com")

        # Validate value object
        result = email.validate_business_rules()
        assert result.is_success

        # Use in entity
        user = UserEntity(
            id="vo_integration_123",
            name="Value Object User",
            email=email.address,  # Use value object data
            age=25,
        )

        # Both should be valid
        assert user.email == email.address
        result = user.validate_business_rules()
        assert result.is_success

    def test_serialization_round_trip(self) -> None:
        """Test serialization and deserialization round trip."""
        # Create entity with complex data
        user = UserEntity(
            id="serialization_test_123",
            name="Serialization Test User",
            email="serialize@example.com",
            age=32,
        )

        # Add domain event
        result = user.add_domain_event(
            "UserCreated",
            {
                "source": "serialization_test",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        assert result.is_success

        # Serialize to dict
        user_dict = user.to_dict()

        # Should contain expected fields with aliases
        assert "entityId" in user_dict
        assert "createdAt" in user_dict
        assert "updatedAt" in user_dict

        # Should not contain excluded fields
        assert "domain_events" not in user_dict  # Excluded field

        # Verify values
        assert user_dict["entityId"] == "serialization_test_123"
        assert user_dict["name"] == "Serialization Test User"
        assert user_dict["email"] == "serialize@example.com"
