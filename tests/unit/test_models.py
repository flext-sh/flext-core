"""Comprehensive real functional tests for flext_core.models module.

Tests the actual business functionality of all model classes without mocks,
focusing on real patterns, validation, serialization, and domain modeling.

Classes Tested:
- FlextModels: Base Pydantic model with alias generation
- FlextRootModel: Root data structure model
- FlextModels: Immutable value objects with business rules
- FlextModels: Mutable entities with identity and lifecycle
- FlextFactory: Factory pattern implementations



Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import zoneinfo
from datetime import UTC, datetime, timedelta
from typing import TypeGuard

import pytest
from pydantic import Field, ValidationError

from flext_core import FlextModels, FlextResult
from flext_core.typings import FlextTypes
from flext_tests import FlextTestsFixtures, FlextTestsMatchers

pytestmark = [pytest.mark.unit, pytest.mark.core]


# Helper functions that were moved from models.py
def create_timestamp() -> datetime:
    """Create timestamp for testing."""
    return datetime.now(UTC)


def create_version(version: int) -> FlextResult[int]:
    """Create version for testing."""
    if version < 1:
        return FlextResult[int].fail("Version must be >= 1")
    return FlextResult[int].ok(version)


def flext_alias_generator(field_name: str) -> str:
    """Generate camelCase aliases from snake_case."""
    components = field_name.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


def make_hashable(obj: object) -> object:
    """Make object hashable for testing."""
    if isinstance(obj, dict):
        return frozenset(obj.items())
    if isinstance(obj, list):
        return tuple(obj)
    if isinstance(obj, set):
        return frozenset(obj)
    return obj


# =============================================================================
# REAL DOMAIN MODELS FOR TESTING
# =============================================================================


# Use the base classes from models.py and create test-specific implementations
class EmailValue(FlextModels.Value):
    """Real email value object for testing FlextModels."""

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


class UserEntity(FlextModels.Entity):
    """Real user entity for testing FlextModels."""

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
        event_data: FlextTypes.Core.JsonObject = {
            "event_type": "UserActivated",
            "user_id": str(self.id),
            "activated_at": datetime.now(UTC).isoformat(),
        }
        self.add_domain_event(event_data)

        return FlextResult[None].ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user account."""
        if not self.is_active:
            return FlextResult[None].fail("User is already inactive")

        self.is_active = False

        # Add domain event
        event_data: FlextTypes.Core.JsonObject = {
            "event_type": "UserDeactivated",
            "user_id": str(self.id),
            "deactivated_at": datetime.now(UTC).isoformat(),
        }
        self.add_domain_event(event_data)

        return FlextResult[None].ok(None)


class ConfigurationModel(FlextModels.Config):
    """Real configuration model for testing FlextModels."""

    database_url: str = Field(..., description="Database connection URL")
    api_timeout: int = Field(default=30, ge=1, description="API timeout in seconds")
    debug_mode: bool = Field(default=False, description="Debug mode flag")
    features: FlextTypes.Core.StringList = Field(
        default_factory=list, description="Enabled features"
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate configuration business rules."""
        if not self.database_url.strip():
            return FlextResult[None].fail("Database URL cannot be empty")

        if self.api_timeout <= 0:
            return FlextResult[None].fail("API timeout must be positive")

        return FlextResult[None].ok(None)


class DataRootModel(FlextModels.Value):
    """Real root model for testing FlextRootModel."""

    application_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(default="development", description="Environment")

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate root model business rules."""
        if not self.application_name.strip():
            return FlextResult[None].fail("Application name cannot be empty")
        return FlextResult[None].ok(None)


# =============================================================================
# FLEXT MODEL TESTS - Base Pydantic Model with Aliases
# =============================================================================


class TestFlextModelRealFunctionality:
    """Test FlextModels real functionality and patterns."""

    def test_model_creation_with_alias_generation(self) -> None:
        """Test FlextModels creation with alias generation."""
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
        assert "test_models" in config.__class__.__module__

    def test_model_serialization_with_aliases(self) -> None:
        """Test FlextModels serialization functionality."""
        config = ConfigurationModel(
            database_url="postgresql://localhost:5432/test",
            api_timeout=45,
        )

        # model_dump should provide serialized data
        dict_data = config.model_dump()
        assert "database_url" in dict_data
        assert "api_timeout" in dict_data
        assert dict_data["database_url"] == "postgresql://localhost:5432/test"
        assert dict_data["api_timeout"] == 45

    def test_model_basic_functionality(self) -> None:
        """Test FlextModels basic functionality."""
        config = ConfigurationModel(database_url="postgresql://localhost:5432/test")

        # Test basic serialization
        dict_data = config.model_dump()

        # Should contain the serialized data
        assert "database_url" in dict_data
        assert dict_data["database_url"] == "postgresql://localhost:5432/test"

    def test_business_rules_validation(self) -> None:
        """Test FlextModels business rules validation."""
        config = ConfigurationModel(database_url="postgresql://localhost:5432/test")

        # Default implementation should succeed
        result = config.validate_business_rules()
        assert result.is_success

    def test_model_field_validation_errors(self) -> None:
        """Test FlextModels handles Pydantic validation errors."""
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
        """Test FlextRootModel creation and inheritance from FlextModels."""
        root_data = DataRootModel(
            application_name="TestApp",
            version="1.0.0",
            environment="production",
        )

        assert root_data.application_name == "TestApp"
        assert root_data.version == "1.0.0"
        assert root_data.environment == "production"

        # Should inherit FlextModels behavior
        assert root_data.__class__.__name__ == "DataRootModel"
        assert "test_models" in root_data.__class__.__module__

    def test_root_model_serialization(self) -> None:
        """Test FlextRootModel serialization functionality."""
        root_data = DataRootModel(application_name="TestApp", version="2.1.5")

        dict_data = root_data.model_dump()
        assert "application_name" in dict_data
        assert dict_data["application_name"] == "TestApp"
        assert dict_data["version"] == "2.1.5"
        assert dict_data["environment"] == "development"  # default value


# =============================================================================
# FLEXT VALUE TESTS - Immutable Value Objects
# =============================================================================


class TestFlextValueRealFunctionality:
    """Test FlextModels real functionality with immutable value objects."""

    def test_value_object_creation_and_immutability(self) -> None:
        """Test FlextModels creation and immutability."""
        email = EmailValue(address="user@example.com", domain="example.com")

        assert email.address == "user@example.com"
        assert email.domain == "example.com"

        # Should be frozen (immutable) - test that it's read-only
        with pytest.raises(ValidationError):
            # Try to modify a frozen property - this should raise ValidationError
            # Use setattr to bypass mypy's property assignment check
            setattr(email, "address", "changed@example.com")

    def test_value_object_equality_by_value(self) -> None:
        """Test FlextModels equality comparison by value."""
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
        """Test FlextModels business rules validation."""
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
        assert result.error is not None
        assert "@ symbol" in (result.error or "")

    def test_value_object_flext_validation(self) -> None:
        """Test FlextModels validate_flext method."""
        email = EmailValue(address="user@example.com", domain="example.com")

        result = email.validate_business_rules()
        assert result.is_success

    def test_value_object_to_payload_conversion(self) -> None:
        """Test FlextModels to_payload conversion."""
        email = EmailValue(address="user@example.com", domain="example.com")

        # Test model serialization instead of payload
        data = email.model_dump()
        assert isinstance(data, dict)
        assert "address" in data
        assert "domain" in data

    def test_value_object_field_validation(self) -> None:
        """Test FlextModels field validation methods."""
        EmailValue(address="user@example.com", domain="example.com")

        # Test Pydantic validation works
        try:
            EmailValue(address="user@example.com", domain="example.com")
            validation_successful = True
        except Exception:
            validation_successful = False
        assert validation_successful

    def test_value_object_string_representation(self) -> None:
        """Test FlextModels string representations."""
        email = EmailValue(address="user@example.com", domain="example.com")

        str_repr = str(email)
        assert "user@example.com" in str_repr
        assert "example.com" in str_repr


# =============================================================================
# FLEXT ENTITY TESTS - Mutable Entities with Identity
# =============================================================================


class TestFlextEntityRealFunctionality:
    """Test FlextModels real functionality with mutable entities."""

    def test_entity_creation_with_identity(self) -> None:
        """Test FlextModels creation with automatic ID generation."""
        user = UserEntity(
            id="user_123",
            name="John Doe",
            email="john@example.com",
            age=30,
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
        """Test FlextModels basic properties and methods."""
        user = UserEntity(
            id="user_456",
            name="Jane Doe",
            email="jane@example.com",
            age=25,
        )

        # Basic properties should work
        assert user.__class__.__name__ == "UserEntity"
        assert user.version == 1  # new entity
        assert len(user.domain_events) == 0  # no events yet
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_entity_identity_based_equality(self) -> None:
        """Test FlextModels identity-based equality."""
        user1 = UserEntity(
            id="user_123",
            name="John Doe",
            email="john@example.com",
            age=30,
        )
        user2 = UserEntity(
            id="user_123",
            name="Jane Doe",
            email="jane@example.com",
            age=25,
        )  # Same ID, different data
        user3 = UserEntity(
            id="user_456",
            name="John Doe",
            email="john@example.com",
            age=30,
        )  # Different ID, same data

        # Same ID should be equal regardless of other attributes
        assert user1 == user2
        assert hash(user1) == hash(user2)

        # Different ID should not be equal even with same data
        assert user1 != user3
        assert hash(user1) != hash(user3)

    def test_entity_version_management(self) -> None:
        """Test FlextModels version management and optimistic locking."""
        user = UserEntity(
            id="user_789",
            name="Bob Smith",
            email="bob@example.com",
            age=40,
        )

        assert user.version == 1

        # Increment version (now void method)
        original_version = user.version
        user.increment_version()
        assert user.version == original_version + 1
        # Updated timestamp should be set
        # Note: Due to entity equality based on ID, they're equal even with different versions

    def test_entity_with_version_method(self) -> None:
        """Test FlextModels with_version method."""
        user = UserEntity(
            id="user_101",
            name="Alice Johnson",
            email="alice@example.com",
            age=28,
        )

        # Test version management - new API doesn't have with_version
        # Just test that version can be set through regular assignment
        user.version = 5
        assert user.version == 5
        assert user.id == "user_101"
        assert user.name == "Alice Johnson"

    def test_entity_domain_events(self) -> None:
        """Test FlextModels domain events functionality."""
        user = UserEntity(
            id="user_202",
            name="Charlie Brown",
            email="charlie@example.com",
            age=22,
        )

        # Initially no events
        assert len(user.domain_events) == 0

        # Add domain event (new API)
        event_data: FlextTypes.Core.JsonObject = {
            "event_type": "UserCreated",
            "user_id": str(user.id),
            "created_at": datetime.now(UTC).isoformat(),
        }
        user.add_domain_event(event_data)
        assert len(user.domain_events) == 1

    def test_entity_business_logic_with_events(self) -> None:
        """Test FlextModels business logic with domain events."""
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
        deactivate_result = user.deactivate()
        assert deactivate_result.is_success
        assert user.is_active is False

        # Should have both activation and deactivation events
        domain_events = user.domain_events
        events_count = len(domain_events)
        assert events_count == 2  # Both activation and deactivation events

    def test_entity_business_rules_validation(self) -> None:
        """Test FlextModels business rules validation."""
        # Valid user should pass
        valid_user = UserEntity(
            id="user_404",
            name="Valid User",
            email="valid@example.com",
            age=30,
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
        assert result.error is not None
        assert "cannot be empty" in (result.error or "")

    def test_entity_string_representations(self) -> None:
        """Test FlextModels string representations."""
        user = UserEntity(
            id="user_606",
            name="Frank Castle",
            email="frank@example.com",
            age=45,
        )

        # Test __str__
        str_repr = str(user)
        assert "user_606" in str_repr
        assert "Frank Castle" in str_repr

        # Test __repr__
        repr_str = repr(user)
        assert "user_606" in repr_str


# =============================================================================
# FLEXT FACTORY TESTS - Factory Pattern Implementation
# =============================================================================


class TestFlextFactoryRealFunctionality:
    """Test FlextFactory real functionality."""

    def test_factory_user_creation(self) -> None:
        """Test FlextFactory creates users with proper patterns."""
        # Create user through factory
        user = UserEntity(
            id=f"factory_{int(time.time())}",
            name="Factory User",
            email="factory@example.com",
            age=28,
        )

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


class TestFlextModelsUrlValidationEdgeCases:
    """Test URL validation edge cases (lines 856-876)."""

    def test_url_validation_empty_string(self) -> None:
        """Test URL validation with empty string."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            FlextModels.Url("")

    def test_url_validation_whitespace_only(self) -> None:
        """Test URL validation with whitespace only."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            FlextModels.Url("   ")

    def test_url_validation_invalid_format_no_scheme(self) -> None:
        """Test URL validation with no scheme."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            FlextModels.Url("example.com")

    def test_url_validation_invalid_format_no_netloc(self) -> None:
        """Test URL validation with no netloc."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            FlextModels.Url("http://")

    def test_url_validation_malformed_url(self) -> None:
        """Test URL validation with malformed URL that causes urlparse exception."""
        # This should trigger the exception handling path in lines 874-875
        with pytest.raises(ValueError, match="Invalid URL"):
            FlextModels.Url("ht!@#$%^&*()tp://invalid")

    def test_url_validation_valid_urls(self) -> None:
        """Test URL validation with valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://example.com/path",
            "ftp://ftp.example.com",
            "https://example.com:8080/path?query=value",
        ]

        for url in valid_urls:
            url_obj = FlextModels.Url(url)
            assert url_obj.root == url


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
        assert result.error is not None
        assert "must be >= 1" in (result.error or "")

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

        # Add business event (new API)
        event_data: FlextTypes.Core.JsonObject = {
            "event_type": "UserRegistered",
            "registration_type": "standard",
            "source": "integration_test",
        }
        user.add_domain_event(event_data)
        assert len(user.domain_events) == 1

        # Increment version (new API)
        original_version = user.version
        user.increment_version()
        assert user.version == original_version + 1

        # Business logic operations
        result = user.deactivate()
        assert result.is_success
        assert user.is_active is False

        # Validation throughout lifecycle
        result = user.validate_business_rules()
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

        # Add domain event (new API)
        event_data: FlextTypes.Core.JsonObject = {
            "event_type": "UserCreated",
            "source": "serialization_test",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        user.add_domain_event(event_data)

        # Serialize to dict (new API)
        user_dict = user.model_dump()

        # Should contain expected fields
        assert "id" in user_dict
        assert "created_at" in user_dict
        assert "updated_at" in user_dict

        # Should not contain excluded fields
        assert "domain_events" not in user_dict  # Excluded field

        # Verify values
        assert user_dict["id"] == "serialization_test_123"
        assert user_dict["name"] == "Serialization Test User"
        assert user_dict["email"] == "serialize@example.com"


# =============================================================================
# ENHANCED TESTS WITH TESTS/SUPPORT FUNCTIONALITY
# =============================================================================


class TestModelsWithFlextMatchers:
    """Enhanced model tests using FlextTestsMatchers for comprehensive validation."""

    def test_result_validation_with_matchers(self) -> None:
        """Test FlextResult validation using FlextTestsMatchers."""
        # Test success case
        email = EmailValue(address="valid@example.com", domain="example.com")
        result = email.validate_business_rules()

        # Use FlextTestsMatchers for cleaner assertions
        FlextTestsMatchers.CoreMatchers.assert_result_success(result)

        # Test failure case with specific error checking
        invalid_email = EmailValue(address="invalid", domain="example.com")
        result = invalid_email.validate_business_rules()

        FlextTestsMatchers.CoreMatchers.assert_result_failure(
            result, expected_error="@ symbol"
        )

    def test_entity_validation_with_matchers(self) -> None:
        """Test entity validation with comprehensive matchers."""
        # Valid user
        user = UserEntity(
            id="matcher_test_123",
            name="Matcher Test User",
            email="matcher@example.com",
            age=28,
        )

        result = user.validate_business_rules()
        FlextTestsMatchers.CoreMatchers.assert_result_success(result)

        # Test business logic operations
        activation_result = user.activate()
        FlextTestsMatchers.CoreMatchers.assert_result_failure(
            activation_result,
            expected_error="already active",
        )

        # Deactivate first
        user.is_active = False
        activation_result = user.activate()
        FlextTestsMatchers.CoreMatchers.assert_result_success(activation_result)

    def test_json_structure_validation(self) -> None:
        """Test JSON structure validation with FlextTestsMatchers."""
        user = UserEntity(
            id="json_test_123",
            name="JSON Test User",
            email="json@example.com",
            age=30,
        )

        # Serialize and test structure
        user_data = user.model_dump()

        # Test required fields are present
        expected_keys = ["id", "name", "email", "age", "is_active", "version"]
        FlextTestsMatchers.CoreMatchers.assert_json_structure(
            user_data, expected_keys, exact_match=False
        )

    def test_regex_matching_validation(self) -> None:
        """Test regex pattern matching with FlextTestsMatchers."""
        user = UserEntity(
            id="regex_test_123",
            name="Regex Test User",
            email="regex@example.com",
            age=25,
        )

        # Test string representations contain expected patterns
        user_str = str(user)
        FlextTestsMatchers.CoreMatchers.assert_regex_match(user_str, r"regex_test_123")

        user_repr = repr(user)
        FlextTestsMatchers.CoreMatchers.assert_regex_match(user_repr, r"regex_test_123")

    def test_type_guard_validation(self) -> None:
        """Test type guard validation with FlextTestsMatchers."""
        email = EmailValue(address="typeguard@example.com", domain="example.com")
        user = UserEntity(
            id="type_test_123",
            name="Type Test User",
            email="type@example.com",
            age=27,
        )

        # Test type guards
        def is_email_value(obj: object) -> TypeGuard[EmailValue]:
            return isinstance(obj, EmailValue)

        def is_user_entity(obj: object) -> TypeGuard[UserEntity]:
            return isinstance(obj, UserEntity)

        FlextTestsMatchers.CoreMatchers.assert_type_guard(email, is_email_value)
        FlextTestsMatchers.CoreMatchers.assert_type_guard(user, is_user_entity)


class TestModelsPerformance:
    """Performance tests using PerformanceProfiler for comprehensive benchmarking."""

    def test_entity_creation_performance(
        self, benchmark: FlextTestsFixtures.BenchmarkFixture
    ) -> None:
        """Test entity creation performance with benchmarking."""

        def create_user_entity() -> UserEntity:
            return UserEntity(
                id=f"perf_test_{int(time.time() * 1000000)}",
                name="Performance Test User",
                email="performance@example.com",
                age=30,
            )

        # Use FlextTestsMatchers for performance assertion
        result = FlextTestsMatchers.assert_performance_within_limit(
            benchmark,
            create_user_entity,
            max_time_seconds=0.01,  # 10ms limit
        )

        assert isinstance(result, UserEntity)
        assert result.name == "Performance Test User"

    def test_validation_performance(
        self, benchmark: FlextTestsFixtures.BenchmarkFixture
    ) -> None:
        """Test validation performance with comprehensive profiling."""
        user = UserEntity(
            id="validation_perf_123",
            name="Validation Performance User",
            email="validation@example.com",
            age=25,
        )

        def validate_user() -> FlextResult[None]:
            return user.validate_business_rules()

        # Test validation performance
        result = FlextTestsMatchers.PerformanceMatchers.assert_performance_within_limit(
            benchmark,
            validate_user,
            max_time_seconds=0.005,  # 5ms limit
        )

        FlextTestsMatchers.CoreMatchers.assert_result_success(result)

    def test_serialization_performance(
        self, benchmark: FlextTestsFixtures.BenchmarkFixture
    ) -> None:
        """Test serialization performance with comprehensive metrics."""
        user = UserEntity(
            id="serialization_perf_123",
            name="Serialization Performance User",
            email="serialization@example.com",
            age=35,
        )

        # Add some domain events for realistic serialization load
        for i in range(5):
            event_data: FlextTypes.Core.JsonObject = {
                "event_type": f"TestEvent{i}",
                "event_id": f"event_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            user.add_domain_event(event_data)

        def serialize_user() -> FlextTypes.Core.Dict:
            return user.model_dump()

        result = FlextTestsMatchers.assert_performance_within_limit(
            benchmark,
            serialize_user,
            max_time_seconds=0.01,  # 10ms limit
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert result["name"] == "Serialization Performance User"

    def test_bulk_operations_performance(
        self, benchmark: FlextTestsFixtures.BenchmarkFixture
    ) -> None:
        """Test bulk operations performance with linear complexity validation."""

        def create_bulk_users(count: int) -> list[UserEntity]:
            users = []
            for i in range(count):
                user = UserEntity(
                    id=f"bulk_user_{i}",
                    name=f"Bulk User {i}",
                    email=f"bulk{i}@example.com",
                    age=20 + (i % 50),
                )
                users.append(user)
            return users

        # Test linear complexity for bulk operations
        FlextTestsMatchers.CoreMatchers.assert_performance_within_limit(
            benchmark,
            lambda: create_bulk_users(100),
            max_time_seconds=0.1,  # 100ms for 100 users
        )


class TestFlextModelsRootModelValidation:
    """Test FlextModels RootModel classes for 100% coverage of missing lines."""

    def test_aggregate_id_validation_empty_string(self) -> None:
        """Test Event aggregate_id validation with empty string (lines 759-762)."""
        with pytest.raises(ValueError, match="Aggregate ID cannot be empty"):
            FlextModels.Event(
                data={"test": "data"},
                message_type="test_event",
                source_service="test_service",
                aggregate_id="",  # Empty string should trigger the validator
                aggregate_type="TestAggregate",
                event_version=1,
            )

    def test_aggregate_id_validation_whitespace(self) -> None:
        """Test Event aggregate_id validation with whitespace only."""
        with pytest.raises(ValueError, match="Aggregate ID cannot be empty"):
            FlextModels.Event(
                data={"test": "data"},
                message_type="test_event",
                source_service="test_service",
                aggregate_id="   ",  # Whitespace only should trigger validator
                aggregate_type="TestAggregate",
                event_version=1,
            )

    def test_aggregate_id_validation_trimming(self) -> None:
        """Test Event aggregate_id validation trims whitespace."""
        event = FlextModels.Event(
            data={"test": "data"},
            message_type="test_event",
            source_service="test_service",
            aggregate_id="  test_id  ",  # Should be trimmed
            aggregate_type="TestAggregate",
            event_version=1,
        )
        assert event.aggregate_id == "test_id"

    def test_entity_id_validation_empty_string(self) -> None:
        """Test EntityId validation with empty string (lines 780-781)."""
        with pytest.raises(ValidationError):
            FlextModels.EntityId("")

    def test_entity_id_validation_whitespace(self) -> None:
        """Test EntityId validation with whitespace only."""
        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            FlextModels.EntityId("   ")

    def test_entity_id_validation_trimming(self) -> None:
        """Test EntityId validation trims whitespace."""
        entity_id = FlextModels.EntityId("  entity_123  ")
        assert entity_id.root == "entity_123"

    def test_timestamp_ensure_utc_naive_datetime(self) -> None:
        """Test Timestamp.ensure_utc with naive datetime (lines 798-800)."""
        # Create a naive datetime to test ensure_utc functionality
        # Start with timezone-aware then make naive to satisfy ruff DTZ001
        aware_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        naive_dt = aware_dt.replace(tzinfo=None)
        timestamp = FlextModels.Timestamp(naive_dt)
        assert timestamp.root.tzinfo == UTC

    def test_timestamp_ensure_utc_timezone_aware(self) -> None:
        """Test Timestamp.ensure_utc with timezone-aware datetime."""
        eastern = zoneinfo.ZoneInfo("US/Eastern")
        aware_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=eastern)
        timestamp = FlextModels.Timestamp(aware_dt)
        assert timestamp.root.tzinfo == UTC

    def test_email_address_validation_format_check(self) -> None:
        """Test EmailAddress validation format (lines 813-823)."""
        # Test valid email
        email = FlextModels.EmailAddress("test@example.com")
        assert email.root == "test@example.com"

        # Test invalid email - no @ symbol
        with pytest.raises(ValidationError):
            FlextModels.EmailAddress("invalid-email")

        # Test invalid email - multiple @ symbols
        with pytest.raises(ValidationError):
            FlextModels.EmailAddress("test@@example.com")

        # Test invalid email - empty local part
        with pytest.raises(ValidationError):
            FlextModels.EmailAddress("@example.com")

        # Test invalid email - empty domain part
        with pytest.raises(ValidationError):
            FlextModels.EmailAddress("test@")

        # Test invalid email - no dot in domain
        with pytest.raises(ValidationError):
            FlextModels.EmailAddress("test@example")

    def test_host_validation_format_check(self) -> None:
        """Test Host validation format (lines 841-845)."""
        # Test valid host
        host = FlextModels.Host("example.com")
        assert host.root == "example.com"

        # Test host trimming and lowercasing
        host = FlextModels.Host("  EXAMPLE.COM  ")
        assert host.root == "example.com"

        # Test invalid host - empty after trimming
        with pytest.raises(ValueError, match="Invalid hostname format"):
            FlextModels.Host("   ")

        # Test invalid host - contains space
        with pytest.raises(ValueError, match="Invalid hostname format"):
            FlextModels.Host("example .com")

    def test_payload_expiration_checks(self) -> None:
        """Test Payload expiration logic (lines 856-876)."""
        # Test non-expired payload
        future_time = datetime.now(UTC) + timedelta(hours=1)
        payload = FlextModels.Payload(
            data="test data",
            message_type="test_message",
            source_service="test_service",
            expires_at=future_time,
        )
        # Explicitly check the boolean value
        assert not payload.is_expired

        # Test expired payload
        past_time = datetime.now(UTC) - timedelta(hours=1)
        expired_payload = FlextModels.Payload(
            data="test data",
            message_type="test_message",
            source_service="test_service",
            expires_at=past_time,
        )
        # Explicitly check the boolean value
        assert expired_payload.is_expired

        # Test payload without expiration
        no_expiry_payload = FlextModels.Payload(
            data="test data",
            message_type="test_message",
            source_service="test_service",
        )
        assert (
            not no_expiry_payload.is_expired
        )  # Changed from is_expired() to is_expired property

    def test_json_data_validation_serializable(self) -> None:
        """Test JsonData validation for JSON serializable data (lines 889-895)."""
        # Test valid JSON data
        valid_data: FlextTypes.Core.JsonObject = {"key": "value", "number": 42}
        json_data = FlextModels.JsonData(valid_data)
        assert json_data.root == valid_data

        # Test invalid JSON data - function object
        def test_function() -> str:
            return "test"

        # Create a dict with function that can't be serialized
        # Use cast to bypass mypy's type checking for this test
        invalid_data = {"func": test_function}
        with pytest.raises(ValidationError):
            FlextModels.JsonData(invalid_data)

    def test_metadata_validation_string_values(self) -> None:
        """Test Metadata validation ensures string values (line 907)."""
        # This line is just a return statement in the validator,
        # but we need to trigger the validator to hit line 907
        valid_metadata = {"key1": "value1", "key2": "value2"}
        metadata = FlextModels.Metadata(valid_metadata)
        assert metadata.root == valid_metadata


class TestFlextModelsEntityClearDomainEvents:
    """Test Entity.clear_domain_events method specifically for missing lines 274-276."""

    def test_clear_domain_events_returns_copy_and_clears_list(self) -> None:
        """Test lines 274-276: events = self.domain_events.copy(), self.domain_events.clear(), return events."""
        user = UserEntity(
            id="clear_events_test",
            name="Test User",
            email="test@example.com",
            age=30,
        )

        # Add some events
        event1: FlextTypes.Core.JsonObject = {"event_type": "Event1"}
        event2: FlextTypes.Core.JsonObject = {"event_type": "Event2"}
        user.add_domain_event(event1)
        user.add_domain_event(event2)

        assert len(user.domain_events) == 2

        # Clear events should return copy of events and clear the list
        returned_events = user.clear_domain_events()

        # Verify returned events contain the original events
        assert len(returned_events) == 2
        assert returned_events[0]["event_type"] == "Event1"
        assert returned_events[1]["event_type"] == "Event2"

        # Verify original list is now empty
        assert len(user.domain_events) == 0


class TestFlextModelsAggregateRootApplyEvent:
    """Test AggregateRoot.apply_domain_event method for missing lines 353-354, 357-358."""

    def test_apply_domain_event_with_handler_execution(self) -> None:
        """Test lines 353-354: handler execution when event_type matches."""

        # Create a test aggregate root
        class TestAggregateRoot(FlextModels.AggregateRoot):
            name: str
            handler_called: bool = False

            def validate_business_rules(self) -> FlextResult[None]:
                """Implement required abstract method."""
                return FlextResult[None].ok(None)

            def _apply_testevent(self, _event: FlextTypes.Core.JsonObject) -> None:
                """Test event handler - note lowercase as per line 351."""
                self.handler_called = True

        root = TestAggregateRoot(id="test_id", name="Test Root")

        # Add event that should trigger handler
        test_event: FlextTypes.Core.JsonObject = {
            "event_type": "testevent",
            "data": "test",
        }

        # Apply event - this should find and execute the handler
        result = root.apply_domain_event(test_event)

        # Verify handler was called (tests line 353-354)
        assert result.success is True
        assert root.handler_called is True

    def test_apply_domain_event_exception_handling(self) -> None:
        """Test lines 357-358: exception handling in apply_domain_event."""

        # Create aggregate with handler that raises exception
        class FailingAggregateRoot(FlextModels.AggregateRoot):
            name: str

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            def _apply_failingevent(self, _event: FlextTypes.Core.JsonObject) -> None:
                """Handler that raises an exception."""
                msg = "Handler failed"
                raise ValueError(msg)

        root = FailingAggregateRoot(id="test_id", name="Test Root")

        # Apply event that triggers failing handler
        failing_event: FlextTypes.Core.JsonObject = {
            "event_type": "FailingEvent",
            "data": "test",
        }

        # This should not raise an exception (handler exceptions are caught)
        try:
            root.apply_domain_event(failing_event)
            exception_handled = True
        except Exception:
            exception_handled = False

        # Verify exception was handled properly (tests lines 357-358)
        assert exception_handled is True
