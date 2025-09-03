"""Unit tests for 01_railway_result.py example.

Tests the refactored railway pattern implementation using real FlextResult patterns
and FlextUtilities without mocks, following FLEXT testing standards.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from flext_core import FlextConstants, FlextResult, FlextUtilities

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent / "../../examples"))

# Import the example module directly
example_path = Path(__file__).parent / "../../examples/01_railway_result.py"
spec = importlib.util.spec_from_file_location("railway_example", example_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {example_path}")
railway_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(railway_example)

# Import the classes we need
BatchResult = railway_example.BatchResult
RegistrationResult = railway_example.RegistrationResult
User = railway_example.User
UserRegistrationRequest = railway_example.UserRegistrationRequest
UserRegistrationService = railway_example.UserRegistrationService


class TestUserRegistrationRequest:
    """Test UserRegistrationRequest value object."""

    def test_valid_request_creation(self) -> None:
        """Test creating valid UserRegistrationRequest."""
        request = UserRegistrationRequest(
            name="John Doe", email="john@example.com", age=30
        )
        assert request.name == "John Doe"
        assert request.email == "john@example.com"
        assert request.age == 30

    def test_request_validation_success(self) -> None:
        """Test successful validation of valid request."""
        request = UserRegistrationRequest(
            name="Jane Smith", email="jane@example.com", age=25
        )
        result = request.validate_business_rules()
        assert result.is_success
        assert result.value is None

    def test_request_validation_empty_name(self) -> None:
        """Test validation failure with empty name."""
        request = UserRegistrationRequest(name="", email="test@example.com", age=30)
        result = request.validate_business_rules()
        assert result.is_failure
        assert result.error == FlextConstants.Errors.VALIDATION_ERROR

    def test_request_validation_invalid_email(self) -> None:
        """Test validation failure with invalid email."""
        request = UserRegistrationRequest(
            name="Test User", email="invalid-email", age=30
        )
        result = request.validate_business_rules()
        assert result.is_failure
        assert result.error == FlextConstants.Errors.VALIDATION_ERROR


class TestUser:
    """Test User entity."""

    def test_user_creation(self) -> None:
        """Test creating User entity."""
        user = User(
            id="user_123", name="Alice Johnson", email="alice@example.com", age=28
        )
        assert user.id == "user_123"
        assert user.name == "Alice Johnson"
        assert user.email == "alice@example.com"
        assert user.age == 28
        assert user.status == FlextConstants.Status.ACTIVE

    def test_user_validation_success(self) -> None:
        """Test successful user validation."""
        user = User(id="user_456", name="Bob Wilson", email="bob@example.com", age=35)
        result = user.validate_business_rules()
        assert result.is_success

    def test_user_validation_short_name(self) -> None:
        """Test user validation with short name."""
        user = User(id="user_789", name="A", email="a@example.com", age=30)
        result = user.validate_business_rules()
        assert result.is_failure
        assert "at least 2 characters" in result.error

    def test_user_validation_invalid_age(self) -> None:
        """Test user validation with invalid age."""
        user = User(id="user_999", name="Test User", email="test@example.com", age=150)
        result = user.validate_business_rules()
        assert result.is_failure
        assert "between 0 and 120" in result.error


class TestRegistrationResult:
    """Test RegistrationResult model."""

    def test_registration_result_creation(self) -> None:
        """Test creating RegistrationResult."""
        result = RegistrationResult(
            user_id="user_123",
            status="active",
            processing_time_ms=1.5,
            correlation_id="corr_456",
        )
        assert result.user_id == "user_123"
        assert result.status == "active"
        assert result.processing_time_ms == 1.5
        assert result.correlation_id == "corr_456"


class TestBatchResult:
    """Test BatchResult model."""

    def test_batch_result_creation(self) -> None:
        """Test creating BatchResult."""
        results = [
            RegistrationResult(
                user_id="user_1",
                status="active",
                processing_time_ms=1.0,
                correlation_id="corr_1",
            )
        ]
        batch = BatchResult(
            total=2,
            successful=1,
            failed=1,
            success_rate=0.5,
            results=results,
            errors=["Validation failed"],
        )
        assert batch.total == 2
        assert batch.successful == 1
        assert batch.failed == 1
        assert batch.success_rate == 0.5
        assert len(batch.results) == 1
        assert len(batch.errors) == 1


class TestUserRegistrationService:
    """Test UserRegistrationService consolidated service."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = UserRegistrationService()

    def test_service_initialization(self) -> None:
        """Test service initialization."""
        assert self.service is not None
        assert hasattr(self.service, "_utilities")
        assert isinstance(self.service._utilities, FlextUtilities)

    def test_process_registration_success(self) -> None:
        """Test successful registration processing."""
        request = UserRegistrationRequest(
            name="Test User", email="test@example.com", age=30
        )
        result = self.service.process_registration(request)

        assert result.is_success
        assert isinstance(result.value, RegistrationResult)
        assert result.value.user_id.startswith("entity_")
        assert result.value.status == FlextConstants.Status.ACTIVE
        assert result.value.processing_time_ms == 1.0
        assert result.value.correlation_id.startswith("corr_")

    def test_process_registration_validation_failure(self) -> None:
        """Test registration processing with validation failure."""
        request = UserRegistrationRequest(name="", email="invalid", age=-5)
        result = self.service.process_registration(request)

        assert result.is_failure
        assert result.error == FlextConstants.Errors.VALIDATION_ERROR

    def test_process_batch_success(self) -> None:
        """Test successful batch processing."""
        requests = [
            UserRegistrationRequest(name="User 1", email="user1@example.com", age=25),
            UserRegistrationRequest(name="User 2", email="user2@example.com", age=30),
            UserRegistrationRequest(name="User 3", email="user3@example.com", age=35),
        ]
        result = self.service.process_batch(requests)

        assert result.is_success
        assert isinstance(result.value, BatchResult)
        assert result.value.total == 3
        assert result.value.successful == 3
        assert result.value.failed == 0
        assert result.value.success_rate == 1.0
        assert len(result.value.results) == 3
        assert len(result.value.errors) == 0

    def test_process_batch_mixed_results(self) -> None:
        """Test batch processing with mixed success/failure results."""
        requests = [
            UserRegistrationRequest(
                name="Valid User", email="valid@example.com", age=30
            ),
            UserRegistrationRequest(name="", email="invalid", age=-5),  # Invalid
            UserRegistrationRequest(
                name="Another Valid", email="another@example.com", age=25
            ),
        ]
        result = self.service.process_batch(requests)

        assert result.is_success
        assert isinstance(result.value, BatchResult)
        assert result.value.total == 3
        assert result.value.successful == 2
        assert result.value.failed == 1
        assert abs(result.value.success_rate - 2 / 3) < 1e-10
        assert len(result.value.results) == 2
        assert len(result.value.errors) == 1

    def test_process_batch_empty_list(self) -> None:
        """Test batch processing with empty request list."""
        result = self.service.process_batch([])

        assert result.is_failure
        assert result.error == FlextConstants.Errors.VALIDATION_ERROR

    def test_process_json_registration_success(self) -> None:
        """Test successful JSON registration processing."""
        json_data = '{"name": "JSON User", "email": "json@example.com", "age": 32}'
        result = self.service.process_json_registration(json_data)

        assert result.is_success
        assert isinstance(result.value, RegistrationResult)
        assert result.value.user_id.startswith("entity_")
        assert result.value.status == FlextConstants.Status.ACTIVE

    def test_process_json_registration_invalid_json(self) -> None:
        """Test JSON registration processing with invalid JSON."""
        json_data = '{"invalid": json}'
        result = self.service.process_json_registration(json_data)

        assert result.is_failure
        assert "Invalid JSON data" in result.error

    def test_process_json_registration_empty_json(self) -> None:
        """Test JSON registration processing with empty JSON."""
        json_data = "{}"
        result = self.service.process_json_registration(json_data)

        assert result.is_failure
        assert "Invalid JSON data" in result.error

    def test_process_json_registration_malformed_data(self) -> None:
        """Test JSON registration processing with malformed data."""
        json_data = (
            '{"name": "Test", "email": "test@example.com", "age": "not_a_number"}'
        )
        result = self.service.process_json_registration(json_data)

        # Should still succeed because FlextUtilities.Conversions.safe_int handles this
        assert result.is_success
        assert isinstance(result.value, RegistrationResult)

    def test_service_uses_flext_utilities(self) -> None:
        """Test that service properly uses FlextUtilities for ID generation."""
        request = UserRegistrationRequest(
            name="Test User", email="test@example.com", age=30
        )
        result1 = self.service.process_registration(request)
        result2 = self.service.process_registration(request)

        assert result1.is_success
        assert result2.is_success

        # IDs should be different (generated by FlextUtilities)
        assert result1.value.user_id != result2.value.user_id
        assert result1.value.correlation_id != result2.value.correlation_id

    def test_service_error_handling_consistency(self) -> None:
        """Test that service consistently returns FlextResult for all operations."""
        # Test all service methods return FlextResult
        request = UserRegistrationRequest(
            name="Test User", email="test@example.com", age=30
        )

        # Single registration
        result1 = self.service.process_registration(request)
        assert isinstance(result1, FlextResult)

        # Batch processing
        result2 = self.service.process_batch([request])
        assert isinstance(result2, FlextResult)

        # JSON processing
        json_data = '{"name": "Test", "email": "test@example.com", "age": 30}'
        result3 = self.service.process_json_registration(json_data)
        assert isinstance(result3, FlextResult)


class TestRailwayPatternIntegration:
    """Test integration of railway patterns with flext-core components."""

    def test_flext_result_chain_operations(self) -> None:
        """Test chaining FlextResult operations."""
        service = UserRegistrationService()

        # Create a valid request
        request = UserRegistrationRequest(
            name="Chain Test", email="chain@example.com", age=28
        )

        # Process registration
        result = service.process_registration(request)
        assert result.is_success

        # Test that we can chain operations
        registration_result = result.value
        assert registration_result.user_id.startswith("entity_")
        assert registration_result.correlation_id.startswith("corr_")

    def test_flext_utilities_integration(self) -> None:
        """Test integration with FlextUtilities."""
        # Test that FlextUtilities methods are being used
        uuid1 = FlextUtilities.generate_uuid()
        uuid2 = FlextUtilities.generate_uuid()

        assert uuid1 != uuid2
        assert len(uuid1) > 0
        assert len(uuid2) > 0

    def test_flext_constants_usage(self) -> None:
        """Test usage of FlextConstants."""
        # Test that FlextConstants are being used correctly
        assert FlextConstants.Status.ACTIVE == "active"
        assert FlextConstants.Errors.VALIDATION_ERROR == "FLEXT_3001"

    def test_railway_pattern_error_propagation(self) -> None:
        """Test that errors properly propagate through railway pattern."""
        service = UserRegistrationService()

        # Test with invalid request
        invalid_request = UserRegistrationRequest(name="", email="invalid", age=-10)

        result = service.process_registration(invalid_request)
        assert result.is_failure

        # Error should be from validation
        assert result.error == FlextConstants.Errors.VALIDATION_ERROR

    def test_railway_pattern_success_propagation(self) -> None:
        """Test that success properly propagates through railway pattern."""
        service = UserRegistrationService()

        # Test with valid request
        valid_request = UserRegistrationRequest(
            name="Success Test", email="success@example.com", age=30
        )

        result = service.process_registration(valid_request)
        assert result.is_success

        # Should have proper data structure
        registration_result = result.value
        assert isinstance(registration_result, RegistrationResult)
        assert registration_result.user_id.startswith("entity_")
        assert registration_result.correlation_id.startswith("corr_")
        assert registration_result.status == FlextConstants.Status.ACTIVE
