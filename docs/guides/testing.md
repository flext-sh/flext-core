# Testing Guide

Comprehensive guide to testing FLEXT-Core applications with pytest, fixtures, and domain-driven testing patterns.

## Overview

FLEXT-Core applications use **pytest** for testing with emphasis on:

- **Domain logic testing**: Business rules and patterns
- **Integration testing**: Service interactions
- **Unit testing**: Individual components with clear separation

**Key principles:**

- Railway-oriented patterns make testing straightforward
- FlextResult enables deterministic testing
- No mocking for happy path - use real implementations
- Mock external services only (APIs, databases)

## Setup

### Project Configuration

**pyproject.toml:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests for isolated components",
    "integration: Integration tests for service interactions",
    "slow: Slow tests requiring external services",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### Test Structure

```
tests/
├── unit/
│   ├── test_result.py
│   ├── test_container.py
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── test_user_service.py
│   ├── test_order_service.py
│   └── test_api_integration.py
├── fixtures/
│   ├── conftest.py
│   └── factories.py
└── __init__.py
```

## Unit Testing

### Testing FlextResult

```python
import pytest
from flext_core import FlextResult

class TestFlextResult:
    """Test FlextResult[T] monad."""

    def test_ok_creates_success_result(self):
        """Test creating success result."""
        result = FlextResult[int].ok(42)
        assert result.is_success
        assert not result.is_failure
        assert result.unwrap() == 42

    def test_fail_creates_failure_result(self):
        """Test creating failure result."""
        result = FlextResult[int].fail("Error message")
        assert result.is_failure
        assert not result.is_success
        assert result.error == "Error message"

    def test_map_transforms_success_value(self):
        """Test map on success result."""
        result = (
            FlextResult[int].ok(5)
            .map(lambda x: x * 2)
            .map(lambda x: x + 1)
        )
        assert result.is_success
        assert result.unwrap() == 11

    def test_map_ignores_failure(self):
        """Test map on failure result."""
        result = (
            FlextResult[int].fail("Error")
            .map(lambda x: x * 2)
        )
        assert result.is_failure
        assert result.error == "Error"

    def test_flat_map_chains_operations(self):
        """Test flat_map for chaining."""
        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        result = (
            FlextResult[int].ok(5)
            .flat_map(double)
            .flat_map(double)
        )
        assert result.unwrap() == 20

    def test_flat_map_stops_on_failure(self):
        """Test flat_map stops at first failure."""
        def double_if_positive(x: int) -> FlextResult[int]:
            if x < 0:
                return FlextResult[int].fail("Negative number")
            return FlextResult[int].ok(x * 2)

        result = (
            FlextResult[int].ok(5)
            .flat_map(double_if_positive)
            .flat_map(lambda x: FlextResult[int].fail("Next error"))
            .flat_map(lambda x: double_if_positive(x))
        )
        assert result.is_failure
        assert result.error == "Next error"

    def test_filter_succeeds_for_true_predicate(self):
        """Test filter with passing predicate."""
        result = (
            FlextResult[int].ok(15)
            .filter(lambda x: x > 10, "Value too small")
        )
        assert result.is_success
        assert result.unwrap() == 15

    def test_filter_fails_for_false_predicate(self):
        """Test filter with failing predicate."""
        result = (
            FlextResult[int].ok(5)
            .filter(lambda x: x > 10, "Value too small")
        )
        assert result.is_failure
        assert result.error == "Value too small"

    def test_unwrap_or_returns_value_on_success(self):
        """Test unwrap_or on success."""
        result = FlextResult[int].ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_returns_default_on_failure(self):
        """Test unwrap_or on failure."""
        result = FlextResult[int].fail("Error")
        assert result.unwrap_or(0) == 0
```

### Testing Domain Logic

```python
import pytest
from flext_core import FlextResult, FlextModels

class User(FlextModels.Entity):
    """User entity for testing."""
    name: str
    email: str
    age: int

def validate_user(user: User) -> FlextResult[User]:
    """Validate user data."""
    if not user.name:
        return FlextResult[User].fail("Name is required")
    if user.age < 18:
        return FlextResult[User].fail("User must be 18+")
    return FlextResult[User].ok(user)

class TestUserValidation:
    """Test user validation logic."""

    def test_valid_user_passes(self):
        """Test valid user passes validation."""
        user = User(id="1", name="Alice", email="alice@example.com", age=25)
        result = validate_user(user)
        assert result.is_success
        assert result.unwrap().name == "Alice"

    def test_empty_name_fails(self):
        """Test empty name fails validation."""
        user = User(id="2", name="", email="bob@example.com", age=30)
        result = validate_user(user)
        assert result.is_failure
        assert "Name is required" in result.error

    def test_underage_user_fails(self):
        """Test underage user fails validation."""
        user = User(id="3", name="Charlie", email="charlie@example.com", age=16)
        result = validate_user(user)
        assert result.is_failure
        assert "18+" in result.error

    @pytest.mark.parametrize("age", [17, 18, 19, 65, 100])
    def test_age_boundary_cases(self, age):
        """Test age validation boundary cases."""
        user = User(id="test", name="Test", email="test@example.com", age=age)
        result = validate_user(user)
        if age >= 18:
            assert result.is_success
        else:
            assert result.is_failure
```

### Testing Services

```python
import pytest
from flext_core import FlextService, FlextResult, FlextContainer

class UserService(FlextService):
    """User management service."""

    def __init__(self):
        super().__init__()
        self.users = {}

    def create_user(self, user_id: str, name: str, email: str) -> FlextResult[dict]:
        """Create new user."""
        if user_id in self.users:
            return FlextResult[dict].fail("User already exists")
        if not name:
            return FlextResult[dict].fail("Name is required")

        user = {"id": user_id, "name": name, "email": email}
        self.users[user_id] = user
        return FlextResult[dict].ok(user)

    def get_user(self, user_id: str) -> FlextResult[dict]:
        """Get user by ID."""
        if user_id not in self.users:
            return FlextResult[dict].fail(f"User {user_id} not found")
        return FlextResult[dict].ok(self.users[user_id])

class TestUserService:
    """Test UserService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return UserService()

    def test_create_user_succeeds(self, service):
        """Test creating user."""
        result = service.create_user("1", "Alice", "alice@example.com")
        assert result.is_success
        assert result.unwrap()["name"] == "Alice"

    def test_create_duplicate_user_fails(self, service):
        """Test creating duplicate user fails."""
        service.create_user("1", "Alice", "alice@example.com")
        result = service.create_user("1", "Bob", "bob@example.com")
        assert result.is_failure
        assert "already exists" in result.error

    def test_get_user_succeeds(self, service):
        """Test getting existing user."""
        service.create_user("1", "Alice", "alice@example.com")
        result = service.get_user("1")
        assert result.is_success
        assert result.unwrap()["name"] == "Alice"

    def test_get_nonexistent_user_fails(self, service):
        """Test getting nonexistent user fails."""
        result = service.get_user("999")
        assert result.is_failure
        assert "not found" in result.error
```

## Integration Testing

### Testing Service Interactions

```python
import pytest
from flext_core import FlextContainer, FlextResult

class EmailService:
    """Email sending service."""

    def send_welcome_email(self, email: str) -> FlextResult[str]:
        """Send welcome email."""
        if not email or "@" not in email:
            return FlextResult[str].fail("Invalid email")
        return FlextResult[str].ok(f"Email sent to {email}")

class UserRegistrationService:
    """User registration with email notification."""

    def __init__(self, user_service, email_service):
        self.user_service = user_service
        self.email_service = email_service

    def register_user(
        self, user_id: str, name: str, email: str
    ) -> FlextResult[dict]:
        """Register user and send welcome email."""
        return (
            self.user_service.create_user(user_id, name, email)
            .flat_map(
                lambda user: self.email_service.send_welcome_email(email)
                .map(lambda _: user)
            )
        )

class TestUserRegistration:
    """Test user registration flow."""

    @pytest.fixture
    def services(self):
        """Setup services."""
        user_service = UserService()
        email_service = EmailService()
        registration_service = UserRegistrationService(
            user_service, email_service
        )
        return {
            "registration": registration_service,
            "user": user_service,
            "email": email_service,
        }

    def test_successful_registration(self, services):
        """Test complete registration flow."""
        result = services["registration"].register_user(
            "1", "Alice", "alice@example.com"
        )
        assert result.is_success
        user = result.unwrap()
        assert user["name"] == "Alice"
        assert services["user"].get_user("1").is_success

    def test_registration_fails_with_invalid_email(self, services):
        """Test registration fails with invalid email."""
        result = services["registration"].register_user(
            "1", "Alice", "invalid-email"
        )
        assert result.is_failure
        assert "Invalid email" in result.error
```

## Fixtures and Factories

### Shared Fixtures

**tests/fixtures/conftest.py:**

```python
import pytest
from flext_core import FlextContainer, FlextLogger

@pytest.fixture
def container():
    """Provide clean container for each test."""
    container = FlextContainer.get_global()
    container.clear()  # Clean state
    return container

@pytest.fixture
def logger():
    """Provide logger for tests."""
    return FlextLogger("test")

@pytest.fixture(autouse=True)
def reset_container():
    """Reset container after each test."""
    yield
    container = FlextContainer.get_global()
    container.clear()
```

### Factory Pattern for Test Data

**tests/fixtures/factories.py:**

```python
from flext_core import FlextModels

class UserFactory:
    """Factory for creating test users."""

    @staticmethod
    def create(
        user_id: str = "test_user_1",
        name: str = "Test User",
        email: str = "test@example.com",
        age: int = 25
    ) -> 'User':
        """Create user instance."""
        return User(
            id=user_id,
            name=name,
            email=email,
            age=age
        )

    @staticmethod
    def create_batch(count: int) -> list:
        """Create multiple users."""
        return [
            UserFactory.create(
                user_id=f"user_{i}",
                name=f"User {i}",
                email=f"user{i}@example.com"
            )
            for i in range(count)
        ]

# Usage in tests
def test_with_factory():
    user = UserFactory.create(name="Alice")
    assert user.name == "Alice"

    users = UserFactory.create_batch(5)
    assert len(users) == 5
```

## Test Markers

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run everything except slow tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_result.py

# Run specific test class
pytest tests/unit/test_result.py::TestFlextResult

# Run specific test
pytest tests/unit/test_result.py::TestFlextResult::test_ok_creates_success_result

# Run with coverage
pytest --cov=src --cov-report=html
```

## Parametrized Testing

```python
import pytest

class TestUserAgeValidation:
    """Test user age validation with parameters."""

    @pytest.mark.parametrize("age,should_pass", [
        (15, False),    # Too young
        (17, False),    # Too young
        (18, True),     # Minimum age
        (25, True),     # Adult
        (65, True),     # Senior
        (100, True),    # Very old
    ])
    def test_age_validation(self, age, should_pass):
        """Test age validation for various ages."""
        result = validate_user_age(age)
        if should_pass:
            assert result.is_success
        else:
            assert result.is_failure

    @pytest.mark.parametrize("email", [
        "valid@example.com",
        "user.name@example.co.uk",
        "test+tag@domain.com",
    ])
    def test_valid_emails(self, email):
        """Test valid email formats."""
        result = validate_email(email)
        assert result.is_success

    @pytest.mark.parametrize("email", [
        "invalid",
        "missing@domain",
        "@nodomain.com",
    ])
    def test_invalid_emails(self, email):
        """Test invalid email formats."""
        result = validate_email(email)
        assert result.is_failure
```

## Mocking and Patching

### Mocking External Services

```python
import pytest
from unittest.mock import Mock, patch

class TestExternalServiceIntegration:
    """Test with mocked external services."""

    @patch('requests.get')
    def test_api_call_success(self, mock_get):
        """Test successful API call."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"id": 1, "name": "Test"}

        service = ExternalApiService()
        result = service.get_user(1)

        assert result.is_success
        assert result.unwrap()["name"] == "Test"

    def test_with_mock_dependency(self):
        """Test with mock dependency injection."""
        mock_email_service = Mock()
        mock_email_service.send.return_value = FlextResult[str].ok("Sent")

        service = UserService(email_service=mock_email_service)
        result = service.register_and_email_user("1", "Alice")

        assert result.is_success
        mock_email_service.send.assert_called_once()
```

## Coverage and Quality

### Running with Coverage Report

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Set minimum coverage threshold
pytest --cov=src --cov-fail-under=80

# Show coverage for specific module
pytest --cov=src.flext_core.result --cov-report=term-missing
```

## Best Practices

### 1. Test One Thing Per Test

```python
# ✅ CORRECT - One assertion per test
def test_create_user_with_valid_data():
    """Test creating user with valid data."""
    result = service.create_user("1", "Alice", "alice@example.com")
    assert result.is_success

def test_create_user_with_missing_name():
    """Test creating user with missing name."""
    result = service.create_user("1", "", "alice@example.com")
    assert result.is_failure

# ❌ WRONG - Multiple things in one test
def test_user_service():
    result1 = service.create_user("1", "Alice", "alice@example.com")
    assert result1.is_success
    result2 = service.get_user("1")
    assert result2.is_success
    result3 = service.delete_user("1")
    assert result3.is_success
```

### 2. Use Descriptive Test Names

```python
# ✅ CORRECT - Clear what's being tested
def test_create_user_succeeds_with_valid_data():
    pass

def test_create_user_fails_with_missing_name():
    pass

# ❌ WRONG - Vague names
def test_user():
    pass

def test_create():
    pass
```

### 3. Use Fixtures for Setup

```python
# ✅ CORRECT - Fixture for common setup
@pytest.fixture
def authenticated_user(service):
    return service.create_user("1", "Alice", "alice@example.com").unwrap()

def test_update_user_profile(authenticated_user):
    result = service.update_profile(authenticated_user["id"], "New Name")
    assert result.is_success

# ❌ WRONG - Repeated setup in each test
def test_update_user_profile_1():
    user = service.create_user("1", "Alice", "alice@example.com").unwrap()
    result = service.update_profile(user["id"], "New Name")
    assert result.is_success
```

### 4. Test Happy and Sad Paths

```python
# ✅ CORRECT - Test both success and failure
def test_withdraw_succeeds_with_sufficient_funds():
    result = account.withdraw(100)
    assert result.is_success

def test_withdraw_fails_with_insufficient_funds():
    result = account.withdraw(1000)
    assert result.is_failure
    assert "Insufficient funds" in result.error

# ❌ WRONG - Only testing happy path
def test_withdraw():
    result = account.withdraw(100)
    assert result.is_success
```

## Summary

FLEXT-Core testing:

- ✅ Use pytest for all testing
- ✅ Test domain logic thoroughly
- ✅ Use FlextResult to test both success and failure paths
- ✅ Use fixtures for shared setup
- ✅ Mock only external services
- ✅ Parametrize tests for multiple scenarios
- ✅ Aim for 75%+ test coverage minimum

This approach makes tests maintainable, deterministic, and focused on behavior.
