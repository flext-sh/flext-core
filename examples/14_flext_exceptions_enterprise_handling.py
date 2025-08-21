#!/usr/bin/env python3
"""Enterprise exception handling with FlextExceptions.

Demonstrates exception hierarchy, observability, metrics tracking,
and structured context management for robust error handling.
    - Exception factory methods for consistent creation patterns
    - Observability integration with automatic metrics tracking
    - Structured context capture for debugging and error resolution
    - Enterprise exception patterns for different domains
    - Exception serialization and transport capabilities
    - Error recovery patterns and best practices

Key Components:
    - FlextError: Base exception with observability and context management
    - FlextValidationError: Field validation failures with detailed context
    - FlextTypeError: Type mismatch errors with expected/actual type info
    - FlextOperationError: Operation failures with stage tracking
    - Domain-specific exceptions: Configuration, connection, authentication, etc.
    - FlextExceptions: Unified factory interface for exception creation
    - Global metrics tracking for operational insights

This example shows real-world enterprise exception handling scenarios
demonstrating the power and flexibility of the FlextExceptions system.
"""

import contextlib
import operator
import os
import time
import traceback
from collections.abc import Mapping
from typing import cast

from flext_core import (
    FlextAlreadyExistsError,
    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextCriticalError,
    FlextError,
    FlextExceptions,
    FlextNotFoundError,
    FlextOperationError,
    FlextPermissionError,
    FlextProcessingError,
    FlextResult,
    FlextTimeoutError,
    FlextTypeError,
    FlextValidationError,
    clear_exception_metrics,
    get_exception_metrics,
)

# =============================================================================
# EXCEPTION CONSTANTS - Validation and error handling constraints
# =============================================================================

# Age validation constants
MIN_USER_AGE = 18  # Minimum legal age for user registration
MAX_USER_AGE = 120  # Maximum reasonable age for validation

# Retry attempt constants
MAX_RETRY_ATTEMPTS = 2  # Maximum retry attempts before final failure

# =============================================================================
# DOMAIN MODELS - Business entities for examples
# =============================================================================


class User:
    """User domain model for examples."""

    def __init__(
        self,
        user_id: str,
        name: str,
        email: str,
        age: int | None = None,
    ) -> None:
        """Initialize user with required and optional attributes.

        Args:
            user_id: Unique identifier for the user
            name: User's display name
            email: User's email address
            age: Optional user age

        """
        self.user_id = user_id
        self.name = name
        self.email = email
        self.age = age
        self.is_active = True

    def __repr__(self) -> str:
        """Return string representation of user object."""
        return f"User(id={self.user_id}, name='{self.name}', email='{self.email}')"


class DatabaseConnection:
    """Mock database connection for examples."""

    def __init__(self, host: str, port: int, database: str) -> None:
        """Initialize database connection parameters.

        Args:
            host: Database host address
            port: Database port number
            database: Database name

        """
        self.host = host
        self.port = port
        self.database = database
        self.connected = False

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        if self.host == "unreachable-host":
            error_msg = "Database connection failed"
            raise FlextConnectionError(
                error_msg,
                service=f"database://{self.host}:{self.port}/{self.database}",
                context={
                    "host": self.host,
                    "port": self.port,
                    "database": self.database,
                    "timeout_seconds": 30,
                },
            )

        self.connected = True
        return FlextResult[None].ok(None)

    def authenticate(self, username: str, password: str) -> FlextResult[None]:
        """Authenticate with database."""
        if not self.connected:
            error_msg = "Cannot authenticate without connection"
            raise FlextOperationError(
                error_msg,
                operation="database_authentication",
                stage="pre_authentication_check",
            )

        # Demo credentials - in production use secure authentication
        demo_username = "admin"
        demo_password = "secret"  # noqa: S105
        if username != demo_username or password != demo_password:
            msg = "Invalid database credentials"
            raise FlextAuthenticationError(
                msg,
                service=f"database://{self.database}",
                context={"endpoint": f"auth://{username}"},
            )

        return FlextResult[None].ok(None)


# =============================================================================
# APPLICATION SERVICES - Business logic with exception handling
# =============================================================================


class UserValidationService:
    """User validation service demonstrating validation exceptions."""

    @staticmethod
    def _ensure_str_or_type_error(value: object, field: str) -> str:
        """Ensure a value is a string or raise a TypeError."""
        if not isinstance(value, str):
            msg = f"{field} is not a string after validation"
            raise TypeError(msg)
        return value

    @staticmethod
    def _validate_name_required(data: Mapping[str, object]) -> None:
        """Validate that name field is provided."""
        if "name" not in data:
            msg = "Name is required"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "name",
                    "value": None,
                    "rules": ["required", "non_empty"],
                },
            )

    @staticmethod
    def _validate_name_format(name: object) -> None:
        """Validate name format and type."""
        if not isinstance(name, str) or len(name.strip()) == 0:
            msg = "Name must be a non-empty string"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "name",
                    "value": name,
                    "rules": ["string_type", "non_empty"],
                },
            )

    @staticmethod
    def _validate_email_required(data: Mapping[str, object]) -> None:
        """Validate that email field is provided."""
        if "email" not in data:
            msg = "Email is required"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "email",
                    "value": None,
                    "rules": ["required", "email_format"],
                },
            )

    @staticmethod
    def _validate_email_format(email: object) -> None:
        """Validate email format and type."""
        if not isinstance(email, str) or "@" not in email:
            msg = "Email must be a valid email address"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "email",
                    "value": email,
                    "rules": ["string_type", "email_format"],
                },
            )

    @staticmethod
    def _validate_age_type(age: object) -> None:
        """Validate age type."""
        if not isinstance(age, int):
            msg = "Age must be an integer"
            raise FlextTypeError(
                msg,
                expected_type="int",
                actual_type=type(age).__name__,
            )

    @staticmethod
    def _validate_age_range(age: int) -> None:
        """Validate age range."""
        if age < MIN_USER_AGE or age > MAX_USER_AGE:
            msg: str = f"Age must be between {MIN_USER_AGE} and {MAX_USER_AGE}"
            raise FlextValidationError(
                msg,
                validation_details={
                    "field": "age",
                    "value": age,
                    "rules": ["age_range"],
                },
            )

    def validate_user_data(self, data: Mapping[str, object]) -> FlextResult[User]:
        """Validate user data and create a User instance.

        Raises validation and type errors for invalid input.
        """
        try:
            # Validate required fields
            self._validate_name_required(data)

            name = data["name"]
            self._validate_name_format(name)
            # After validation, ensure str type
            name_str = self._ensure_str_or_type_error(name, "Name")

            # Validate email
            self._validate_email_required(data)

            email = data["email"]
            self._validate_email_format(email)
            email_str = self._ensure_str_or_type_error(email, "Email")

            # Validate age if provided
            age_obj = data.get("age")
            age: int | None = None
            if age_obj is not None:
                self._validate_age_type(age_obj)
                # After validation, we can safely cast to int
                age = cast("int", age_obj)
                self._validate_age_range(age)

            # Create user
            user_id_obj = data.get("user_id", f"user_{int(time.time())}")
            user_id = self._ensure_str_or_type_error(user_id_obj, "User ID")

            user = User(
                user_id,
                name_str.strip(),
                email_str.lower().strip(),
                age,
            )

            return FlextResult[User].ok(user)

        except (FlextValidationError, FlextTypeError) as e:
            return FlextResult[User].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            # Wrap unexpected exceptions
            msg = "Unexpected error during user validation"
            raise FlextProcessingError(
                msg,
                operation="user_validation",
                context={
                    "stage": "data_processing",
                    "original_error": str(e),
                },
            ) from e


class UserManagementService:
    """User management service demonstrating operational exceptions."""

    def __init__(self) -> None:
        """Initialize UserManagementService."""
        self._users: dict[str, User] = {}
        self._deleted_users: set[str] = set()
        self._validation_service = UserValidationService()

    def create_user(self, user_data: Mapping[str, object]) -> FlextResult[User]:
        """Create new user with comprehensive error handling."""

        def _raise_validation_error(validation_result: FlextResult[User]) -> None:
            """Raise validation error with proper context."""
            msg = validation_result.error or "User validation failed"
            raise FlextValidationError(msg)

        def _raise_user_id_exists_error(user: User) -> None:
            """Raise error for existing user ID."""
            msg: str = f"User with ID {user.user_id} already exists"
            raise FlextAlreadyExistsError(
                msg,
                resource_id=user.user_id,
                resource_type="User",
                context={"conflicting_field": "user_id"},
            )

        def _raise_email_exists_error(user: User, existing_user: User) -> None:
            """Raise error for existing email."""
            msg: str = (
                f"Email {user.email} is already used by user {existing_user.user_id}"
            )
            raise FlextAlreadyExistsError(
                msg,
                resource_id=user.user_id,
                resource_type="User",
                context={
                    "conflicting_field": "email",
                    "conflicting_value": user.email,
                },
            )

        def _raise_none_user_error() -> None:
            """Raise error for None user."""
            msg = "Validation returned None user"
            raise FlextValidationError(msg)

        try:
            # Validate user data
            validation_result = self._validation_service.validate_user_data(user_data)
            if validation_result.is_failure:
                _raise_validation_error(validation_result)

            user = validation_result.value
            if user is None:
                _raise_none_user_error()

            # After None check, assert that user is not None for type narrowing
            if user is None:
                msg = "User is None after validation"
                raise AssertionError(msg)

            # Check if user already exists
            if user.user_id in self._users:
                _raise_user_id_exists_error(user)

            # Check if email is already used
            for existing_user in self._users.values():
                if existing_user.email == user.email:
                    _raise_email_exists_error(user, existing_user)

            # Save user
            self._users[user.user_id] = user
            return FlextResult[User].ok(user)

        except (FlextValidationError, FlextAlreadyExistsError) as e:
            return FlextResult[User].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            # Wrap unexpected exceptions
            msg = "User creation failed unexpectedly"
            raise FlextOperationError(
                msg,
                operation="user_creation",
                stage="user_persistence",
            ) from e

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID with proper error handling."""

        def _raise_deleted_user_error(user_id: str) -> None:
            """Raise error for deleted user."""
            msg: str = f"User {user_id} was deleted"
            raise FlextNotFoundError(
                msg,
                resource_id=user_id,
                resource_type="User",
                context={"status": "deleted"},
            )

        def _raise_user_not_found_error(user_id: str) -> None:
            """Raise error for user not found."""
            msg: str = f"User {user_id} not found"
            raise FlextNotFoundError(
                msg,
                resource_id=user_id,
                resource_type="User",
                context={"searched_in": "active_users"},
            )

        try:
            if user_id in self._deleted_users:
                _raise_deleted_user_error(user_id)

            if user_id not in self._users:
                _raise_user_not_found_error(user_id)

            return FlextResult[User].ok(self._users[user_id])

        except FlextNotFoundError as e:
            return FlextResult[User].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = "User retrieval failed unexpectedly"
            raise FlextOperationError(
                msg,
                operation="user_retrieval",
                stage="user_lookup",
            ) from e

    def delete_user(self, user_id: str, requester_id: str) -> FlextResult[None]:
        """Delete user with authorization and error handling."""

        def _raise_permission_error(requester_id: str, user_id: str) -> None:
            """Raise permission error for unauthorized deletion."""
            msg = "Insufficient permissions to delete user"
            raise FlextPermissionError(
                msg,
                service="user_management",
                required_permission="user_delete",
                context={
                    "requester_id": requester_id,
                    "target_resource": user_id,
                    "action": "delete_user",
                },
            )

        def _raise_user_not_found_for_delete(user_id: str) -> None:
            """Raise error for user not found during deletion."""
            msg: str = f"User {user_id} not found"
            raise FlextNotFoundError(
                msg,
                resource_id=user_id,
                resource_type="User",
            )

        try:
            # Check authorization (simplified)
            if requester_id not in {"admin", user_id}:
                _raise_permission_error(requester_id, user_id)

            # Check if user exists
            if user_id not in self._users:
                _raise_user_not_found_for_delete(user_id)

            # Delete user
            del self._users[user_id]
            self._deleted_users.add(user_id)

            return FlextResult[None].ok(None)

        except (FlextPermissionError, FlextNotFoundError) as e:
            return FlextResult[None].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = "User deletion failed unexpectedly"
            raise FlextOperationError(
                msg,
                operation="user_deletion",
                stage="user_removal",
            ) from e


class ConfigurationService:
    """Configuration service demonstrating configuration exceptions."""

    def __init__(self) -> None:
        """Initialize ConfigurationService."""
        self._config: dict[str, object] = {}

    def load_configuration(
        self, config_data: Mapping[str, object]
    ) -> FlextResult[None]:
        """Load and validate configuration."""

        def _raise_missing_config_error(
            key: str,
            required_keys: list[str],
            config_data: Mapping[str, object],
        ) -> None:
            """Raise error for missing configuration key."""
            msg: str = f"Missing required configuration: {key}"
            raise FlextConfigurationError(
                msg,
                config_key=key,
                context={
                    "missing_key": key,
                    "required_keys": required_keys,
                    "provided_keys": list(config_data.keys()),
                },
            )

        def _raise_invalid_db_url_error(db_url: object) -> None:
            """Raise error for invalid database URL format."""
            msg = "Invalid database URL format"
            raise FlextConfigurationError(
                msg,
                config_key="database_url",
                context={
                    "config_value": db_url,
                    "expected_format": "protocol://[user:pass@]host[:port]/database",
                },
            )

        def _raise_invalid_log_level_error(
            log_level: object,
            valid_levels: list[str],
        ) -> None:
            """Raise error for invalid log level."""
            msg: str = f"Invalid log level: {log_level}"
            raise FlextConfigurationError(
                msg,
                config_key="log_level",
                context={
                    "config_value": log_level,
                    "valid_values": valid_levels,
                },
            )

        try:
            # Required configuration keys
            required_keys = ["database_url", "api_key", "log_level"]

            for key in required_keys:
                if key not in config_data:
                    _raise_missing_config_error(key, required_keys, config_data)

            # Validate database URL format
            db_url = config_data["database_url"]
            if not isinstance(db_url, str) or not db_url.startswith(
                ("postgresql://", "mysql://", "sqlite://"),
            ):
                _raise_invalid_db_url_error(db_url)

            # Validate log level
            log_level = config_data["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if log_level not in valid_levels:
                _raise_invalid_log_level_error(log_level, valid_levels)

            # Save configuration
            self._config = dict(config_data)
            return FlextResult[None].ok(None)

        except FlextConfigurationError as e:
            return FlextResult[None].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = "Critical configuration loading failure"
            raise FlextCriticalError(
                msg,
                service="configuration_service",
                context={
                    "operation": "config_loading",
                    "component": "configuration_service",
                },
            ) from e


class ExternalAPIService:
    """External API service demonstrating connection and timeout exceptions."""

    def __init__(self, api_url: str, timeout_seconds: int = 30) -> None:
        """Initialize ExternalAPIService.

        Args:
            api_url: The URL of the external API
            timeout_seconds: The timeout in seconds for API requests

        """
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def fetch_user_profile(self, user_id: str) -> FlextResult[dict[str, object]]:
        """Fetch user profile from external API."""

        def _raise_connection_error() -> None:
            """Raise connection error for unreachable API."""
            msg = "Cannot connect to external API"
            raise FlextConnectionError(
                msg,
                service="external_api",
                endpoint=self.api_url,
                context={
                    "api_url": self.api_url,
                    "connection_timeout": self.timeout_seconds,
                    "error_type": "connection_refused",
                },
            )

        def _raise_timeout_error() -> None:
            """Raise timeout error for slow API."""
            msg = "API request timed out"
            raise FlextTimeoutError(
                msg,
                service="external_api",
                timeout_seconds=self.timeout_seconds,
                context={
                    "api_url": self.api_url,
                    "operation": "fetch_user_profile",
                },
            )

        def _raise_authentication_error() -> None:
            """Raise authentication error for unauthorized API."""
            msg = "API authentication failed"
            raise FlextAuthenticationError(
                msg,
                service="external_api",
                context={
                    "api_url": self.api_url,
                    "authentication_method": "bearer_token",
                    "token_type": os.environ.get("API_TOKEN_TYPE", "invalid_token"),
                },
            )

        try:
            # Simulate connection issues
            if "unreachable" in self.api_url:
                _raise_connection_error()

            # Simulate timeout
            if "slow" in self.api_url:
                _raise_timeout_error()

            # Simulate API authentication failure
            if "unauthorized" in self.api_url:
                _raise_authentication_error()

            # Simulate successful response
            profile_data: dict[str, object] = {
                "user_id": user_id,
                "external_id": f"ext_{user_id}",
                "profile_data": {
                    "avatar_url": f"https://avatars.example.com/{user_id}",
                    "last_seen": time.time(),
                    "preferences": {"theme": "dark", "language": "en"},
                },
            }

            return FlextResult[dict[str, object]].ok(profile_data)

        except (FlextConnectionError, FlextTimeoutError, FlextAuthenticationError) as e:
            return FlextResult[dict[str, object]].fail(str(e))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = "External API call failed unexpectedly"
            raise FlextProcessingError(
                msg,
                operation="external_api_call",
                context={
                    "stage": "response_processing",
                    "api_url": self.api_url,
                },
            ) from e


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_base_exceptions() -> None:
    """Demonstrate base exception functionality with context and observability."""
    # Clear metrics for clean demo
    clear_exception_metrics()

    # 1. Basic FlextError usage

    def _raise_service_init_error() -> None:
        """Simulate a service initialization error."""
        msg = "Service initialization failed"
        raise FlextError(
            msg,
            error_code="SERVICE_INIT_ERROR",
            context={
                "service_name": "UserService",
                "initialization_stage": "database_connection",
                "retry_count": 3,
                "last_error": "Connection timeout",
            },
        )

    try:
        # Simulate a basic error
        _raise_service_init_error()
    except FlextError as e:
        # Demonstrate serialization
        e.to_dict()

    # 2. Exception hierarchy demonstration

    # Create different types of exceptions
    exceptions_to_create = [
        FlextValidationError(
            "Email format invalid",
            validation_details={"field": "email", "value": "invalid"},
        ),
        FlextTypeError(
            "Expected string, got integer",
            expected_type="str",
            actual_type="int",
        ),
        FlextOperationError(
            "Database operation failed",
            operation="user_insert",
            stage="validation",
        ),
        FlextConfigurationError("Missing API key", config_key="api_key"),
        FlextConnectionError(
            "Database unreachable",
            service="database",
            context={"host": "db.example.com", "port": 5432},
        ),
        FlextAuthenticationError(
            "Invalid credentials",
            service="auth",
            context={"endpoint": "testuser"},
        ),
        FlextPermissionError(
            "Access denied",
            service="access_control",
            context={"resource": "user_data", "action": "read"},
        ),
        FlextNotFoundError("User not found", resource_id="user_123"),
        FlextAlreadyExistsError("User already exists", resource_id="user_123"),
        FlextTimeoutError(
            "Operation timed out",
            service="data_sync",
            timeout_seconds=30,
        ),
        FlextProcessingError("Data processing failed", operation="data_transform"),
        FlextCriticalError(
            "System overload",
            service="database",
            context={"component": "database", "severity": "high"},
        ),
    ]

    for exc in exceptions_to_create:
        try:
            raise exc
        except FlextError:
            pass

    # 3. Exception metrics

    metrics = get_exception_metrics()

    for _exc_type, _count in metrics.items():
        # exc_metrics is actually just the count (int), not a dict
        # For demonstration purposes, we'll simulate additional metrics
        pass  # Simulate last seen timestamp


def demonstrate_validation_exceptions() -> None:
    """Demonstrate validation exceptions with detailed context."""
    # 1. User validation service

    validation_service = UserValidationService()

    # Test cases with different validation errors
    test_cases = [
        {
            "name": "Valid User",
            "data": {"name": "John Doe", "email": "john@example.com", "age": 30},
            "should_pass": True,
        },
        {
            "name": "Missing name",
            "data": {"email": "test@example.com", "age": 25},
            "should_pass": False,
        },
        {
            "name": "Empty name",
            "data": {"name": "", "email": "test@example.com", "age": 25},
            "should_pass": False,
        },
        {
            "name": "Invalid email",
            "data": {"name": "Jane Doe", "email": "invalid-email", "age": 28},
            "should_pass": False,
        },
        {
            "name": "Invalid age type",
            "data": {"name": "Bob Smith", "email": "bob@example.com", "age": "thirty"},
            "should_pass": False,
        },
        {
            "name": "Age too young",
            "data": {"name": "Teen User", "email": "teen@example.com", "age": 16},
            "should_pass": False,
        },
        {
            "name": "Age too old",
            "data": {"name": "Old User", "email": "old@example.com", "age": 150},
            "should_pass": False,
        },
    ]

    for test_case in test_cases:
        data = test_case["data"]
        if isinstance(data, Mapping):
            result = validation_service.validate_user_data(data)
        else:
            result = FlextResult[User].fail("Invalid data type")

        if (result.success and test_case["should_pass"]) or (
            result.is_failure and not test_case["should_pass"]
        ):
            pass
        else:
            "pass" if test_case["should_pass"] else "fail"

    # 2. Factory method demonstration

    # Create validation error using factory
    factory = FlextExceptions()
    factory.create_validation_error(
        "Email domain not allowed",
        field="email",
        value="user@blocked-domain.com",
        rules=["email_format", "domain_whitelist"],
    )

    # Create type error directly
    FlextTypeError(
        "Configuration value must be integer",
        expected_type="int",
        actual_type="str",
    )


def demonstrate_operational_exceptions() -> None:
    """Demonstrate operational exceptions with service interactions."""
    # 1. User management operations

    user_service = UserManagementService()

    # Successful user creation
    valid_user_data = {
        "user_id": "user_001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
    }

    result = user_service.create_user(valid_user_data)
    if result.success:
        pass

    # Try to create duplicate user
    result = user_service.create_user(valid_user_data)
    if result.is_failure:
        pass

    # Try to create user with duplicate email
    duplicate_email_data = {
        "user_id": "user_002",
        "name": "Bob Wilson",
        "email": "alice@example.com",  # Same email
        "age": 35,
    }

    result = user_service.create_user(duplicate_email_data)
    if result.is_failure:
        pass

    # User retrieval operations

    # Get existing user
    result = user_service.get_user("user_001")
    if result.success:
        pass

    # Try to get non-existent user
    result = user_service.get_user("user_999")
    if result.is_failure:
        pass

    # User deletion operations

    # Unauthorized deletion
    delete_result1 = user_service.delete_user("user_001", "unauthorized_user")
    if delete_result1.is_failure:
        pass

    # Authorized deletion
    delete_result2 = user_service.delete_user("user_001", "admin")
    if delete_result2.success:
        pass

    # Try to retrieve deleted user
    result = user_service.get_user("user_001")
    if result.is_failure:
        pass


def demonstrate_configuration_exceptions() -> None:
    """Demonstrate configuration exceptions with validation."""
    # 1. Configuration service

    config_service = ConfigurationService()

    # Valid configuration
    valid_config: dict[str, object] = {
        "database_url": "postgresql://user:pass@localhost:5432/mydb",
        "api_key": "sk-1234567890abcdef",
        "log_level": "INFO",
        "optional_setting": "value",
    }

    result = config_service.load_configuration(valid_config)
    if result.success:
        pass

    # Test invalid configurations
    invalid_configs = [
        {
            "name": "Missing required key",
            "config": {
                "database_url": "postgresql://localhost/db",
                "log_level": "INFO",
                # Missing api_key
            },
        },
        {
            "name": "Invalid database URL",
            "config": {
                "database_url": "invalid-url",
                "api_key": "sk-test",
                "log_level": "INFO",
            },
        },
        {
            "name": "Invalid log level",
            "config": {
                "database_url": "postgresql://localhost/db",
                "api_key": "sk-test",
                "log_level": "INVALID",
            },
        },
    ]

    for test_config in invalid_configs:
        config_data = cast("dict[str, object]", test_config["config"])
        result = config_service.load_configuration(config_data)
        if result.is_failure:
            pass


def _demo_database_scenarios() -> None:
    """Show database connection/authentication scenarios."""
    # Successful connection
    db_conn = DatabaseConnection("localhost", 5432, "myapp_db")
    try:
        result = db_conn.connect()
        if result.success:
            # Successful authentication
            auth_result = db_conn.authenticate("admin", "secret")
            if auth_result.success:
                pass

    except FlextConnectionError:
        pass
    except FlextAuthenticationError:
        pass

    # Connection failure
    unreachable_db = DatabaseConnection("unreachable-host", 5432, "myapp_db")
    with contextlib.suppress(FlextConnectionError):
        unreachable_db.connect()

    # Authentication failure
    with contextlib.suppress(FlextAuthenticationError):
        db_conn.authenticate("wrong_user", "wrong_pass")


def _demo_external_api_scenarios() -> None:
    """Show external API connection/timeout/authentication scenarios."""
    # Successful API call
    api_service = ExternalAPIService("https://api.example.com/v1")
    api_result = api_service.fetch_user_profile("user_123")
    if api_result.success:
        pass

    # Connection error
    unreachable_api = ExternalAPIService(
        "https://unreachable-api.example.com/v1",
    )
    conn_result = unreachable_api.fetch_user_profile("user_123")
    if conn_result.is_failure:
        pass

    # Timeout error
    slow_api = ExternalAPIService(
        "https://slow-api.example.com/v1",
        timeout_seconds=1,
    )
    timeout_result = slow_api.fetch_user_profile("user_123")
    if timeout_result.is_failure:
        pass

    # Authentication error
    auth_api = ExternalAPIService("https://unauthorized-api.example.com/v1")
    auth_api_result = auth_api.fetch_user_profile("user_123")
    if auth_api_result.is_failure:
        pass


def demonstrate_connection_exceptions() -> None:
    """Demonstrate connection and timeout exceptions."""
    _demo_database_scenarios()
    _demo_external_api_scenarios()


def _complex_operation() -> FlextResult[str]:
    """Complex operation that can fail at multiple stages."""

    def _raise_config_error(_: FlextResult[None]) -> None:
        msg = "Complex operation failed at configuration stage"
        raise FlextOperationError(
            msg,
            operation="complex_operation",
            stage="configuration",
        )

    def _raise_validation_error(_: FlextResult[User]) -> None:
        msg = "Complex operation failed at validation stage"
        raise FlextOperationError(
            msg,
            operation="complex_operation",
            stage="user_validation",
        )

    def _raise_api_error(_: FlextResult[dict[str, object]]) -> None:
        msg = "Complex operation failed at API stage"
        raise FlextOperationError(
            msg,
            operation="complex_operation",
            stage="external_api",
        )

    try:
        config_service = ConfigurationService()
        config_result = config_service.load_configuration(
            {
                "database_url": "postgresql://localhost/db",
                "api_key": "sk-test",
                "log_level": "INFO",
            },
        )
        if config_result.is_failure:
            _raise_config_error(config_result)

        validation_service = UserValidationService()
        user_result = validation_service.validate_user_data(
            {"name": "Test User", "email": "test@example.com", "age": 25},
        )
        if user_result.is_failure:
            _raise_validation_error(user_result)

        api_service = ExternalAPIService("https://api.example.com/v1")
        api_result = api_service.fetch_user_profile("user_123")
        if api_result.is_failure:
            _raise_api_error(api_result)

        return FlextResult[str].ok("Complex operation completed successfully")

    except FlextOperationError as e:
        return FlextResult[str].fail(str(e))
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        msg = "Critical failure in complex operation"
        raise FlextCriticalError(
            msg,
            service="enterprise_service",
            context={
                "operation": "complex_operation",
                "component": "enterprise_service",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        ) from e


def _operation_with_retry(max_retries: int = 3) -> FlextResult[str]:
    """Operation with retry logic and exception handling."""

    def _simulate_operation_failure(attempt: int, max_retries: int) -> None:
        msg: str = f"Simulated failure on attempt {attempt + 1}"
        raise FlextConnectionError(
            msg,
            service="retry_operation",
            context={
                "attempt": attempt + 1,
                "max_retries": max_retries,
            },
        )

    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if attempt < MAX_RETRY_ATTEMPTS:
                _simulate_operation_failure(attempt, max_retries)
            return FlextResult[str].ok(f"Operation succeeded on attempt {attempt + 1}")
        except FlextConnectionError as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(0.01)
    return FlextResult[str].fail(
        f"Operation failed after {max_retries + 1} attempts: {last_exception}",
    )


def _print_exception_metrics() -> None:
    """Print aggregated exception metrics."""
    metrics = get_exception_metrics()
    sum(count for count in metrics.values())
    sorted_metrics = sorted(metrics.items(), key=operator.itemgetter(1), reverse=True)
    for _exc_type, _count in sorted_metrics[:5]:
        pass


def demonstrate_exception_patterns() -> None:
    """Demonstrate enterprise exception handling patterns."""
    result = _complex_operation()
    if result.success:
        pass

    retry_result = _operation_with_retry()
    if retry_result.success:
        pass

    _print_exception_metrics()


def main() -> None:
    """Execute all FlextExceptions demonstrations."""
    try:
        demonstrate_base_exceptions()
        demonstrate_validation_exceptions()
        demonstrate_operational_exceptions()
        demonstrate_configuration_exceptions()
        demonstrate_connection_exceptions()
        demonstrate_exception_patterns()

        # Final metrics summary
        final_metrics = get_exception_metrics()
        sum(count for count in final_metrics.values())

    except (ValueError, TypeError, ImportError, AttributeError):
        traceback.print_exc()


if __name__ == "__main__":
    main()
