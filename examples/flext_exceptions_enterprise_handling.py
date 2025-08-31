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
    - FlextExceptions: Base exception with observability and context management
    - FlextExceptions: Field validation failures with detailed context
    - FlextExceptions.TypeError: Type mismatch errors with expected/actual type info
    - FlextExceptions: Operation failures with stage tracking
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
from collections.abc import Mapping
from typing import cast

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextResult,
    FlextTypes,
)

# =============================================================================
# EXCEPTION CONSTANTS - Using FlextConstants centralized approach
# =============================================================================

# Age validation constants - using FlextConstants pattern
MIN_USER_AGE: int = 18
MAX_USER_AGE: int = 120

# Retry attempt constants - using FlextConstants
MAX_RETRY_ATTEMPTS: int = 2

# =============================================================================
# DOMAIN MODELS - Business entities for examples
# =============================================================================


class User:
    """User domain model following FLEXT patterns.

    Business domain entity for user management with proper type annotations
    using FlextTypes centralized type system.
    """

    def __init__(
        self,
        user_id: FlextTypes.Domain.EntityId,
        name: FlextTypes.Core.String,
        email: FlextTypes.Core.String,
        age: int | None = None,
    ) -> None:
        """Initialize user with proper FlextTypes annotations.

        Args:
            user_id: Unique identifier using FlextTypes.Domain.EntityId
            name: User display name using FlextTypes.Core.String
            email: User email using FlextTypes.Core.String
            age: Optional user age using int

        """
        self.user_id: FlextTypes.Domain.EntityId = user_id
        self.name: FlextTypes.Core.String = name
        self.email: FlextTypes.Core.String = email
        self.age: int | None = age
        self.is_active: FlextTypes.Core.Boolean = True

    def __repr__(self) -> FlextTypes.Core.String:
        """Return string representation using FlextTypes.Core.String."""
        return f"User(id={self.user_id}, name='{self.name}', email='{self.email}')"


class DatabaseConnection:
    """Database connection following FLEXT patterns.

    Infrastructure service for database connectivity with proper type annotations
    using FlextTypes centralized type system.
    """

    def __init__(
        self,
        host: FlextTypes.Core.String,
        port: int,
        database: FlextTypes.Core.String,
    ) -> None:
        """Initialize database connection with FlextTypes annotations.

        Args:
            host: Database host using FlextTypes.Core.String
            port: Database port using int
            database: Database name using FlextTypes.Core.String

        """
        self.host: FlextTypes.Core.String = host
        self.port: int = port
        self.database: FlextTypes.Core.String = database
        self.connected: FlextTypes.Core.Boolean = False

    def connect(self) -> FlextResult[None]:
        """Connect to database using FlextResult pattern."""
        if self.host == "unreachable-host":
            error_msg: FlextTypes.Core.String = "Database connection failed"
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
            )

        self.connected = True
        return FlextResult[None].ok(None)

    def authenticate(
        self, username: FlextTypes.Core.String, password: FlextTypes.Core.String
    ) -> FlextResult[None]:
        """Authenticate with database using FlextResult pattern."""
        if not self.connected:
            error_msg: FlextTypes.Core.String = "Cannot authenticate without connection"
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.OPERATION_ERROR,
            )

        # Demo credentials - in production use environment variables or secure vault
        demo_username: FlextTypes.Core.String = "REDACTED_LDAP_BIND_PASSWORD"
        # Password hardcoded for demo only - use os.environ["DB_PASSWORD"] in production
        demo_password: FlextTypes.Core.String = "demo_secret_not_for_production"  # noqa: S105
        if username != demo_username or password != demo_password:
            msg: FlextTypes.Core.String = "Invalid database credentials"
            return FlextResult[None].fail(
                msg,
                error_code=FlextConstants.Errors.AUTHENTICATION_ERROR,
            )

        return FlextResult[None].ok(None)


# =============================================================================
# APPLICATION SERVICES - Business logic with exception handling
# =============================================================================


class UserValidationService:
    """User validation service following FLEXT patterns.

    Application service for user validation using FlextResult patterns
    instead of raising exceptions directly.
    """

    @staticmethod
    def _ensure_string_type(
        value: FlextTypes.Core.Object, field: FlextTypes.Core.String
    ) -> FlextResult[FlextTypes.Core.String]:
        """Ensure value is string using FlextResult pattern."""
        if not isinstance(value, str):
            msg: FlextTypes.Core.String = f"{field} is not a string after validation"
            return FlextResult[FlextTypes.Core.String].fail(
                msg,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )
        return FlextResult[FlextTypes.Core.String].ok(value)

    @staticmethod
    def _validate_name_required(data: FlextTypes.Core.Dict) -> FlextResult[None]:
        """Validate name field using FlextResult pattern."""
        if "name" not in data:
            msg: FlextTypes.Core.String = "Name is required"
            return FlextResult[None].fail(
                msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[None].ok(None)

    @staticmethod
    def _validate_name_format(
        name: FlextTypes.Core.Object,
    ) -> FlextResult[FlextTypes.Core.String]:
        """Validate name format using FlextResult pattern."""
        if not isinstance(name, str) or len(name.strip()) == 0:
            msg: FlextTypes.Core.String = "Name must be a non-empty string"
            return FlextResult[FlextTypes.Core.String].fail(
                msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[FlextTypes.Core.String].ok(name)

    @staticmethod
    def _validate_email_required(data: FlextTypes.Core.Dict) -> FlextResult[None]:
        """Validate email field using FlextResult pattern."""
        if "email" not in data:
            msg: FlextTypes.Core.String = "Email is required"
            return FlextResult[None].fail(
                msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[None].ok(None)

    @staticmethod
    def _validate_email_format(email: object) -> None:
        """Validate email format and type."""
        if not isinstance(email, str) or "@" not in email:
            msg = "Email must be a valid email address"
            raise FlextExceptions.ValidationError(
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
            raise FlextExceptions.TypeError(
                msg,
                expected_type="int",
                actual_type=type(age).__name__,
            )

    @staticmethod
    def _validate_age_range(age: int) -> None:
        """Validate age range."""
        if age < MIN_USER_AGE or age > MAX_USER_AGE:
            msg: str = f"Age must be between {MIN_USER_AGE} and {MAX_USER_AGE}"
            raise FlextExceptions.ValidationError(
                msg,
                validation_details={
                    "field": "age",
                    "value": age,
                    "rules": ["age_range"],
                },
            )

    def _extract_string_or_raise(self, result: FlextResult[str]) -> str:
        """Extract string from FlextResult or raise appropriate exception."""
        if result.is_failure:
            error_msg = result.error or "Type error occurred"
            raise TypeError(error_msg)
        return result.unwrap()

    def _handle_validation_failure(self, result: FlextResult[None]) -> None:
        """Handle validation failure by raising appropriate exception."""
        if result.is_failure:
            error_msg = result.error or "Validation failed"
            raise ValueError(error_msg)

    def _raise_name_format_error(
        self, result: FlextResult[FlextTypes.Core.String]
    ) -> None:
        """Raise name format validation error."""
        error_msg = result.error or "Name format validation failed"
        raise ValueError(error_msg)

    def validate_user_data(self, data: dict[str, object]) -> FlextResult[User]:
        """Validate user data and create a User instance.

        Raises validation and type errors for invalid input.
        """
        try:
            # Validate required fields
            name_required_result = self._validate_name_required(data)
            self._handle_validation_failure(name_required_result)

            name = data["name"]
            name_format_result = self._validate_name_format(name)
            if name_format_result.is_failure:
                self._raise_name_format_error(name_format_result)
            # After validation, ensure str type using railway pattern
            name_result = self._ensure_string_type(name, "Name")
            name_str = self._extract_string_or_raise(name_result)

            # Validate email
            email_required_result = self._validate_email_required(data)
            self._handle_validation_failure(email_required_result)

            email = data["email"]
            # _validate_email_format raises exception directly, no FlextResult returned
            self._validate_email_format(email)
            email_result = self._ensure_string_type(email, "Email")
            email_str = self._extract_string_or_raise(email_result)

            # Validate age if provided
            age_obj = data.get("age")
            age: int | None = None
            if age_obj is not None:
                # _validate_age_type and _validate_age_range raise exceptions directly
                self._validate_age_type(age_obj)
                # After validation, we can safely cast to int
                age = cast("int", age_obj)
                self._validate_age_range(age)

            # Create user
            user_id_obj = data.get("user_id", f"user_{int(time.time())}")
            user_id_result = self._ensure_string_type(user_id_obj, "User ID")
            user_id = self._extract_string_or_raise(user_id_result)

            user = User(
                user_id,
                name_str.strip(),
                email_str.lower().strip(),
                age,
            )

            return FlextResult[User].ok(user)

        except Exception as e:
            # Check if it's one of our expected exception types using dynamic access
            validation_error_class = FlextExceptions
            type_error_class = FlextExceptions.TypeError

            if isinstance(e, (validation_error_class, type_error_class)):
                return FlextResult[User].fail(str(e))
            if isinstance(e, (ValueError, TypeError, KeyError, AttributeError)):
                # Wrap unexpected exceptions
                msg = "Unexpected error during user validation"
                processing_error_class = FlextExceptions.ProcessingError
                raise processing_error_class(
                    msg,
                    operation="user_validation",
                    context={
                        "stage": "data_processing",
                        "original_error": str(e),
                    },
                ) from e
            # Re-raise unknown exceptions
            raise


class UserManagementService:
    """User management service demonstrating operational exceptions."""

    def __init__(self) -> None:
        """Initialize UserManagementService."""
        self._users: dict[str, User] = {}
        self._deleted_users: set[str] = set()
        self._validation_service = UserValidationService()

    def create_user(self, user_data: dict[str, object]) -> FlextResult[User]:
        """Create new user with comprehensive error handling."""

        def _raise_validation_error(validation_result: FlextResult[User]) -> None:
            """Raise validation error with proper context."""
            msg = validation_result.error or "User validation failed"
            raise FlextExceptions.ValidationError(msg)

        def _raise_user_id_exists_error(user: User) -> None:
            """Raise error for existing user ID."""
            msg: str = f"User with ID {user.user_id} already exists"
            raise FlextExceptions.AlreadyExistsError(
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
            raise FlextExceptions.AlreadyExistsError(
                msg,
                resource_id=user.user_id,
                resource_type="User",
                context={
                    "conflicting_field": "email",
                    "conflicting_value": user.email,
                },
            )

        try:
            # Validate user data
            validation_result = self._validation_service.validate_user_data(user_data)
            if validation_result.is_failure:
                _raise_validation_error(validation_result)

            user = validation_result.value

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

        except (ValueError, TypeError) as e:
            return FlextResult[User].fail(str(e))
        except (KeyError, AttributeError) as e:
            # Wrap unexpected exceptions
            msg = "User creation failed unexpectedly"
            raise FlextExceptions.OperationError(
                msg,
                operation="user_creation",
                stage="user_persistence",
            ) from e

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID with proper error handling."""

        def _raise_deleted_user_error(user_id: str) -> None:
            """Raise error for deleted user."""
            msg: str = f"User {user_id} was deleted"
            raise FlextExceptions.NotFoundError(
                msg,
                resource_id=user_id,
                resource_type="User",
                context={"status": "deleted"},
            )

        def _raise_user_not_found_error(user_id: str) -> None:
            """Raise error for user not found."""
            msg: str = f"User {user_id} not found"
            raise FlextExceptions.NotFoundError(
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

        except KeyError as e:
            return FlextResult[User].fail(str(e))
        except (ValueError, TypeError, AttributeError) as e:
            msg = "User retrieval failed unexpectedly"
            raise FlextExceptions.OperationError(
                msg,
                operation="user_retrieval",
                stage="user_lookup",
            ) from e

    def delete_user(self, user_id: str, requester_id: str) -> FlextResult[None]:
        """Delete user with authorization and error handling."""

        def _raise_permission_error(requester_id: str, user_id: str) -> None:
            """Raise permission error for unauthorized deletion."""
            msg = "Insufficient permissions to delete user"
            raise FlextExceptions.PermissionError(
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
            raise FlextExceptions.NotFoundError(
                msg,
                resource_id=user_id,
                resource_type="User",
            )

        try:
            # Check authorization (simplified)
            if requester_id not in {"REDACTED_LDAP_BIND_PASSWORD", user_id}:
                _raise_permission_error(requester_id, user_id)

            # Check if user exists
            if user_id not in self._users:
                _raise_user_not_found_for_delete(user_id)

            # Delete user
            del self._users[user_id]
            self._deleted_users.add(user_id)

            return FlextResult[None].ok(None)

        except (ValueError, KeyError) as e:
            return FlextResult[None].fail(str(e))
        except (TypeError, AttributeError) as e:
            msg = "User deletion failed unexpectedly"
            raise FlextExceptions.OperationError(
                msg,
                operation="user_deletion",
                stage="user_removal",
            ) from e


class ConfigurationService:
    """Configuration service demonstrating configuration exceptions."""

    def __init__(self) -> None:
        """Initialize ConfigurationService."""
        self._config: dict[str, object] = {}

    def load_configuration(self, config_data: dict[str, object]) -> FlextResult[None]:
        """Load and validate configuration."""

        def _raise_missing_config_error(
            key: str,
            required_keys: list[str],
            config_data: Mapping[str, object],
        ) -> None:
            """Raise error for missing configuration key."""
            msg: str = f"Missing required configuration: {key}"
            raise FlextExceptions.ConfigurationError(
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
            raise FlextExceptions.ConfigurationError(
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
            raise FlextExceptions.ConfigurationError(
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

        except (ValueError, TypeError) as e:
            return FlextResult[None].fail(str(e))
        except (KeyError, AttributeError) as e:
            msg = "Critical configuration loading failure"
            raise FlextExceptions.CriticalError(
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
            raise FlextExceptions.ConnectionError(
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
            raise FlextExceptions.TimeoutError(
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
            raise FlextExceptions.AuthenticationError(
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

        except (ValueError, TypeError, ConnectionError) as e:
            return FlextResult[dict[str, object]].fail(str(e))
        except (KeyError, AttributeError) as e:
            msg = "External API call failed unexpectedly"
            raise FlextExceptions.ProcessingError(
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
    FlextExceptions.clear_metrics()

    # 1. Basic FlextExceptions usage

    def _raise_service_init_error() -> None:
        """Simulate a service initialization error."""
        msg = "Service initialization failed"
        raise FlextExceptions.CriticalError(
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
    except Exception as e:
        # Demonstrate serialization using available attributes
        if (
            hasattr(e, "error_code")
            and getattr(e, "error_code", None) == "SERVICE_INIT_ERROR"
        ):
            pass

    # 2. Exception hierarchy demonstration

    exceptions_to_create = [
        FlextExceptions.ValidationError(
            "Email format invalid",
            validation_details={"field": "email", "value": "invalid"},
        ),
        FlextExceptions.TypeError(
            "Expected string, got integer",
            expected_type="str",
            actual_type="int",
        ),
        FlextExceptions.OperationError(
            "Database operation failed",
            operation="user_insert",
            stage="validation",
        ),
        FlextExceptions.ConfigurationError("Missing API key", config_key="api_key"),
        FlextExceptions.ConnectionError(
            "Database unreachable",
            service="database",
            context={"host": "db.example.com", "port": 5432},
        ),
        FlextExceptions.AuthenticationError(
            "Invalid credentials",
            service="auth",
            context={"endpoint": "testuser"},
        ),
        FlextExceptions.PermissionError(
            "Access denied",
            service="access_control",
            context={"resource": "user_data", "action": "read"},
        ),
        FlextExceptions.NotFoundError("User not found", resource_id="user_123"),
        FlextExceptions.AlreadyExistsError(
            "User already exists", resource_id="user_123"
        ),
        FlextExceptions.TimeoutError(
            "Operation timed out",
            service="data_sync",
            timeout_seconds=30,
        ),
        FlextExceptions.ProcessingError(
            "Data processing failed", operation="data_transform"
        ),
        FlextExceptions.CriticalError(
            "System overload",
            service="database",
            context={"component": "database", "severity": "high"},
        ),
    ]

    def _raise_example_exception(exc: Exception) -> None:
        """Raise exception for demonstration purposes."""
        raise exc

    for exc in exceptions_to_create:
        try:
            _raise_example_exception(exc)
        except Exception as e:
            if hasattr(e, "error_code"):
                pass

    # 3. Exception metrics

    metrics = FlextExceptions.get_metrics()

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
        try:
            if isinstance(data, Mapping):
                # Convert Mapping to dict for proper type compatibility
                data_dict = dict(data) if not isinstance(data, dict) else data
                validation_service.validate_user_data(data_dict)
            else:
                FlextResult[User].fail("Invalid data type")

            # If we get here, validation passed
            if test_case["should_pass"]:
                pass  # Expected success
            else:
                pass  # Unexpected success - should have raised exception
        except FlextExceptions.BaseError:
            # Exception was raised as expected for invalid data
            if not test_case["should_pass"]:
                pass  # Expected failure via exception
            else:
                pass  # Unexpected failure via exception

    # 2. Factory method demonstration

    # Create validation error directly using dynamic class access
    msg = "Email domain not allowed"
    _validation_error = FlextExceptions.ValidationError(
        msg,
        field="email",
        value="user@blocked-domain.com",
        rules=["email_format", "domain_whitelist"],
    )

    # Create type error directly using dynamic class access
    msg = "Configuration value must be integer"
    _type_error = FlextExceptions.TypeError(
        msg,
        expected_type="int",
        actual_type="str",
    )

    # Examples created (not raised for demonstration)


def demonstrate_operational_exceptions() -> None:  # noqa: PLR0912
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

    # Try to create duplicate user (expecting AlreadyExistsError exception)
    try:
        result = user_service.create_user(valid_user_data)
        if result.is_failure:
            pass
    except FlextExceptions.AlreadyExistsError:
        # Expected exception for duplicate user creation
        pass

    # Try to create user with duplicate email (expecting AlreadyExistsError exception)
    duplicate_email_data = {
        "user_id": "user_002",
        "name": "Bob Wilson",
        "email": "alice@example.com",  # Same email
        "age": 35,
    }

    try:
        result = user_service.create_user(duplicate_email_data)
        if result.is_failure:
            pass
    except FlextExceptions.AlreadyExistsError:
        # Expected exception for duplicate email creation
        pass

    # User retrieval operations

    # Get existing user
    result = user_service.get_user("user_001")
    if result.success:
        pass

    # Try to get non-existent user (expecting NotFoundError exception)
    try:
        result = user_service.get_user("user_999")
        if result.is_failure:
            pass
    except FlextExceptions.NotFoundError:
        # Expected exception for user not found
        pass

    # User deletion operations

    # Unauthorized deletion (expecting PermissionError exception)
    try:
        delete_result1 = user_service.delete_user("user_001", "unauthorized_user")
        if delete_result1.is_failure:
            pass
    except FlextExceptions.PermissionError:
        # Expected exception for unauthorized deletion
        pass

    # Authorized deletion
    delete_result2 = user_service.delete_user("user_001", "REDACTED_LDAP_BIND_PASSWORD")
    if delete_result2.success:
        pass

    # Try to retrieve deleted user (expecting NotFoundError exception)
    try:
        result = user_service.get_user("user_001")
        if result.is_failure:
            pass
    except FlextExceptions.NotFoundError:
        # Expected exception for deleted user retrieval
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
            auth_result = db_conn.authenticate("REDACTED_LDAP_BIND_PASSWORD", "secret")
            if auth_result.success:
                pass

    except FlextExceptions.ConnectionError:
        pass

    # Connection failure
    unreachable_db = DatabaseConnection("unreachable-host", 5432, "myapp_db")
    with contextlib.suppress(Exception):
        unreachable_db.connect()

    # Authentication failure
    with contextlib.suppress(Exception):
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
        raise FlextExceptions.OperationError(
            msg,
            operation="complex_operation",
            stage="configuration",
        )

    def _raise_validation_error(_: FlextResult[User]) -> None:
        msg = "Complex operation failed at validation stage"
        raise FlextExceptions.OperationError(
            msg,
            operation="complex_operation",
            stage="user_validation",
        )

    def _raise_api_error(_: FlextResult[dict[str, object]]) -> None:
        msg = "Complex operation failed at API stage"
        raise FlextExceptions.OperationError(
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

    except FlextExceptions.BaseError as e:
        return FlextResult[str].fail(str(e))
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        msg = "Critical failure in complex operation"
        raise FlextExceptions.CriticalError(
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
        raise FlextExceptions.ConnectionError(
            msg,
            service="retry_operation",
            context={
                "attempt": attempt + 1,
                "max_retries": max_retries,
            },
        )

    last_exception: FlextExceptions.BaseError | None = None
    for attempt in range(max_retries + 1):
        try:
            if attempt < MAX_RETRY_ATTEMPTS:
                _simulate_operation_failure(attempt, max_retries)
            return FlextResult[str].ok(f"Operation succeeded on attempt {attempt + 1}")
        except FlextExceptions.BaseError as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(0.01)
    return FlextResult[str].fail(
        f"Operation failed after {max_retries + 1} attempts: {last_exception}",
    )


def _print_exception_metrics() -> None:
    """Print aggregated exception metrics."""
    metrics = FlextExceptions.get_metrics()
    sum(count for count in metrics.values())  # counts are already int
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
    demonstration_functions = [
        ("Base Exceptions", demonstrate_base_exceptions),
        ("Validation Exceptions", demonstrate_validation_exceptions),
        ("Operational Exceptions", demonstrate_operational_exceptions),
        ("Configuration Exceptions", demonstrate_configuration_exceptions),
        ("Connection Exceptions", demonstrate_connection_exceptions),
        ("Exception Patterns", demonstrate_exception_patterns),
    ]

    # Run each demonstration and continue even if individual demos raise exceptions
    for _, demo_func in demonstration_functions:
        with contextlib.suppress(Exception):
            # Demonstrations are expected to raise exceptions for educational purposes
            # Continue to next demonstration
            demo_func()

    # Final metrics summary
    with contextlib.suppress(Exception):
        # Even metrics can fail in demonstration - that's okay
        final_metrics = FlextExceptions.get_metrics()
        _ = sum(count for count in final_metrics.values())  # counts are already int


if __name__ == "__main__":
    main()
