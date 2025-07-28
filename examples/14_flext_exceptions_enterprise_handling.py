#!/usr/bin/env python3
"""FLEXT Exceptions Enterprise Handling Example.

Comprehensive demonstration of FlextExceptions system showing enterprise-grade
exception handling with observability, metrics tracking, and structured context
management for robust error handling across the application.

Features demonstrated:
    - Base exception hierarchy with FlextError foundation
    - Specialized exceptions with enhanced context information
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

import time
from typing import Any

from flext_core.exceptions import (
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
    FlextTimeoutError,
    FlextTypeError,
    FlextValidationError,
    clear_exception_metrics,
    get_exception_metrics,
)
from flext_core.result import FlextResult


# =============================================================================
# DOMAIN MODELS - Business entities for examples
# =============================================================================


class User:
    """User domain model for examples."""

    def __init__(
        self, user_id: str, name: str, email: str, age: int | None = None
    ) -> None:
        self.user_id = user_id
        self.name = name
        self.email = email
        self.age = age
        self.is_active = True

    def __repr__(self) -> str:
        return f"User(id={self.user_id}, name='{self.name}', email='{self.email}')"


class DatabaseConnection:
    """Mock database connection for examples."""

    def __init__(self, host: str, port: int, database: str) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.connected = False

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        if self.host == "unreachable-host":
            raise FlextConnectionError(
                "Database connection failed",
                host=self.host,
                port=self.port,
                database=self.database,
                timeout_seconds=30,
            )

        self.connected = True
        return FlextResult.ok(None)

    def authenticate(self, username: str, password: str) -> FlextResult[None]:
        """Authenticate with database."""
        if not self.connected:
            raise FlextOperationError(
                "Cannot authenticate without connection",
                operation="database_authentication",
                stage="pre_authentication_check",
            )

        if username != "REDACTED_LDAP_BIND_PASSWORD" or password != "secret":
            raise FlextAuthenticationError(
                "Invalid database credentials",
                username=username,
                database=self.database,
                authentication_method="username_password",
            )

        return FlextResult.ok(None)


# =============================================================================
# APPLICATION SERVICES - Business logic with exception handling
# =============================================================================


class UserValidationService:
    """User validation service demonstrating validation exceptions."""

    def validate_user_data(self, data: dict[str, Any]) -> FlextResult[User]:
        """Validate user data and create user."""
        try:
            # Validate required fields
            if "name" not in data:
                raise FlextValidationError(
                    "Name is required",
                    validation_details={
                        "field": "name",
                        "value": None,
                        "rules": ["required", "non_empty"],
                    },
                )

            name = data["name"]
            if not isinstance(name, str) or len(name.strip()) == 0:
                raise FlextValidationError(
                    "Name must be a non-empty string",
                    validation_details={
                        "field": "name",
                        "value": name,
                        "rules": ["string_type", "non_empty"],
                    },
                )

            # Validate email
            if "email" not in data:
                raise FlextValidationError(
                    "Email is required",
                    validation_details={
                        "field": "email",
                        "value": None,
                        "rules": ["required", "email_format"],
                    },
                )

            email = data["email"]
            if not isinstance(email, str) or "@" not in email:
                raise FlextValidationError(
                    "Email must be a valid email address",
                    validation_details={
                        "field": "email",
                        "value": email,
                        "rules": ["string_type", "email_format"],
                    },
                )

            # Validate age if provided
            age = data.get("age")
            if age is not None:
                if not isinstance(age, int):
                    raise FlextTypeError(
                        "Age must be an integer",
                        expected_type=int,
                        actual_type=type(age),
                    )

                if age < 18 or age > 120:
                    raise FlextValidationError(
                        "Age must be between 18 and 120",
                        validation_details={
                            "field": "age",
                            "value": age,
                            "rules": ["age_range"],
                        },
                    )

            # Create user
            user_id = data.get("user_id", f"user_{int(time.time())}")
            user = User(user_id, name.strip(), email.lower().strip(), age)

            return FlextResult.ok(user)

        except (FlextValidationError, FlextTypeError) as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            # Wrap unexpected exceptions
            raise FlextProcessingError(
                "Unexpected error during user validation",
                operation="user_validation",
                stage="data_processing",
                original_error=str(e),
            ) from e


class UserManagementService:
    """User management service demonstrating operational exceptions."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._deleted_users: set[str] = set()

    def create_user(self, user_data: dict[str, Any]) -> FlextResult[User]:
        """Create new user with comprehensive error handling."""
        try:
            # Validate user data
            validation_service = UserValidationService()
            validation_result = validation_service.validate_user_data(user_data)

            if validation_result.is_failure:
                # Re-raise validation errors for proper handling
                raise FlextValidationError(
                    validation_result.error or "User validation failed",
                    validation_details={"validation_result": validation_result.error},
                )

            user = validation_result.data

            # Check if user already exists
            if user.user_id in self._users:
                raise FlextAlreadyExistsError(
                    f"User with ID {user.user_id} already exists",
                    resource_id=user.user_id,
                    resource_type="User",
                )

            # Check if email is already used
            for existing_user in self._users.values():
                if existing_user.email == user.email:
                    raise FlextAlreadyExistsError(
                        f"User with email {user.email} already exists",
                        email=user.email,
                        existing_user_id=existing_user.user_id,
                        resource_type="User",
                    )

            # Save user
            self._users[user.user_id] = user
            return FlextResult.ok(user)

        except (FlextValidationError, FlextAlreadyExistsError) as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            # Wrap unexpected exceptions
            raise FlextOperationError(
                "User creation failed unexpectedly",
                operation="user_creation",
                stage="user_persistence",
            ) from e

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID with proper error handling."""
        try:
            if user_id in self._deleted_users:
                raise FlextNotFoundError(
                    f"User {user_id} was deleted",
                    resource_id=user_id,
                    resource_type="User",
                    status="deleted",
                )

            if user_id not in self._users:
                raise FlextNotFoundError(
                    f"User {user_id} not found",
                    resource_id=user_id,
                    resource_type="User",
                    searched_in="active_users",
                )

            return FlextResult.ok(self._users[user_id])

        except FlextNotFoundError as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            raise FlextOperationError(
                "User retrieval failed unexpectedly",
                operation="user_retrieval",
                stage="user_lookup",
            ) from e

    def delete_user(self, user_id: str, requester_id: str) -> FlextResult[None]:
        """Delete user with authorization and error handling."""
        try:
            # Check authorization (simplified)
            if requester_id != "REDACTED_LDAP_BIND_PASSWORD" and requester_id != user_id:
                raise FlextPermissionError(
                    "Insufficient permissions to delete user",
                    requester_id=requester_id,
                    target_resource=user_id,
                    required_permission="user_delete",
                    action="delete_user",
                )

            # Check if user exists
            if user_id not in self._users:
                raise FlextNotFoundError(
                    f"User {user_id} not found",
                    resource_id=user_id,
                    resource_type="User",
                )

            # Delete user
            del self._users[user_id]
            self._deleted_users.add(user_id)

            return FlextResult.ok(None)

        except (FlextPermissionError, FlextNotFoundError) as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            raise FlextOperationError(
                "User deletion failed unexpectedly",
                operation="user_deletion",
                stage="user_removal",
            ) from e


class ConfigurationService:
    """Configuration service demonstrating configuration exceptions."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def load_configuration(self, config_data: dict[str, Any]) -> FlextResult[None]:
        """Load and validate configuration."""
        try:
            # Required configuration keys
            required_keys = ["database_url", "api_key", "log_level"]

            for key in required_keys:
                if key not in config_data:
                    raise FlextConfigurationError(
                        f"Missing required configuration: {key}",
                        missing_key=key,
                        required_keys=required_keys,
                        provided_keys=list(config_data.keys()),
                    )

            # Validate database URL format
            db_url = config_data["database_url"]
            if not isinstance(db_url, str) or not db_url.startswith(
                ("postgresql://", "mysql://", "sqlite://")
            ):
                raise FlextConfigurationError(
                    "Invalid database URL format",
                    config_key="database_url",
                    config_value=db_url,
                    expected_format="protocol://[user:pass@]host[:port]/database",
                )

            # Validate log level
            log_level = config_data["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if log_level not in valid_levels:
                raise FlextConfigurationError(
                    f"Invalid log level: {log_level}",
                    config_key="log_level",
                    config_value=log_level,
                    valid_values=valid_levels,
                )

            # Save configuration
            self._config = dict(config_data)
            return FlextResult.ok(None)

        except FlextConfigurationError as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            raise FlextCriticalError(
                "Critical configuration loading failure",
                operation="config_loading",
                component="configuration_service",
            ) from e


class ExternalAPIService:
    """External API service demonstrating connection and timeout exceptions."""

    def __init__(self, api_url: str, timeout_seconds: int = 30) -> None:
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def fetch_user_profile(self, user_id: str) -> FlextResult[dict[str, Any]]:
        """Fetch user profile from external API."""
        try:
            # Simulate connection issues
            if "unreachable" in self.api_url:
                raise FlextConnectionError(
                    "Cannot connect to external API",
                    api_url=self.api_url,
                    connection_timeout=self.timeout_seconds,
                    error_type="connection_refused",
                )

            # Simulate timeout
            if "slow" in self.api_url:
                raise FlextTimeoutError(
                    "API request timed out",
                    api_url=self.api_url,
                    timeout_seconds=self.timeout_seconds,
                    operation="fetch_user_profile",
                )

            # Simulate API authentication failure
            if "unauthorized" in self.api_url:
                raise FlextAuthenticationError(
                    "API authentication failed",
                    api_url=self.api_url,
                    authentication_method="bearer_token",
                    error_code="invalid_token",
                )

            # Simulate successful response
            profile_data = {
                "user_id": user_id,
                "external_id": f"ext_{user_id}",
                "profile_data": {
                    "avatar_url": f"https://avatars.example.com/{user_id}",
                    "last_seen": time.time(),
                    "preferences": {"theme": "dark", "language": "en"},
                },
            }

            return FlextResult.ok(profile_data)

        except (FlextConnectionError, FlextTimeoutError, FlextAuthenticationError) as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            raise FlextProcessingError(
                "External API call failed unexpectedly",
                operation="external_api_call",
                stage="response_processing",
                api_url=self.api_url,
            ) from e


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_base_exceptions() -> None:
    """Demonstrate base exception functionality with context and observability."""
    print("\n" + "=" * 80)
    print("🔥 BASE EXCEPTIONS - FOUNDATION AND OBSERVABILITY")
    print("=" * 80)

    # Clear metrics for clean demo
    clear_exception_metrics()

    # 1. Basic FlextError usage
    print("\n1. Basic FlextError with context:")

    try:
        # Simulate a basic error
        raise FlextError(
            "Service initialization failed",
            error_code="SERVICE_INIT_ERROR",
            context={
                "service_name": "UserService",
                "initialization_stage": "database_connection",
                "retry_count": 3,
                "last_error": "Connection timeout",
            },
        )
    except FlextError as e:
        print(f"   ❌ Exception caught: {e}")
        print(f"   Error code: {e.error_code}")
        print(f"   Context: {e.context}")
        print(f"   Timestamp: {e.timestamp}")

        # Demonstrate serialization
        error_dict = e.to_dict()
        print(f"   Serialized: {error_dict}")

    # 2. Exception hierarchy demonstration
    print("\n2. Exception hierarchy with specialized exceptions:")

    # Create different types of exceptions
    exceptions_to_create = [
        FlextValidationError(
            "Email format invalid",
            validation_details={"field": "email", "value": "invalid"},
        ),
        FlextTypeError(
            "Expected string, got integer", expected_type=str, actual_type=int
        ),
        FlextOperationError(
            "Database operation failed", operation="user_insert", stage="validation"
        ),
        FlextConfigurationError("Missing API key", config_key="api_key"),
        FlextConnectionError("Database unreachable", host="db.example.com", port=5432),
        FlextAuthenticationError("Invalid credentials", username="testuser"),
        FlextPermissionError("Access denied", resource="user_data", action="read"),
        FlextNotFoundError("User not found", resource_id="user_123"),
        FlextAlreadyExistsError("User already exists", resource_id="user_123"),
        FlextTimeoutError(
            "Operation timed out", operation="data_sync", timeout_seconds=30
        ),
        FlextProcessingError("Data processing failed", operation="data_transform"),
        FlextCriticalError("System overload", component="database", severity="high"),
    ]

    for exc in exceptions_to_create:
        try:
            raise exc
        except FlextError as e:
            print(f"   ❌ {e.__class__.__name__}: {e}")

    # 3. Exception metrics
    print("\n3. Exception metrics and observability:")

    metrics = get_exception_metrics()
    print(f"   Total exception types tracked: {len(metrics)}")

    for exc_type, exc_metrics in metrics.items():
        count = exc_metrics.get("count", 0)
        error_codes = exc_metrics.get("error_codes", set())
        last_seen = exc_metrics.get("last_seen", 0)

        print(
            f"   {exc_type}: {count} occurrences, codes: {len(error_codes)}, last: {int(last_seen)}"
        )

    print("✅ Base exceptions demonstration completed")


def demonstrate_validation_exceptions() -> None:
    """Demonstrate validation exceptions with detailed context."""
    print("\n" + "=" * 80)
    print("✅ VALIDATION EXCEPTIONS - FIELD VALIDATION ERRORS")
    print("=" * 80)

    # 1. User validation service
    print("\n1. User validation service with detailed errors:")

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
        print(f"\n   Test: {test_case['name']}")
        result = validation_service.validate_user_data(test_case["data"])

        if result.is_success and test_case["should_pass"]:
            user = result.data
            print(f"     ✅ Validation passed: {user}")
        elif result.is_failure and not test_case["should_pass"]:
            print(f"     ❌ Validation failed (expected): {result.error}")
        else:
            status = "passed" if result.is_success else "failed"
            expected = "pass" if test_case["should_pass"] else "fail"
            print(f"     ⚠️ Unexpected result: {status} (expected {expected})")

    # 2. Factory method demonstration
    print("\n2. Exception factory methods:")

    # Create validation error using factory
    validation_error = FlextExceptions.create_validation_error(
        "Email domain not allowed",
        field="email",
        value="user@blocked-domain.com",
        rules=["email_format", "domain_whitelist"],
    )

    print(f"   Factory-created validation error: {validation_error}")
    print(f"   Context: {validation_error.context}")

    # Create type error using factory
    type_error = FlextExceptions.create_type_error(
        "Configuration value must be integer", expected_type=int, actual_type=str
    )

    print(f"   Factory-created type error: {type_error}")
    print(f"   Context: {type_error.context}")

    print("✅ Validation exceptions demonstration completed")


def demonstrate_operational_exceptions() -> None:
    """Demonstrate operational exceptions with service interactions."""
    print("\n" + "=" * 80)
    print("⚙️ OPERATIONAL EXCEPTIONS - SERVICE OPERATIONS")
    print("=" * 80)

    # 1. User management operations
    print("\n1. User management service operations:")

    user_service = UserManagementService()

    # Successful user creation
    valid_user_data = {
        "user_id": "user_001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
    }

    result = user_service.create_user(valid_user_data)
    if result.is_success:
        user = result.data
        print(f"   ✅ User created: {user}")
    else:
        print(f"   ❌ User creation failed: {result.error}")

    # Try to create duplicate user
    result = user_service.create_user(valid_user_data)
    if result.is_failure:
        print(f"   ❌ Duplicate user creation prevented (expected): {result.error}")

    # Try to create user with duplicate email
    duplicate_email_data = {
        "user_id": "user_002",
        "name": "Bob Wilson",
        "email": "alice@example.com",  # Same email
        "age": 35,
    }

    result = user_service.create_user(duplicate_email_data)
    if result.is_failure:
        print(f"   ❌ Duplicate email prevented (expected): {result.error}")

    # User retrieval operations
    print("\n   User retrieval operations:")

    # Get existing user
    result = user_service.get_user("user_001")
    if result.is_success:
        user = result.data
        print(f"   ✅ User retrieved: {user}")

    # Try to get non-existent user
    result = user_service.get_user("user_999")
    if result.is_failure:
        print(f"   ❌ User not found (expected): {result.error}")

    # User deletion operations
    print("\n   User deletion operations:")

    # Unauthorized deletion
    result = user_service.delete_user("user_001", "unauthorized_user")
    if result.is_failure:
        print(f"   ❌ Unauthorized deletion prevented (expected): {result.error}")

    # Authorized deletion
    result = user_service.delete_user("user_001", "REDACTED_LDAP_BIND_PASSWORD")
    if result.is_success:
        print("   ✅ User deleted by REDACTED_LDAP_BIND_PASSWORD")

    # Try to retrieve deleted user
    result = user_service.get_user("user_001")
    if result.is_failure:
        print(f"   ❌ Deleted user not found (expected): {result.error}")

    print("✅ Operational exceptions demonstration completed")


def demonstrate_configuration_exceptions() -> None:
    """Demonstrate configuration exceptions with validation."""
    print("\n" + "=" * 80)
    print("🔧 CONFIGURATION EXCEPTIONS - SETTINGS VALIDATION")
    print("=" * 80)

    # 1. Configuration service
    print("\n1. Configuration loading and validation:")

    config_service = ConfigurationService()

    # Valid configuration
    valid_config = {
        "database_url": "postgresql://user:pass@localhost:5432/mydb",
        "api_key": "sk-1234567890abcdef",
        "log_level": "INFO",
        "optional_setting": "value",
    }

    result = config_service.load_configuration(valid_config)
    if result.is_success:
        print("   ✅ Valid configuration loaded successfully")
    else:
        print(f"   ❌ Configuration loading failed: {result.error}")

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
        print(f"\n   Test: {test_config['name']}")
        result = config_service.load_configuration(test_config["config"])
        if result.is_failure:
            print(f"     ❌ Configuration invalid (expected): {result.error}")
        else:
            print("     ⚠️ Configuration unexpectedly accepted")

    print("✅ Configuration exceptions demonstration completed")


def demonstrate_connection_exceptions() -> None:
    """Demonstrate connection and timeout exceptions."""
    print("\n" + "=" * 80)
    print("🌐 CONNECTION EXCEPTIONS - NETWORK AND TIMEOUTS")
    print("=" * 80)

    # 1. Database connection errors
    print("\n1. Database connection scenarios:")

    # Successful connection
    db_conn = DatabaseConnection("localhost", 5432, "myapp_db")
    try:
        result = db_conn.connect()
        if result.is_success:
            print("   ✅ Database connection successful")

            # Successful authentication
            auth_result = db_conn.authenticate("REDACTED_LDAP_BIND_PASSWORD", "secret")
            if auth_result.is_success:
                print("   ✅ Database authentication successful")

    except FlextConnectionError as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   Context: {e.context}")
    except FlextAuthenticationError as e:
        print(f"   ❌ Authentication failed: {e}")
        print(f"   Context: {e.context}")

    # Connection failure
    print("\n   Testing connection failure:")
    unreachable_db = DatabaseConnection("unreachable-host", 5432, "myapp_db")
    try:
        unreachable_db.connect()
    except FlextConnectionError as e:
        print(f"   ❌ Connection failed (expected): {e}")
        print(f"   Error context: {e.context}")

    # Authentication failure
    print("\n   Testing authentication failure:")
    try:
        auth_result = db_conn.authenticate("wrong_user", "wrong_pass")
    except FlextAuthenticationError as e:
        print(f"   ❌ Authentication failed (expected): {e}")
        print(f"   Error context: {e.context}")

    # 2. External API connection scenarios
    print("\n2. External API scenarios:")

    # Successful API call
    api_service = ExternalAPIService("https://api.example.com/v1")
    result = api_service.fetch_user_profile("user_123")
    if result.is_success:
        profile = result.data
        print(f"   ✅ API call successful: {profile}")

    # Connection error
    unreachable_api = ExternalAPIService("https://unreachable-api.example.com/v1")
    result = unreachable_api.fetch_user_profile("user_123")
    if result.is_failure:
        print(f"   ❌ API connection failed (expected): {result.error}")

    # Timeout error
    slow_api = ExternalAPIService("https://slow-api.example.com/v1", timeout_seconds=1)
    result = slow_api.fetch_user_profile("user_123")
    if result.is_failure:
        print(f"   ❌ API timeout (expected): {result.error}")

    # Authentication error
    auth_api = ExternalAPIService("https://unauthorized-api.example.com/v1")
    result = auth_api.fetch_user_profile("user_123")
    if result.is_failure:
        print(f"   ❌ API authentication failed (expected): {result.error}")

    print("✅ Connection exceptions demonstration completed")


def demonstrate_exception_patterns() -> None:
    """Demonstrate enterprise exception handling patterns."""
    print("\n" + "=" * 80)
    print("🏢 ENTERPRISE EXCEPTION PATTERNS")
    print("=" * 80)

    # 1. Exception chaining and context preservation
    print("\n1. Exception chaining and context preservation:")

    def complex_operation() -> FlextResult[str]:
        """Complex operation that can fail at multiple stages."""
        try:
            # Stage 1: Configuration
            config_service = ConfigurationService()
            config_result = config_service.load_configuration(
                {
                    "database_url": "postgresql://localhost/db",
                    "api_key": "sk-test",
                    "log_level": "INFO",
                }
            )

            if config_result.is_failure:
                raise FlextOperationError(
                    "Complex operation failed at configuration stage",
                    operation="complex_operation",
                    stage="configuration",
                    underlying_error=config_result.error,
                )

            # Stage 2: User validation
            validation_service = UserValidationService()
            user_result = validation_service.validate_user_data(
                {"name": "Test User", "email": "test@example.com", "age": 25}
            )

            if user_result.is_failure:
                raise FlextOperationError(
                    "Complex operation failed at validation stage",
                    operation="complex_operation",
                    stage="user_validation",
                    underlying_error=user_result.error,
                )

            # Stage 3: External API call
            api_service = ExternalAPIService("https://api.example.com/v1")
            api_result = api_service.fetch_user_profile("user_123")

            if api_result.is_failure:
                raise FlextOperationError(
                    "Complex operation failed at API stage",
                    operation="complex_operation",
                    stage="external_api",
                    underlying_error=api_result.error,
                )

            return FlextResult.ok("Complex operation completed successfully")

        except FlextOperationError as e:
            return FlextResult.fail(str(e))
        except Exception as e:
            # Wrap any unexpected exceptions
            raise FlextCriticalError(
                "Critical failure in complex operation",
                operation="complex_operation",
                component="enterprise_service",
                exception_type=type(e).__name__,
                exception_message=str(e),
            ) from e

    # Execute complex operation
    result = complex_operation()
    if result.is_success:
        print(f"   ✅ Complex operation: {result.data}")
    else:
        print(f"   ❌ Complex operation failed: {result.error}")

    # 2. Exception recovery patterns
    print("\n2. Exception recovery patterns:")

    def operation_with_retry(max_retries: int = 3) -> FlextResult[str]:
        """Operation with retry logic and exception handling."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Simulate operation that might fail
                if attempt < 2:  # Fail first 2 attempts
                    raise FlextConnectionError(
                        f"Simulated failure on attempt {attempt + 1}",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )

                # Success on final attempt
                return FlextResult.ok(f"Operation succeeded on attempt {attempt + 1}")

            except FlextConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"   ⚠️ Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.01)  # Brief delay
                else:
                    print(f"   ❌ All retries exhausted: {e}")

        # If we get here, all retries failed
        return FlextResult.fail(
            f"Operation failed after {max_retries + 1} attempts: {last_exception}"
        )

    # Test retry operation
    retry_result = operation_with_retry()
    if retry_result.is_success:
        print(f"   ✅ Retry operation: {retry_result.data}")
    else:
        print(f"   ❌ Retry operation failed: {retry_result.error}")

    # 3. Exception metrics and monitoring
    print("\n3. Exception metrics and monitoring:")

    # Get current metrics
    metrics = get_exception_metrics()

    print("   Exception metrics summary:")
    total_exceptions = sum(
        int(metrics_data.get("count", 0)) for metrics_data in metrics.values()
    )
    print(f"   Total exceptions tracked: {total_exceptions}")

    # Show top exception types
    sorted_metrics = sorted(
        metrics.items(), key=lambda x: int(x[1].get("count", 0)), reverse=True
    )

    print("   Top exception types:")
    for exc_type, exc_data in sorted_metrics[:5]:  # Top 5
        count = exc_data.get("count", 0)
        error_codes = exc_data.get("error_codes", set())
        print(
            f"     {exc_type}: {count} occurrences, {len(error_codes)} unique error codes"
        )

    print("✅ Enterprise exception patterns demonstration completed")


def main() -> None:
    """Execute all FlextExceptions demonstrations."""
    print("🚀 FLEXT EXCEPTIONS - ENTERPRISE HANDLING EXAMPLE")
    print(
        "Demonstrating comprehensive exception handling patterns for enterprise applications"
    )

    try:
        demonstrate_base_exceptions()
        demonstrate_validation_exceptions()
        demonstrate_operational_exceptions()
        demonstrate_configuration_exceptions()
        demonstrate_connection_exceptions()
        demonstrate_exception_patterns()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT EXCEPTIONS DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of capabilities demonstrated:")
        print("   🔥 Base exceptions with observability and context management")
        print("   ✅ Validation exceptions with detailed field-specific context")
        print("   ⚙️ Operational exceptions with service interaction patterns")
        print("   🔧 Configuration exceptions with settings validation")
        print("   🌐 Connection exceptions with network and timeout handling")
        print("   🏢 Enterprise patterns with chaining, recovery, and monitoring")
        print("\n💡 FlextExceptions provides enterprise-grade error handling")
        print(
            "   with observability, structured context, and comprehensive exception hierarchy!"
        )

        # Final metrics summary
        final_metrics = get_exception_metrics()
        total_tracked = sum(
            int(data.get("count", 0)) for data in final_metrics.values()
        )
        print(f"\n📈 Total exceptions tracked during demo: {total_tracked}")
        print(f"📊 Exception types encountered: {len(final_metrics)}")

    except Exception as e:
        print(f"\n❌ Error during FlextExceptions demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
