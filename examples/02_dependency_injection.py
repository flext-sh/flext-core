#!/usr/bin/env python3
"""02 - Enhanced Dependency Injection using FlextCore Native Features.

Demonstrates advanced FlextCore dependency injection patterns showcasing:
• FlextCore.get_instance(): Singleton access to all FlextCore features
• Native validation using FlextCore built-in validators
• Railway-oriented programming with FlextResult composition
• Service registration with FlextCore.register_servi    # Apply performance tracking manually to avoid decorator typing issues
    tracker = cast(Callable[[Callable[[], FlextResult[str]]], Callable[[], FlextResult[str]]],
                   core.track_performance("demo_operation"))
    tracked_operation = tracker(demo_operation)
    result = tracked_operation()
    if result.success:
        print(f"✅ Performance-tracked operation: {result.value}")ctory patterns with FlextCore.register_factory()
• Configuration management with FlextCore.create_environment_core_config()
• Enterprise logging with structured context using FlextCore logging

Key Improvements over Previous Version:
• Eliminated custom validators in favor of FlextCore native validation
• Leveraged FlextCore.setup_container_with_services() for bulk service setup
• Used FlextCore railway-oriented patterns for composable operations
• Integrated FlextCore observability and performance tracking
• Simplified code by removing redundant classes and focusing on FlextCore patterns
"""

from __future__ import annotations

from typing import Protocol, cast

from flext_core import FlextCore, FlextResult, FlextTypes, FlextUtilities

# Get FlextCore singleton instance for all operations
core = FlextCore.get_instance()

# Configure structured logging for better observability
core.configure_logging(log_level="INFO", _json_output=True)

# Create application logger with proper naming
logger = core.FlextLogger("flext.examples.enhanced_di")

# =============================================================================
# DOMAIN MODELS USING FLEXTCORE NATIVE FEATURES
# Demonstrates proper use of FlextModels and FlextCore validation
# =============================================================================


class User:
    """User domain model using FlextCore validation patterns."""

    def __init__(self, name: str, email: str, age: int) -> None:
        """Initialize user with FlextCore validation."""
        self.name = name
        self.email = email
        self.age = age

    @classmethod
    def create(cls, name: str, email: str, age: int) -> FlextResult[User]:
        """Create user using FlextCore railway-oriented validation."""
        # Use FlextCore's built-in validation methods
        return (
            core.validate_string(name, min_length=2, max_length=100)
            .flat_map(lambda _: core.validate_email(email))
            .flat_map(lambda _: core.validate_numeric(age, min_value=0, max_value=150))
            .map(lambda _: User(name, email, age))
            .tap(
                lambda user: logger.info(
                    "User created successfully",
                    user_name=user.name,
                    user_email=user.email,
                )
            )
        )

    def __str__(self) -> str:
        """String representation of user."""
        return f"User(name='{self.name}', email='{self.email}', age={self.age})"


# =============================================================================
# SERVICE INTERFACES USING FLEXTCORE PROTOCOLS
# Demonstrates proper use of protocols for dependency inversion
# =============================================================================


class UserServiceProtocol(Protocol):
    """User service protocol for dependency inversion."""

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create a new user."""
        ...

    def find_user_by_email(self, email: str) -> FlextResult[User | None]:
        """Find user by email address."""
        ...


class NotificationServiceProtocol(Protocol):
    """Notification service protocol for loose coupling."""

    def send_welcome_email(self, user: User) -> FlextResult[None]:
        """Send welcome email to user."""
        ...


# =============================================================================
# SERVICE IMPLEMENTATIONS USING FLEXTCORE ENTERPRISE PATTERNS
# Demonstrates proper service implementation with FlextCore features
# =============================================================================


class EnterpriseUserService:
    """Enterprise user service using FlextCore patterns."""

    def __init__(self) -> None:
        """Initialize with FlextCore logger."""
        self._logger = core.FlextLogger("flext.services.user")
        self._users: dict[str, User] = {}

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create user with enterprise validation and logging."""
        # Check if user already exists
        if email in self._users:
            return FlextResult[User].fail(f"User with email {email} already exists")

        # Use User.create which includes FlextCore validation
        user_result = User.create(name, email, age)

        if user_result.success:
            self._users[email] = user_result.value
            self._logger.info(
                "User stored in service", email=email, total_users=len(self._users)
            )

        return user_result

    def find_user_by_email(self, email: str) -> FlextResult[User | None]:
        """Find user by email with validation."""
        # Validate email format first using FlextCore
        email_validation = core.validate_email(email)
        if not email_validation.success:
            return FlextResult[User | None].fail(f"Invalid email format: {email}")

        user = self._users.get(email)
        self._logger.info("User lookup performed", email=email, found=user is not None)
        return FlextResult[User | None].ok(user)


class MockNotificationService:
    """Mock notification service for demonstration."""

    def __init__(self) -> None:
        """Initialize with FlextCore logger."""
        self._logger = core.FlextLogger("flext.services.notification")

    def send_welcome_email(self, user: User) -> FlextResult[None]:
        """Mock sending welcome email."""
        # Simulate email validation and sending
        self._logger.info(
            "Welcome email sent", recipient=user.email, user_name=user.name
        )
        return FlextResult[None].ok(None)


# =============================================================================
# APPLICATION SERVICES USING FLEXTCORE DEPENDENCY INJECTION
# Demonstrates proper dependency injection patterns with FlextCore container
# =============================================================================


class UserRegistrationService:
    """User registration service demonstrating FlextCore DI patterns."""

    def __init__(
        self,
        user_service: UserServiceProtocol,
        notification_service: NotificationServiceProtocol,
    ) -> None:
        """Initialize with injected dependencies."""
        self._user_service = user_service
        self._notification_service = notification_service
        self._logger = core.FlextLogger("flext.services.registration")

    def register_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Register user using railway-oriented programming patterns."""
        # Chain operations using FlextResult composition
        return (
            self._user_service.create_user(name, email, age)
            .tap(
                lambda user: self._logger.info(
                    "User registration initiated", user_email=user.email
                )
            )
            .flat_map(self._send_welcome_notification)
            .tap(
                lambda user: self._logger.info(
                    "User registration completed successfully", user_email=user.email
                )
            )
        )

    def _send_welcome_notification(self, user: User) -> FlextResult[User]:
        """Send welcome notification and return user."""
        notification_result = self._notification_service.send_welcome_email(user)

        if notification_result.success:
            return FlextResult[User].ok(user)

        # Log warning but don't fail the entire registration
        self._logger.warning(
            "Welcome email failed, but user registration succeeded",
            user_email=user.email,
            error=notification_result.error,
        )
        return FlextResult[User].ok(user)


# =============================================================================
# FLEXTCORE CONTAINER CONFIGURATION
# Demonstrates proper use of FlextCore container features
# =============================================================================


def setup_dependency_injection() -> FlextResult[None]:
    """Setup dependency injection using FlextCore container features."""
    # Use FlextCore bulk service setup for better organization
    services_dict: dict[str, object] = {
        "user_service": EnterpriseUserService(),
        "notification_service": MockNotificationService(),
    }

    # Register services with validation using FlextCore
    # Create a validator that matches the expected signature
    def service_validator(name: str) -> FlextResult[object]:
        return core.validate_service_name(name).map(lambda _: name)

    container_result = core.setup_container_with_services(
        services_dict, validator=service_validator
    )

    if not container_result.success:
        return FlextResult[None].fail(
            f"Container setup failed: {container_result.error}"
        )

    # Register factory for registration service
    registration_factory_result = core.register_factory(
        "registration_service", _create_registration_service
    )

    if not registration_factory_result.success:
        return FlextResult[None].fail(
            f"Factory registration failed: {registration_factory_result.error}"
        )

    logger.info(
        "Dependency injection configured successfully",
        services_count=len(services_dict),
        factories_count=1,
    )

    return FlextResult[None].ok(None)


def _create_registration_service() -> UserRegistrationService:
    """Create registration service with dependency resolution."""
    # Use FlextCore service resolution
    user_service_result = core.get_service("user_service")
    notification_service_result = core.get_service("notification_service")

    if not user_service_result.success:
        msg = "User service not found"
        raise RuntimeError(msg)

    if not notification_service_result.success:
        msg = "Notification service not found"
        raise RuntimeError(msg)

    # Cast to proper protocols for type safety
    user_service = cast("UserServiceProtocol", user_service_result.value)
    notification_service = cast(
        "NotificationServiceProtocol", notification_service_result.value
    )

    return UserRegistrationService(user_service, notification_service)


# =============================================================================
# CONFIGURATION USING FLEXTCORE NATIVE FEATURES
# Demonstrates FlextCore environment-aware configuration
# =============================================================================


def setup_environment_configuration(
    environment: str = "development",
) -> FlextResult[None]:
    """Setup environment-specific configuration using FlextCore."""
    # Use FlextCore environment configuration
    # Cast string to proper Environment type
    env_type = cast("FlextTypes.Config.Environment", environment)
    config_result = core.create_environment_core_config(env_type)

    if not config_result.success:
        return FlextResult[None].fail(
            f"Environment config creation failed: {config_result.error}"
        )

    # Optimize performance based on environment
    optimized_config = core.optimize_core_performance(config_result.value)

    if not optimized_config.success:
        return FlextResult[None].fail(
            f"Performance optimization failed: {optimized_config.error}"
        )

    # Apply configuration to core system
    system_config_result = core.configure_core_system(optimized_config.value)

    if not system_config_result.success:
        return FlextResult[None].fail(
            f"System configuration failed: {system_config_result.error}"
        )

    logger.info(
        "Environment configuration completed",
        environment=environment,
        config_keys=len(system_config_result.value),
    )

    return FlextResult[None].ok(None)


# =============================================================================
# DEMONSTRATION FUNCTIONS
# Shows FlextCore dependency injection in action
# =============================================================================


def demonstrate_flextcore_di() -> None:
    """Demonstrate FlextCore dependency injection patterns."""
    print("=== FlextCore Enhanced Dependency Injection Demo ===")

    # Setup environment configuration
    env_setup = setup_environment_configuration("development")
    if not env_setup.success:
        print(f"❌ Environment setup failed: {env_setup.error}")
        return
    print("✅ Environment configuration completed")

    # Setup dependency injection
    di_setup = setup_dependency_injection()
    if not di_setup.success:
        print(f"❌ DI setup failed: {di_setup.error}")
        return
    print("✅ Dependency injection configured")

    # Get registration service from container
    registration_result = core.get_service("registration_service")
    if not registration_result.success:
        print(f"❌ Registration service not found: {registration_result.error}")
        return

    registration_service = cast("UserRegistrationService", registration_result.value)
    print("✅ Registration service retrieved from container")

    # Test user registration
    test_users = [
        ("Alice Johnson", "alice@example.com", 25),
        ("Bob Smith", "bob@company.com", 30),
        ("Carol Brown", "carol@domain.org", 28),
    ]

    for name, email, age in test_users:
        result = registration_service.register_user(name, email, age)

        if result.success:
            print(f"✅ Registered: {result.value}")
        else:
            print(f"❌ Registration failed for {name}: {result.error}")

    # Test duplicate registration
    duplicate_result = registration_service.register_user(
        "Alice Duplicate", "alice@example.com", 26
    )
    if not duplicate_result.success:
        print(f"✅ Duplicate prevention works: {duplicate_result.error}")

    # Test user lookup
    user_service_result = core.get_service("user_service")
    if user_service_result.success:
        user_service = cast("UserServiceProtocol", user_service_result.value)
        lookup_result = user_service.find_user_by_email("bob@company.com")
        if lookup_result.success and lookup_result.value:
            print(f"✅ User lookup successful: {lookup_result.value}")


def demonstrate_flextcore_features() -> None:
    """Demonstrate additional FlextCore features."""
    print("\n=== FlextCore Advanced Features Demo ===")

    # Health check
    health = core.health_check()
    if health.success:
        print(f"✅ System health check passed: {health.value.get('status', 'unknown')}")

    # System information
    system_info = core.get_system_info()
    print(f"✅ System info: Python {system_info.get('python_version', 'unknown')}")

    # Performance tracking demonstration
    def demo_operation() -> FlextResult[str]:
        """Demo operation with performance tracking."""
        return FlextResult[str].ok("Operation completed successfully")

    # Apply performance tracking using proper FlextUtilities with correct types
    @FlextUtilities.Performance.track_performance("demo_operation")
    def tracked_demo_operation() -> FlextResult[str]:
        """Demo operation with performance tracking using proper types."""
        return FlextResult[str].ok("Operation completed successfully")

    result = tracked_demo_operation()
    if result.success:
        print(f"✅ Performance-tracked operation: {result.value}")

    # Railway composition demonstration
    pipeline_result = (
        FlextResult[str]
        .ok("Hello World")
        .map(lambda s: s.upper())
        .map(lambda s: f"Message: {s}")
        .tap(lambda s: logger.info("Pipeline step completed", content=s))
    )

    if pipeline_result.success:
        print(f"✅ Railway pipeline: {pipeline_result.value}")


if __name__ == "__main__":
    print("FlextCore Enhanced Dependency Injection")
    print("======================================")

    # Demonstrate FlextCore dependency injection
    demonstrate_flextcore_di()

    # Demonstrate additional FlextCore features
    demonstrate_flextcore_features()

    print("\n✅ All FlextCore dependency injection patterns demonstrated successfully!")
    print("Key benefits: Native FlextCore validation, railway-oriented programming,")
    print(
        "enterprise logging, environment-aware configuration, and performance tracking."
    )
