#!/usr/bin/env python3
"""02 - Dependency Injection: Minimal Boilerplate with Maximum FlextCore Power.

Demonstrates how to use FlextCore's existing functionality to minimize boilerplate
while maintaining full enterprise functionality and patterns.

Key Features:
• FlextCore utilities for validation and entity creation
• FlextDecorators for performance, logging, and error handling
• FlextServiceProcessor templates for service logic
• FlextContainer for dependency injection
• FlextLoggerFactory for structured logging
• Minimal boilerplate with maximum functionality
"""

from __future__ import annotations

from typing import Annotated, override

from pydantic import Field, field_validator

from flext_core import (
    FlextCore,
    FlextDecorators,
    FlextEntity,
    FlextModel,
    FlextResult,
    FlextResultUtils,
    FlextServiceProcessor,
    FlextUtilities,
    FlextValue,
    get_flext_container,
    validate_email_address,
)
from flext_core.utilities import FlextProcessingUtils
from flext_core.validation import FlextAbstractValidator

# =============================================================================
# CORE SETUP - SINGLE INSTANCE WITH UTILITIES
# =============================================================================

core = FlextCore.get_instance()
logger = core.get_logger("flext.examples.minimal_di")

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_AGE = 18
MAX_AGE = 120
WARNING_AGE = 21
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100

# =============================================================================
# VALIDATORS - USING FLEXT CORE UTILITIES
# =============================================================================


class UserAgeValidator(FlextAbstractValidator[int]):
    """Age validator using FlextCore utilities."""

    @override
    def validate(self, value: int) -> FlextResult[int]:
        """Validate age using core utilities."""
        # Use FlextCore validation utilities
        numeric_result = core.validate_numeric(value, MIN_AGE, MAX_AGE)
        if numeric_result.is_failure:
            return FlextResult[int].fail(
                numeric_result.error or "Age validation failed"
            )

        # Business rule: warn for under 21
        if value < WARNING_AGE:
            logger.warning("User is under 21", age=value, category="age_validation")

        return FlextResult[int].ok(value)


class UserNameValidator(FlextAbstractValidator[str]):
    """Name validator using FlextCore utilities."""

    @override
    def validate(self, value: str) -> FlextResult[str]:
        """Validate name using core utilities."""
        # Use FlextCore string validation
        string_result = core.validate_string(value, MIN_NAME_LENGTH, MAX_NAME_LENGTH)
        if string_result.is_failure:
            return FlextResult[str].fail(
                string_result.error or "Name validation failed"
            )

        # Business rule: warn for numbers in name
        if any(char.isdigit() for char in value):
            logger.warning(
                "Name contains numbers", name=value, category="name_validation"
            )

        return FlextResult[str].ok(value.strip().title())


# =============================================================================
# DOMAIN ENTITIES - USING FLEXT CORE CREATION
# =============================================================================


class User(FlextEntity):
    """User entity with FlextCore validation."""

    name: Annotated[str, Field(min_length=MIN_NAME_LENGTH, max_length=MAX_NAME_LENGTH)]
    email: Annotated[str, Field(min_length=5, max_length=254)]
    age: Annotated[int, Field(ge=MIN_AGE, le=MAX_AGE)]
    status: str = "pending"
    registration_id: str = Field(default_factory=lambda: core.generate_uuid()[:10])
    service_tier: str = "standard"

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        """Validate name using FlextCore validator."""
        validator = UserNameValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or "Name validation failed"
            raise ValueError(error_msg)
        return result.value

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: str) -> str:
        """Validate email using FlextCore utility."""
        email_result = core.validate_email(v)
        if email_result.is_failure:
            error_msg = email_result.error or "Email validation failed"
            raise ValueError(error_msg)
        return email_result.value

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, v: int) -> int:
        """Validate age using FlextCore validator."""
        validator = UserAgeValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or "Age validation failed"
            raise ValueError(error_msg)
        return result.value


class EmailAddress(FlextValue):
    """Email value object with FlextCore validation."""

    email: Annotated[str, Field(min_length=5, max_length=254)]

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: str) -> str:
        """Validate email using FlextCore utility."""
        try:
            return validate_email_address(v)
        except Exception as e:
            error_msg = f"Email validation failed: {e}"
            raise ValueError(error_msg) from e

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Business rules validation using FlextCore patterns."""
        if ".invalid" in self.email:
            return FlextResult[None].fail("Invalid email domain")
        if ".test" in self.email:
            logger.warning("Test email domain detected", email=self.email)
        return FlextResult[None].ok(None)


class UserRegistrationRequest(FlextValue):
    """Registration request with FlextCore validation."""

    name: Annotated[str, Field(min_length=MIN_NAME_LENGTH, max_length=MAX_NAME_LENGTH)]
    email: Annotated[str, Field(min_length=5, max_length=254)]
    age: Annotated[int, Field(ge=MIN_AGE, le=MAX_AGE)]
    preferred_service_tier: str = "standard"

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        """Validate name using FlextCore validator."""
        validator = UserNameValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or "Name validation failed"
            raise ValueError(error_msg)
        return result.value

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: str) -> str:
        """Validate email using FlextCore utility."""
        email_result = FlextCore.validate_email(v)
        if email_result.is_failure:
            error_msg = email_result.error or "Email validation failed"
            raise ValueError(error_msg)
        return email_result.value

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, v: int) -> int:
        """Validate age using FlextCore validator."""
        validator = UserAgeValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or "Age validation failed"
            raise ValueError(error_msg)
        return result.value

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Business rules validation using FlextCore patterns."""
        valid_tiers = ["standard", "premium", "enterprise"]
        if self.preferred_service_tier not in valid_tiers:
            return FlextResult[None].fail("Invalid service tier")
        return FlextResult[None].ok(None)


# =============================================================================
# RESULT MODELS - USING FLEXT MODEL
# =============================================================================


class DatabaseSaveResult(FlextModel):
    """Database save result with FlextCore patterns."""

    user_id: str
    table_name: str = "users"
    operation: str = "INSERT"
    processing_time_ms: float
    correlation_id: str = Field(default_factory=core.generate_uuid)
    cache_key: str | None = None
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class EmailSendResult(FlextModel):
    """Email send result with FlextCore patterns."""

    recipient: str
    subject: str = "Welcome!"
    status: str
    processing_time_ms: float
    correlation_id: str = Field(default_factory=core.generate_uuid)
    delivery_attempts: int = 1
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class RegistrationResult(FlextModel):
    """Complete registration result with FlextCore patterns."""

    user_id: str
    status: str
    database_result: DatabaseSaveResult
    email_result: EmailSendResult
    total_processing_time_ms: float
    correlation_id: str = Field(default_factory=core.generate_uuid)
    validation_warnings: list[str] = Field(default_factory=list)
    service_metrics: dict[str, float] = Field(default_factory=dict)


# =============================================================================
# SERVICE PROCESSORS - USING FLEXT CORE DECORATORS
# =============================================================================


class DatabaseServiceProcessor(
    FlextServiceProcessor[User, DatabaseSaveResult, DatabaseSaveResult]
):
    """Database processor with FlextCore decorators."""

    def __init__(self) -> None:
        super().__init__()
        self._logger = core.get_logger("flext.services.database")
        self._cache: dict[str, DatabaseSaveResult] = {}

    @override
    def process(self, request: User) -> FlextResult[DatabaseSaveResult]:
        """Process database save using FlextCore utilities."""
        self._logger.info(
            "Processing database save",
            user_id=str(request.id),
            table="users",
            operation="INSERT",
        )

        # Use FlextUtilities for timing
        save_time_ms = FlextUtilities.get_last_duration_ms("database", "_inner") or 5.0

        # Create performance metrics
        performance_metrics = {
            "query_time_ms": save_time_ms * 0.7,
            "connection_time_ms": save_time_ms * 0.2,
            "serialization_time_ms": save_time_ms * 0.1,
        }

        cache_key = f"user_{request.id}"

        # Use FlextCore entity creation pattern
        result = DatabaseSaveResult(
            user_id=str(request.id),
            processing_time_ms=save_time_ms,
            cache_key=cache_key,
            performance_metrics=performance_metrics,
        )

        self._logger.info(
            "Database save completed",
            user_id=str(request.id),
            processing_time_ms=save_time_ms,
            cache_key=cache_key,
        )

        return FlextResult[DatabaseSaveResult].ok(result)

    @override
    def build(
        self, domain: DatabaseSaveResult, *, correlation_id: str
    ) -> DatabaseSaveResult:
        """Build result using FlextCore patterns."""
        domain.correlation_id = correlation_id
        return domain

    def save_user(self, user: User) -> FlextResult[DatabaseSaveResult]:
        """Public interface using FlextCore template."""
        return self.run_with_metrics("database", user)


class EmailServiceProcessor(
    FlextServiceProcessor[User, EmailSendResult, EmailSendResult]
):
    """Email processor with FlextCore decorators."""

    def __init__(self) -> None:
        super().__init__()
        self._logger = core.get_logger("flext.services.email")

    @override
    def process(self, request: User) -> FlextResult[EmailSendResult]:
        """Process email send using FlextCore utilities."""
        self._logger.info(
            "Processing email send",
            recipient=request.email,
            user_id=str(request.id),
            service_tier=request.service_tier,
        )

        # Validate email using FlextCore value object
        email_validation = EmailAddress(email=request.email).validate_business_rules()
        if email_validation.is_failure:
            self._logger.error(
                "Email validation failed",
                recipient=request.email,
                error=email_validation.error,
            )
            return FlextResult[EmailSendResult].fail(
                email_validation.error or "Email validation failed"
            )

        # Use FlextUtilities for timing
        send_time_ms = FlextUtilities.get_last_duration_ms("email", "_inner") or 3.0

        # Create performance metrics
        performance_metrics = {
            "smtp_connect_ms": send_time_ms * 0.4,
            "template_render_ms": send_time_ms * 0.3,
            "send_time_ms": send_time_ms * 0.3,
        }

        result = EmailSendResult(
            recipient=request.email,
            status="sent",
            processing_time_ms=send_time_ms,
            performance_metrics=performance_metrics,
        )

        self._logger.info(
            "Email send completed",
            recipient=request.email,
            status="sent",
            processing_time_ms=send_time_ms,
        )

        return FlextResult[EmailSendResult].ok(result)

    @override
    def build(self, domain: EmailSendResult, *, correlation_id: str) -> EmailSendResult:
        """Build result using FlextCore patterns."""
        domain.correlation_id = correlation_id
        return domain

    def send_welcome_email(self, user: User) -> FlextResult[EmailSendResult]:
        """Public interface using FlextCore template."""
        return self.run_with_metrics("email", user)


class UserRegistrationProcessor(
    FlextServiceProcessor[UserRegistrationRequest, User, RegistrationResult]
):
    """Registration processor with FlextCore patterns."""

    def __init__(self) -> None:
        super().__init__()
        self._logger = core.get_logger("flext.services.registration")
        self._database_service = DatabaseServiceProcessor()
        self._email_service = EmailServiceProcessor()

    @override
    def process(self, request: UserRegistrationRequest) -> FlextResult[User]:
        """Process registration using FlextCore utilities."""
        self._logger.info(
            "Processing user registration",
            email=request.email,
            name=request.name,
            age=request.age,
            service_tier=request.preferred_service_tier,
        )

        # Validate business rules using FlextCore patterns
        validation_result = request.validate_business_rules()
        if validation_result.is_failure:
            self._logger.error(
                "Registration validation failed",
                email=request.email,
                error=validation_result.error,
            )
            return FlextResult[User].fail(
                validation_result.error or "Validation failed"
            )

        # Create user entity using FlextCore
        user_result = core.create_entity(
            User,
            id=f"user_{core.generate_uuid()[:10]}",
            name=request.name,
            email=request.email,
            age=request.age,
            status="active",
            service_tier=request.preferred_service_tier,
        )

        if user_result.is_success:
            self._logger.info(
                "User entity created",
                user_id=str(user_result.value.id),
                name=user_result.value.name,
                service_tier=user_result.value.service_tier,
            )

        return user_result

    @override
    def build(self, domain: User, *, correlation_id: str) -> RegistrationResult:
        """Build registration result using FlextCore orchestration."""
        self._logger.info(
            "Building registration result",
            user_id=str(domain.id),
            correlation_id=correlation_id,
        )

        # Use dependency injection to call services
        db_result = self._database_service.save_user(domain)
        email_result = self._email_service.send_welcome_email(domain)

        # Collect validation warnings using FlextCore patterns
        validation_warnings: list[str] = []
        if domain.age < WARNING_AGE:
            validation_warnings.append("User is under 21")
        if ".test" in domain.email:
            validation_warnings.append("Test email domain")

        # Calculate total processing time
        total_time = (
            db_result.value.processing_time_ms if db_result.is_success else 0
        ) + (email_result.value.processing_time_ms if email_result.is_success else 0)

        # Collect service metrics using FlextUtilities
        service_metrics: dict[str, float] = {}
        for key, data in FlextUtilities.iter_metrics_items():
            # Handle the new metrics structure
            if isinstance(data, dict) and "performance" in data:
                perf_data = data["performance"]
                if "duration" in perf_data:
                    service_metrics[key] = float(perf_data["duration"]) * 1000
            else:
                service_metrics[key] = 0.0

        result = RegistrationResult(
            user_id=str(domain.id),
            status=str(domain.status),
            database_result=db_result.value
            if db_result.is_success
            else DatabaseSaveResult(user_id=str(domain.id), processing_time_ms=0.0),
            email_result=email_result.value
            if email_result.is_success
            else EmailSendResult(
                recipient=domain.email, status="failed", processing_time_ms=0.0
            ),
            total_processing_time_ms=total_time,
            correlation_id=correlation_id,
            validation_warnings=validation_warnings,
            service_metrics=service_metrics,
        )

        self._logger.info(
            "Registration result built",
            user_id=str(domain.id),
            total_time_ms=total_time,
            warnings_count=len(validation_warnings),
            correlation_id=correlation_id,
        )

        return result

    def register_user(
        self, request: UserRegistrationRequest
    ) -> FlextResult[RegistrationResult]:
        """Public interface using FlextCore template."""
        process_result = self.process(request)
        if process_result.is_failure:
            return FlextResult[RegistrationResult].fail(str(process_result.error))

        user = process_result.value
        registration_result = self.build(user, correlation_id="reg_123")
        return FlextResult[RegistrationResult].ok(registration_result)


# =============================================================================
# CONTAINER SETUP - USING FLEXT CORE UTILITIES
# =============================================================================


def setup_container() -> FlextResult[None]:
    """Setup container using FlextCore utilities."""
    container = get_flext_container()

    # Use FlextCore service name validation
    def service_validator(name: str) -> FlextResult[str]:
        return core.validate_service_name(name)

    # Register service processors
    services = {
        "database_service": DatabaseServiceProcessor,
        "email_service": EmailServiceProcessor,
        "registration_service": UserRegistrationProcessor,
    }

    for service_name, service_class in services.items():
        # Validate service name using FlextCore
        validation_result = service_validator(service_name)
        if validation_result.is_failure:
            logger.error(
                "Service name validation failed",
                service_name=service_name,
                error=validation_result.error,
            )
            continue

        # Create factory function with explicit typing
        def create_service_factory(cls: type[object]) -> object:
            return cls()

        # Create factory function instead of lambda to avoid linting issues
        def create_factory(cls: type[object] = service_class) -> object:
            return create_service_factory(cls)

        register_result = container.register_factory(service_name, create_factory)

        if register_result.is_failure:
            return FlextResult[None].fail(
                f"Failed to register {service_name}: {register_result.error}"
            )

        logger.info(
            "Service registered",
            service_name=service_name,
            service_class=service_class.__name__,
        )

    logger.info(
        "Container setup completed",
        total_services=len(services),
        registered_services=list(services.keys()),
    )

    return FlextResult[None].ok(None)


def get_service_with_fallback[T](service_name: str, default_factory: type[T]) -> T:
    """Get service from container with FlextCore utilities."""
    container = get_flext_container()
    result = container.get(service_name)

    if result.is_success:
        logger.debug("Service retrieved from container", service_name=service_name)
        return result.value  # type: ignore[return-value]

    logger.warning(
        "Service not found in container, using default factory",
        service_name=service_name,
        default_factory=default_factory.__name__,
    )
    return default_factory()


# =============================================================================
# UTILITY FUNCTIONS - USING FLEXT CORE PATTERNS
# =============================================================================


def log_result[T](result: FlextResult[T], success_msg: str) -> FlextResult[T]:
    """Utility to log FlextResult using FlextCore patterns."""
    if result.is_success:
        logger.info(f"✅ {success_msg}", result_type=type(result.value).__name__)
        print(f"✅ {success_msg}: {result.value}")
        return result

    logger.error(f"❌ {success_msg} failed", error=result.error)
    print(f"❌ Error: {result.error}")
    return result


# =============================================================================
# DEMONSTRATIONS - USING FLEXT CORE DECORATORS
# =============================================================================


@FlextDecorators.time_execution
def demo_service_injection(*_args: object, **_kwargs: object) -> object:
    """Demonstrate service injection with FlextCore utilities."""
    print("\n🔧 Service Injection with FlextCore Utilities")
    print("=" * 50)

    setup_result = setup_container()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return "setup_failed"

    # Get service using FlextCore utilities
    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Create request using FlextCore validation
    request = UserRegistrationRequest(
        name="Alice Johnson",
        email="alice@company.com",
        age=28,
        preferred_service_tier="premium",
    )

    result = log_result(
        registration_service.register_user(request), "Service injection registration"
    )
    return result.value if result.is_success else None


@FlextDecorators.time_execution
def demo_batch_processing(*_args: object, **_kwargs: object) -> object:
    """Demonstrate batch processing with FlextCore utilities."""
    print("\n📊 Batch Processing with FlextCore Utilities")
    print("=" * 50)

    setup_result = setup_container()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return "setup_failed"

    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Create batch requests using FlextCore patterns
    requests = [
        UserRegistrationRequest(
            name="User 1",
            email="user1@company.com",
            age=25,
            preferred_service_tier="standard",
        ),
        UserRegistrationRequest(
            name="User 2",
            email="user2@company.com",
            age=30,
            preferred_service_tier="premium",
        ),
        UserRegistrationRequest(
            name="User 3",
            email="user3@company.com",
            age=35,
            preferred_service_tier="enterprise",
        ),
    ]

    # Use FlextResultUtils for batch processing
    successes, failures = FlextResultUtils.batch_process(
        requests, registration_service.register_user
    )

    print(f"🎯 Batch Processed: {len(successes)} success, {len(failures)} failed")

    # Show service metrics using FlextUtilities
    print("📊 Service Metrics:")
    for key, data in FlextUtilities.iter_metrics_items():
        # Handle the new metrics structure
        if isinstance(data, dict) and "performance" in data:
            perf_data = data["performance"]
            if "duration" in perf_data and "count" in perf_data:
                print(
                    f"  • {key}: {perf_data['duration'] * 1000:.2f}ms ({perf_data['count']} calls)"
                )
            else:
                print(f"  • {key}: {len(data)} metrics")
        else:
            print(f"  • {key}: {len(data) if isinstance(data, dict) else data}")
    return "batch_processing_completed"


@FlextDecorators.time_execution
def demo_json_processing(*_args: object, **_kwargs: object) -> object:
    """Demonstrate JSON processing with FlextCore utilities."""
    print("\n🔄 JSON Processing with FlextCore Utilities")
    print("=" * 50)

    setup_result = setup_container()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return "setup_failed"

    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Process valid JSON using FlextUtilities
    valid_json = '{"name": "JSON User", "email": "json@company.com", "age": 32, "preferred_service_tier": "premium"}'

    json_result = FlextProcessingUtils.parse_json_to_model(
        valid_json, UserRegistrationRequest
    )
    if json_result.is_success:
        log_result(
            registration_service.register_user(json_result.value),
            "JSON service processing",
        )
    else:
        print(f"❌ JSON parsing failed: {json_result.error}")

    # Process invalid JSON to show validation
    invalid_json = '{"name": "Invalid User", "email": "invalid.invalid", "age": 15}'

    invalid_result = FlextProcessingUtils.parse_json_to_model(
        invalid_json, UserRegistrationRequest
    )
    if invalid_result.is_failure:
        print(f"❌ JSON validation failed (expected): {invalid_result.error}")
    return "json_processing_completed"


@FlextDecorators.time_execution
def demo_advanced_patterns(*_args: object, **_kwargs: object) -> object:
    """Demonstrate advanced patterns with FlextCore utilities."""
    print("\n🔧 Advanced Patterns with FlextCore Utilities")
    print("=" * 50)

    setup_result = setup_container()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return "setup_failed"

    # Use FlextCore functional programming patterns
    requests = [
        UserRegistrationRequest(
            name="Enterprise User1",
            email="u1@enterprise.com",
            age=25,
            preferred_service_tier="enterprise",
        ),
        UserRegistrationRequest(
            name="Premium User2",
            email="u2@premium.com",
            age=30,
            preferred_service_tier="premium",
        ),
    ]

    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Use FlextCore sequence processing
    results = [registration_service.register_user(req) for req in requests]

    # Process results manually since sequence expects object type
    successful_results: list[RegistrationResult] = []
    for result in results:
        if result.is_success:
            successful_results.append(result.value)
        else:
            print(f"❌ Registration failed: {result.error}")

    if successful_results:
        print(f"🎯 Advanced Processing: {len(successful_results)} users processed")

        # Show validation warnings using FlextCore patterns
        for registration_result in successful_results:
            if registration_result.validation_warnings:
                print(
                    f"⚠️ Warnings for {registration_result.user_id}: {registration_result.validation_warnings}"
                )
    else:
        print("❌ No successful registrations")

    # Show container services
    container = get_flext_container()
    services = list(container.list_services().keys())
    print(f"📦 Container Services: {services}")
    return "advanced_patterns_completed"


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    """Minimal boilerplate dependency injection with FlextCore utilities."""
    logger.info("Starting minimal boilerplate dependency injection example")

    print("\n🚀 DEPENDENCY INJECTION - MINIMAL BOILERPLATE WITH FLEXT CORE")
    print("=" * 80)

    # All demos use FlextCore decorators and utilities
    demos = [
        demo_service_injection,
        demo_batch_processing,
        demo_json_processing,
        demo_advanced_patterns,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            # Get function name safely for logging
            demo_name = getattr(demo, "__name__", "unknown_demo")
            logger.exception("Demo failed", demo_name=demo_name, error=str(e))
            print(f"❌ Demo {demo_name} failed: {e}")

    # Show FlextUtilities metrics
    print("\n🚀 FlextUtilities Metrics (Auto-collected)")
    print("=" * 60)
    for key, data in FlextUtilities.iter_metrics_items():
        # Handle the new metrics structure
        if isinstance(data, dict) and "performance" in data:
            perf_data = data["performance"]
            if "duration" in perf_data and "count" in perf_data:
                print(
                    f"  📊 {key}: {perf_data['duration'] * 1000:.2f}ms ({perf_data['count']} calls)"
                )
            else:
                print(f"  📊 {key}: {len(data)} metrics")
        else:
            print(f"  📊 {key}: {len(data) if isinstance(data, dict) else data}")

    print("\n🎓 FlextCore Features Demonstrated:")
    print("  • FlextCore utilities for validation and entity creation")
    print("  • FlextDecorators for performance, logging, and error handling")
    print("  • FlextServiceProcessor templates with enterprise patterns")
    print("  • FlextContainer with dependency injection")
    print("  • FlextLoggerFactory for structured logging")
    print("  • FlextUtilities for automatic metrics collection")
    print("  • FlextResultUtils for batch processing")
    print("  • FlextUtilities for JSON handling")
    print("  • Functional programming patterns with FlextCore")

    logger.info("Minimal boilerplate example completed successfully")


if __name__ == "__main__":
    main()
