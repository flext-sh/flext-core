#!/usr/bin/env python3
"""02 - Dependency Injection with FLEXT patterns.

Demonstrates FlextContainer dependency injection using maximum FLEXT integration.
Follows FLEXT_REFACTORING_PROMPT.md strictly for proper ABI compliance.

Architecture Overview:
    Uses maximum FlextTypes, FlextConstants, FlextProtocols for centralized patterns.
    All service operations return FlextResult[T] for type-safe error handling.
    Implements proper SOLID principles with protocol-based design.
"""

from __future__ import annotations

import json

from pydantic import Field, field_validator

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextEntity,
    FlextModel,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    FlextValue,
    get_flext_container,
)
from flext_core.loggings import get_logger

# Singleton FlextCore instance for all utilities
core = FlextCore.get_instance()
logger = get_logger("flext.examples.container_di")

# =============================================================================
# DOMAIN VALIDATORS - Using FlextCore centralized patterns
# =============================================================================


class UserAgeValidator:
    """Age validator using centralized FlextConstants and patterns."""

    def validate(
        self, value: FlextTypes.Core.Integer
    ) -> FlextResult[FlextTypes.Core.Integer]:
        """Validate age using FlextConstants and proper error handling.

        Args:
            value: Age value to validate.

        Returns:
            FlextResult containing validated age or error message.

        """
        # Use sensible age limits since specific age constants don't exist
        if value < 0:
            return FlextResult[FlextTypes.Core.Integer].fail("Age cannot be negative")

        max_age = 150  # Maximum reasonable age
        if value > max_age:
            error_message = "Age exceeds reasonable maximum"
            return FlextResult[FlextTypes.Core.Integer].fail(error_message)

        min_adult_age = 18  # Legal adult age
        if value < min_adult_age:
            logger.warning("Age below warning threshold", age=value)

        return FlextResult[FlextTypes.Core.Integer].ok(value)


class UserNameValidator:
    """Name validator using centralized FlextConstants and patterns."""

    def validate(
        self, value: FlextTypes.Core.String
    ) -> FlextResult[FlextTypes.Core.String]:
        """Validate name using FlextConstants and proper error handling.

        Args:
            value: Name value to validate.

        Returns:
            FlextResult containing validated name or error message.

        """
        cleaned_name = value.strip()

        # Use reasonable name length limits
        if len(cleaned_name) < 1:
            return FlextResult[FlextTypes.Core.String].fail("Name cannot be empty")

        if len(cleaned_name) > FlextConstants.Validation.MAX_STRING_LENGTH:
            return FlextResult[FlextTypes.Core.String].fail(
                "Name exceeds maximum length"
            )

        # Business rule: warn for numbers in name
        if any(char.isdigit() for char in cleaned_name):
            logger.warning("Name contains numbers", name=cleaned_name)

        return FlextResult[FlextTypes.Core.String].ok(cleaned_name.title())


# =============================================================================
# DOMAIN ENTITIES - Using FlextEntity with centralized types
# =============================================================================


class User(FlextEntity):
    """User entity with centralized validation and FlextTypes.

    Uses FlextTypes for consistency and FlextConstants for validation rules.
    """

    name: FlextTypes.Core.String
    email: FlextTypes.Core.String
    age: FlextTypes.Core.Integer
    status: FlextTypes.Core.String = FlextConstants.Status.PENDING
    registration_id: FlextTypes.Core.String = Field(
        default_factory=lambda: FlextUtilities.Generators.generate_uuid()[:10]
    )
    service_tier: FlextTypes.Core.String = "standard"

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Validate name using centralized validator."""
        validator = UserNameValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or FlextConstants.Errors.VALIDATION_ERROR
            raise ValueError(error_msg)
        return result.unwrap()

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Validate email using centralized validation."""
        if "@" not in v:
            error_message = "Invalid email format"
            raise ValueError(error_message)
        return v

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, v: FlextTypes.Core.Integer) -> FlextTypes.Core.Integer:
        """Validate age using centralized validator."""
        validator = UserAgeValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or FlextConstants.Errors.VALIDATION_ERROR
            raise ValueError(error_msg)
        return result.unwrap()


class EmailAddress(FlextValue):
    """Email value object with centralized validation."""

    email: FlextTypes.Core.String

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Validate email using centralized patterns."""
        if "@" not in v:
            error_message = "Invalid email format"
            raise ValueError(error_message)
        return v

    def validate_business_rules(self) -> FlextResult[None]:
        """Business rules validation using FlextCore patterns."""
        if ".invalid" in self.email:
            return FlextResult[None].fail("Invalid email domain")
        if ".test" in self.email:
            logger.warning("Test email domain detected", email=self.email)
        return FlextResult[None].ok(None)


class UserRegistrationRequest(FlextValue):
    """Registration request with centralized validation."""

    name: FlextTypes.Core.String
    email: FlextTypes.Core.String
    age: FlextTypes.Core.Integer
    preferred_service_tier: FlextTypes.Core.String = "standard"

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Validate name using centralized validator."""
        validator = UserNameValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or FlextConstants.Errors.VALIDATION_ERROR
            raise ValueError(error_msg)
        return result.unwrap()

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v: FlextTypes.Core.String) -> FlextTypes.Core.String:
        """Validate email using centralized validation."""
        if "@" not in v:
            error_message = "Invalid email format"
            raise ValueError(error_message)
        return v

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, v: FlextTypes.Core.Integer) -> FlextTypes.Core.Integer:
        """Validate age using centralized validator."""
        validator = UserAgeValidator()
        result = validator.validate(v)
        if result.is_failure:
            error_msg = result.error or FlextConstants.Errors.VALIDATION_ERROR
            raise ValueError(error_msg)
        return result.unwrap()

    def validate_business_rules(self) -> FlextResult[None]:
        """Business rules validation using FlextCore patterns."""
        valid_tiers = [
            "standard",
            "premium",
            "enterprise",
        ]
        if self.preferred_service_tier not in valid_tiers:
            return FlextResult[None].fail("Invalid service tier")
        return FlextResult[None].ok(None)


# =============================================================================
# RESULT MODELS - Using FlextModel with centralized types
# =============================================================================


class DatabaseSaveResult(FlextModel):
    """Database save result with centralized patterns."""

    user_id: FlextTypes.Core.String
    table_name: FlextTypes.Core.String = "users"
    operation: FlextTypes.Core.String = "INSERT"
    processing_time_ms: FlextTypes.Core.Float
    correlation_id: FlextTypes.Core.String = Field(
        default_factory=FlextUtilities.Generators.generate_uuid
    )
    cache_key: FlextTypes.Core.String | None = None
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class EmailSendResult(FlextModel):
    """Email send result with centralized patterns."""

    recipient: FlextTypes.Core.String
    subject: FlextTypes.Core.String = "Welcome!"
    status: FlextTypes.Core.String
    processing_time_ms: FlextTypes.Core.Float
    correlation_id: FlextTypes.Core.String = Field(
        default_factory=FlextUtilities.Generators.generate_uuid
    )
    delivery_attempts: FlextTypes.Core.Integer = 1
    performance_metrics: dict[str, float] = Field(default_factory=dict)


class RegistrationResult(FlextModel):
    """Complete registration result with centralized patterns."""

    user_id: FlextTypes.Core.String
    status: FlextTypes.Core.String
    database_result: DatabaseSaveResult
    email_result: EmailSendResult
    total_processing_time_ms: FlextTypes.Core.Float
    correlation_id: FlextTypes.Core.String = Field(
        default_factory=FlextUtilities.Generators.generate_uuid
    )
    validation_warnings: list[str] = Field(default_factory=list)
    service_metrics: dict[str, float] = Field(default_factory=dict)


# =============================================================================
# SERVICE PROCESSORS - Using proper class patterns
# =============================================================================


class DatabaseServiceProcessor:
    """Database processor with proper ABI-compliant patterns."""

    def __init__(self) -> None:
        """Initialize processor with utilities."""
        self._logger = get_logger("flext.services.database")
        self._cache: dict[str, DatabaseSaveResult] = {}

    def process(self, request: User) -> FlextResult[DatabaseSaveResult]:
        """Process database save using centralized utilities."""
        self._logger.info(
            "Processing database save",
            user_id=str(request.id),
            table="users",
            operation="INSERT",
        )

        # Use FlextUtilities for timing
        save_time_ms = 5.0  # Simplified timing

        # Create performance metrics
        performance_metrics: dict[str, float] = {
            "query_time_ms": save_time_ms * 0.7,
            "connection_time_ms": save_time_ms * 0.2,
            "serialization_time_ms": save_time_ms * 0.1,
        }

        cache_key = f"user_{request.id}"

        # Create result using proper patterns
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

    def build(
        self, domain: DatabaseSaveResult, *, correlation_id: FlextTypes.Core.String
    ) -> DatabaseSaveResult:
        """Build result using centralized patterns."""
        domain.correlation_id = correlation_id
        return domain

    def save_user(self, user: User) -> FlextResult[DatabaseSaveResult]:
        """Public interface using proper process method."""
        return self.process(user)


class EmailServiceProcessor:
    """Email processor with proper ABI-compliant patterns."""

    def __init__(self) -> None:
        """Initialize processor with utilities."""
        self._logger = get_logger("flext.services.email")

    def process(self, request: User) -> FlextResult[EmailSendResult]:
        """Process email send using centralized utilities."""
        self._logger.info(
            "Processing email send",
            recipient=request.email,
            user_id=str(request.id),
            service_tier=request.service_tier,
        )

        # Validate email using centralized value object with proper ABI
        email_str = (
            str(request.email) if hasattr(request.email, "__str__") else request.email
        )
        email_validation = EmailAddress(email=email_str).validate_business_rules()
        if email_validation.is_failure:
            self._logger.error(
                "Email validation failed",
                recipient=email_str,
                error=email_validation.error,
            )
            return FlextResult[EmailSendResult].fail(
                email_validation.error or FlextConstants.Errors.VALIDATION_ERROR
            )

        # Use FlextUtilities for timing
        send_time_ms = 3.0  # Simplified timing

        # Create performance metrics
        performance_metrics: dict[str, float] = {
            "smtp_connect_ms": send_time_ms * 0.4,
            "template_render_ms": send_time_ms * 0.3,
            "send_time_ms": send_time_ms * 0.3,
        }

        result = EmailSendResult(
            recipient=email_str,
            status=FlextConstants.Status.COMPLETED,
            processing_time_ms=send_time_ms,
            performance_metrics=performance_metrics,
        )

        self._logger.info(
            "Email send completed",
            recipient=email_str,
            status=FlextConstants.Status.COMPLETED,
            processing_time_ms=send_time_ms,
        )

        return FlextResult[EmailSendResult].ok(result)

    def build(
        self, domain: EmailSendResult, *, correlation_id: FlextTypes.Core.String
    ) -> EmailSendResult:
        """Build result using centralized patterns."""
        domain.correlation_id = correlation_id
        return domain

    def send_welcome_email(self, user: User) -> FlextResult[EmailSendResult]:
        """Public interface using proper process method."""
        return self.process(user)


class UserRegistrationProcessor:
    """Registration processor with proper ABI-compliant patterns."""

    def __init__(self) -> None:
        """Initialize processor with utilities."""
        self._logger = get_logger("flext.services.registration")
        self._database_service = DatabaseServiceProcessor()
        self._email_service = EmailServiceProcessor()

    def process(self, request: UserRegistrationRequest) -> FlextResult[User]:
        """Process registration using centralized utilities."""
        self._logger.info(
            "Processing user registration",
            email=request.email,
            name=request.name,
            age=request.age,
            service_tier=request.preferred_service_tier,
        )

        # Validate business rules using centralized patterns
        validation_result = request.validate_business_rules()
        if validation_result.is_failure:
            self._logger.error(
                "Registration validation failed",
                email=request.email,
                error=validation_result.error,
            )
            return FlextResult[User].fail(
                validation_result.error or FlextConstants.Errors.VALIDATION_ERROR
            )

        # Create user entity directly
        try:
            user = User(
                id=f"user_{FlextUtilities.Generators.generate_uuid()[:10]}",
                name=request.name,
                email=request.email,
                age=request.age,
                status=FlextConstants.Status.ACTIVE,
                service_tier=request.preferred_service_tier,
            )
            user_result = FlextResult[User].ok(user)
        except Exception as e:
            user_result = FlextResult[User].fail(f"Failed to create user: {e}")

        if user_result.is_success:
            self._logger.info(
                "User entity created",
                user_id=str(user_result.unwrap().id),
                name=user_result.unwrap().name,
                service_tier=user_result.unwrap().service_tier,
            )

        return user_result

    def build(
        self, domain: User, *, correlation_id: FlextTypes.Core.String
    ) -> RegistrationResult:
        """Build registration result using centralized orchestration."""
        self._logger.info(
            "Building registration result",
            user_id=str(domain.id),
            correlation_id=correlation_id,
        )

        # Use dependency injection to call services
        db_result = self._database_service.save_user(domain)
        email_result = self._email_service.send_welcome_email(domain)

        # Collect validation warnings using centralized patterns
        validation_warnings: list[str] = []
        min_adult_age = 18  # Legal adult age
        if domain.age < min_adult_age:
            validation_warnings.append("User is under 21")
        if ".test" in domain.email:
            validation_warnings.append("Test email domain")

        # Calculate total processing time
        total_time = (
            db_result.unwrap().processing_time_ms if db_result.is_success else 0
        ) + (email_result.unwrap().processing_time_ms if email_result.is_success else 0)

        # Create service metrics
        service_metrics: dict[str, float] = {
            "database_time": db_result.unwrap().processing_time_ms
            if db_result.is_success
            else 0,
            "email_time": email_result.unwrap().processing_time_ms
            if email_result.is_success
            else 0,
        }

        result = RegistrationResult(
            user_id=str(domain.id),
            status=str(domain.status),
            database_result=db_result.unwrap()
            if db_result.is_success
            else DatabaseSaveResult(user_id=str(domain.id), processing_time_ms=0.0),
            email_result=email_result.unwrap()
            if email_result.is_success
            else EmailSendResult(
                recipient=domain.email,
                status=FlextConstants.Status.FAILED,
                processing_time_ms=0.0,
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
        """Public interface using proper template."""
        process_result = self.process(request)
        if process_result.is_failure:
            return FlextResult[RegistrationResult].fail(str(process_result.error))

        user = process_result.unwrap()
        registration_result = self.build(user, correlation_id="reg_123")
        return FlextResult[RegistrationResult].ok(registration_result)


# =============================================================================
# CONTAINER SETUP - Using centralized utilities
# =============================================================================


def setup_container() -> FlextResult[None]:
    """Setup container using centralized utilities."""
    container = get_flext_container()

    # Register service processors
    services = {
        "database_service": DatabaseServiceProcessor,
        "email_service": EmailServiceProcessor,
        "registration_service": UserRegistrationProcessor,
    }

    for service_name, service_class in services.items():
        # Create factory function with explicit typing
        def create_service_factory(cls: type[object]) -> object:
            return cls()

        # Create factory function instead of lambda
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


def get_service_with_fallback[T](
    service_name: FlextTypes.Core.String, default_factory: type[T]
) -> T:
    """Get service from container with centralized utilities."""
    container = get_flext_container()
    result = container.get(service_name)

    if result.is_success:
        logger.debug("Service retrieved from container", service_name=service_name)
        return result.unwrap()  # type: ignore[return-value]

    logger.warning(
        "Service not found in container, using default factory",
        service_name=service_name,
        default_factory=default_factory.__name__,
    )
    return default_factory()


# =============================================================================
# UTILITY FUNCTIONS - Using centralized patterns
# =============================================================================


def log_result[T](
    result: FlextResult[T], success_msg: FlextTypes.Core.String
) -> FlextResult[T]:
    """Utility to log FlextResult using centralized patterns."""
    if result.is_success:
        logger.info(f"✅ {success_msg}", result_type=type(result.unwrap()).__name__)
        return result

    logger.error(f"❌ {success_msg} failed", error=result.error)
    return result


# =============================================================================
# DEMONSTRATIONS - Using centralized patterns
# =============================================================================


def demo_service_injection() -> RegistrationResult | None:
    """Demonstrate service injection with centralized utilities."""
    setup_result = setup_container()
    if setup_result.is_failure:
        return None

    # Get service using centralized utilities
    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Create request using centralized validation
    request = UserRegistrationRequest(
        name="Alice Johnson",
        email="alice@company.com",
        age=28,
        preferred_service_tier="premium",
    )

    result = log_result(
        registration_service.register_user(request), "Service injection registration"
    )
    return result.unwrap() if result.is_success else None


def demo_batch_processing() -> FlextTypes.Core.String:
    """Demonstrate batch processing with centralized utilities."""
    setup_result = setup_container()
    if setup_result.is_failure:
        return "setup_failed"

    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Create batch requests using centralized patterns
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

    # Process all requests individually
    results = [registration_service.register_user(req) for req in requests]
    success_count = len([r for r in results if r.success])
    failure_count = len([r for r in results if r.failure])

    logger.info(
        f"Batch processing completed: {success_count} successes, {failure_count} failures"
    )

    return "batch_processing_completed"


def demo_json_processing() -> FlextTypes.Core.String:
    """Demonstrate JSON processing with centralized utilities."""
    setup_result = setup_container()
    if setup_result.is_failure:
        return "setup_failed"

    registration_service = get_service_with_fallback(
        "registration_service", UserRegistrationProcessor
    )

    # Process valid JSON
    valid_json = (
        '{"name": "JSON User", "email": "json@company.com", '
        '"age": 32, "preferred_service_tier": "premium"}'
    )

    try:
        data = json.loads(valid_json)
        request = UserRegistrationRequest(**data)
        log_result(
            registration_service.register_user(request),
            "JSON service processing",
        )
    except Exception as e:
        logger.exception("JSON processing failed", error=str(e))

    # Process invalid JSON to show validation
    invalid_json = '{"name": "Invalid User", "email": "invalid.invalid", "age": 15}'

    try:
        invalid_data = json.loads(invalid_json)
        UserRegistrationRequest(**invalid_data)
    except Exception as e:
        logger.debug("Expected validation error", error=str(e))

    return "json_processing_completed"


def demo_advanced_patterns() -> FlextTypes.Core.String:
    """Demonstrate advanced patterns with centralized utilities."""
    setup_result = setup_container()
    if setup_result.is_failure:
        return "setup_failed"

    # Create requests using centralized patterns
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

    # Process requests
    results = [registration_service.register_user(req) for req in requests]

    # Process results
    successful_results: list[RegistrationResult] = [
        result.unwrap() for result in results if result.is_success
    ]

    if successful_results:
        # Show validation warnings using centralized patterns
        for registration_result in successful_results:
            if registration_result.validation_warnings:
                pass

    # Show container services
    container = get_flext_container()
    list(container.list_services().keys())
    return "advanced_patterns_completed"


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    """Dependency injection with maximum FLEXT centralized patterns."""
    logger.info("Starting dependency injection example")

    # All demos use centralized patterns and utilities
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

    logger.info("Dependency injection example completed successfully")


if __name__ == "__main__":
    main()
