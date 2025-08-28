#!/usr/bin/env python3
"""FlextMixins composition patterns demonstration.

Shows how to use FlextMixins composition pattern instead of individual mixin inheritance.
Demonstrates centralized behavioral patterns and clean composition over inheritance.

Key Patterns:
• FlextMixins composition over individual mixin inheritance
• Centralized behavioral functionality through a single interface
• Clean separation of concerns with method delegation
• Type-safe operations with FlextResult pattern
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sized
from typing import Protocol, cast

from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)
from shared_example_helpers import run_example_demonstration

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextUtilities,
    FlextMixins,
)

# DIRECT USAGE OF UNIFIED CLASSES - NO ALIASES
# Use FlextMixins.Serializable, FlextMixins.Loggable, etc. directly

# =============================================================================
# PROTOCOL DEFINITIONS - Type protocols for enterprise patterns
# =============================================================================


class UserRepositoryProtocol(Protocol):
    """Protocol for user repository interface."""

    def find_user(self, user_id: str) -> FlextResult[dict[str, object] | None]:
        """Find user by ID."""
        ...

    def save_user(
        self,
        user_id: str,
        user_data: dict[str, object],
    ) -> FlextResult[None]:
        """Save user data."""
        ...


class OrderServiceProtocol(Protocol):
    """Protocol for order service interface."""

    def create_order(
        self,
        user_id: str,
        items: list[dict[str, object]],
    ) -> FlextResult[dict[str, object]]:
        """Create order for user."""
        ...


# =============================================================================
# VALIDATION CONSTANTS - Mixin validation constraints
# =============================================================================

# Username validation constants
MIN_USERNAME_LENGTH = 3  # Minimum characters for username validation

# Content display constants
MAX_CONTENT_PREVIEW_LENGTH = 100  # Maximum characters for content preview

# =============================================================================
# HELPER MIXINS - Additional functionality for demonstrations
# =============================================================================


class SimpleCacheMixin:
    """Simple caching mixin for demonstration purposes."""

    def cache_set(self, key: str, value: object) -> None:
        """Set cache value."""
        if not hasattr(self, "_cache"):
            self._cache: dict[str, object] = {}
        self._cache[key] = value

    def cache_get(self, key: str) -> object:
        """Get cache value."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache.get(key)

    def cache_remove(self, key: str) -> None:
        """Remove cache entry."""
        if hasattr(self, "_cache") and key in self._cache:
            del self._cache[key]


# =============================================================================
# FLEXTMIXINS COMPOSITION PATTERNS - Clean composition over inheritance
# =============================================================================


class TimestampedDocument:
    """Document with automatic timestamp tracking using FlextMixins composition."""

    def __init__(self, title: str, content: str) -> None:
        """Initialize TimestampedDocument.

        Args:
            title: Document title
            content: Document content

        """
        self.title = title
        self.content = content
        # Initialize timestamp functionality
        FlextMixins.create_timestamp_fields(self)

    def update_content(self, new_content: str) -> None:
        """Update document content with timestamp tracking."""
        self.content = new_content
        FlextMixins.update_timestamp(self)

    def get_age_seconds(self) -> float:
        """Get document age in seconds."""
        return FlextMixins.get_age_seconds(self)

    @property
    def created_at(self) -> float:
        """Get creation timestamp."""
        return FlextMixins.get_created_at(self)

    @property
    def updated_at(self) -> float:
        """Get last update timestamp."""
        return FlextMixins.get_updated_at(self)

    def __str__(self) -> str:
        """String representation with timestamps."""
        age = self.get_age_seconds()
        return f"Document('{self.title}', age: {age:.2f}s)"


class IdentifiableUser(SharedUser):
    """Enhanced user with unique identification using FlextMixins composition."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize IdentifiableUser with logging support."""
        super().__init__(*args, **kwargs)
        # Initialize logging functionality
        self._logger = FlextMixins.get_logger(self)

    @property
    def logger(self) -> object:
        """Get logger instance."""
        return FlextMixins.get_logger(self)

    def change_username(self, new_username: str) -> bool:
        """Change username with ID validation and logging.

        Note: Since SharedUser is immutable, this demonstrates the pattern
        but returns a success indicator rather than mutating the instance.
        """
        # Validate ID presence from shared user
        if not getattr(self, "id", None):
            return False

        old_username = self.name

        # Use shared domain copy_with method for immutable updates
        update_result = self.copy_with(name=new_username)
        if update_result.is_failure:
            FlextMixins.log_error(
                self, f"Failed to update username: {update_result.error}"
            )
            return False

        # In a real system, you would return the new instance or update via repository
        FlextMixins.log_info(
            self,
            "Username change validated",
            old_name=old_username,
            new_name=new_username,
        )
        return True

    def __str__(self) -> str:
        """Return string representation with ID."""
        return f"User({self.name}, ID: {self.id}, Email: {self.email_address.email})"


class ValidatableConfiguration:
    """Configuration with validation state tracking using FlextMixins composition."""

    def __init__(self, config_name: str, settings: dict[str, object]) -> None:
        """Initialize ValidatableConfiguration.

        Args:
            config_name: Configuration name
            settings: Configuration settings

        """
        self.config_name = config_name
        self.settings = settings
        # Initialize validation functionality
        FlextMixins.initialize_validation(self)

    def validate_configuration(self) -> bool:
        """Validate configuration settings."""
        FlextMixins.clear_validation_errors(self)

        # Required settings validation
        required_keys = ["database_url", "api_key", "timeout"]
        for key in required_keys:
            if key not in self.settings:
                FlextMixins.add_validation_error(
                    self, f"Missing required setting: {key}"
                )

        # Value validation
        if "timeout" in self.settings:
            timeout = self.settings["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                FlextMixins.add_validation_error(
                    self, "Timeout must be a positive number"
                )

        # Set validation status
        errors = FlextMixins.get_validation_errors(self)
        is_valid = len(errors) == 0
        if is_valid:
            FlextMixins.mark_valid(self)

        return is_valid

    def apply_configuration(self) -> bool:
        """Apply configuration if valid."""
        if not self.validate_configuration():
            for _error in self.validation_errors:
                pass
            return False

        return True

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors."""
        return FlextMixins.get_validation_errors(self)

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return FlextMixins.is_valid(self)

    def __str__(self) -> str:
        """String representation with validation status."""
        status = "Valid" if self.is_valid else "Invalid"
        return f"Config({self.config_name}, {status})"


class LoggableService:
    """Service with structured logging capabilities using FlextMixins composition."""

    def __init__(self, service_name: str) -> None:
        """Initialize LoggableService.

        Args:
            service_name: Service name

        """
        self.service_name = service_name

    @property
    def logger(self) -> object:
        """Get logger instance."""
        return FlextMixins.get_logger(self)

    def process_request(
        self,
        request_id: str,
        _data: dict[str, object],
    ) -> dict[str, object]:
        """Process request with comprehensive logging."""
        FlextMixins.log_info(self, "Processing request", request_id=request_id)

        try:
            # Simulate processing
            time.sleep(0.001)  # Minimal processing time

            result: dict[str, object] = {
                "request_id": request_id,
                "status": "success",
                "processed_at": FlextUtilities.generate_iso_timestamp(),
                "service": self.service_name,
            }

            FlextMixins.log_info(
                self, "Request completed successfully", request_id=request_id
            )
            return result

        except (RuntimeError, ValueError, TypeError) as e:
            FlextMixins.log_error(self, f"Request failed: {e}", request_id=request_id)
            error_result: dict[str, object] = {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
            }
            return error_result

    def health_check(self) -> Mapping[str, object]:
        """Perform health check with logging."""
        FlextMixins.log_debug(self, "Performing health check")

        health_status: dict[str, object] = {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": FlextUtilities.generate_iso_timestamp(),
        }

        FlextMixins.log_info(self, "Health check completed")
        return health_status


class TimedOperation:
    """Operation with execution timing using FlextMixins composition."""

    def __init__(self, operation_name: str) -> None:
        """Initialize TimedOperation.

        Args:
            operation_name: Operation name

        """
        self.operation_name = operation_name

    def execute_operation(self, complexity: int = 1000) -> Mapping[str, object]:
        """Execute operation with timing measurement."""
        FlextMixins.start_timing(self)

        # Simulate operation complexity
        total = 0
        for i in range(complexity):
            total += i * 2

        execution_time = FlextMixins.stop_timing(self)

        return {
            "operation": self.operation_name,
            "result": total,
            "execution_time": execution_time,
            "complexity": complexity,
        }


class CacheableCalculator:
    """Calculator with result caching using FlextMixins composition."""

    def __init__(self) -> None:
        """Initialize CacheableCalculator."""
        self.calculation_count = 0

    def fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number with caching."""
        cache_key = f"fib_{n}"

        # Check cache first
        if FlextMixins.has_cached_value(self, cache_key):
            cached_result = FlextMixins.get_cached_value(self, cache_key)
            if cached_result is not None:
                return (
                    int(cached_result)
                    if isinstance(cached_result, (int, float, str))
                    else 0
                )

        # Calculate if not cached
        self.calculation_count += 1

        result = n if n <= 1 else self.fibonacci(n - 1) + self.fibonacci(n - 2)

        # Cache the result
        FlextMixins.set_cached_value(self, cache_key, result)
        return result

    def get_stats(self) -> Mapping[str, object]:
        """Get calculation statistics."""
        cache_size = 0
        try:
            if hasattr(self, "_cache"):
                cache_size = len(getattr(self, "_cache", {}))
        except Exception:
            cache_size = 0

        return {
            "calculations_performed": self.calculation_count,
            "cache_size": cache_size,
        }


# =============================================================================
# MULTIPLE INHERITANCE PATTERNS - Complex behavioral composition
# =============================================================================


class AdvancedUser:
    """Enhanced user with multiple behavioral mixins using FlextMixins composition."""

    def __init__(self, username: str, email: str, age: int, role: str = "user") -> None:
        """Initialize AdvancedUser using shared domain factory.

        Args:
            username: User username
            email: User email
            age: User age
            role: User role

        """
        # Create user using shared domain factory
        user_result = SharedDomainFactory.create_user(username, email, age)
        if user_result.is_failure:
            msg: str = f"Failed to create user: {user_result.error}"
            raise ValueError(msg)

        # Store user data using composition
        self._user: SharedUser = user_result.value
        self.role = role

        # Initialize FlextMixins functionality
        FlextMixins.create_timestamp_fields(self)
        FlextMixins.initialize_validation(self)

        FlextMixins.log_info(
            self, "Advanced user created", username=username, role=role
        )
        log_domain_operation(
            "advanced_user_created",
            "AdvancedUser",
            str(self._user.id),
            role=role,
        )

    def get_id(self) -> str:
        """Get user ID from composed user."""
        return str(self._user.id)

    @property
    def name(self) -> str:
        """Get user name from composed user."""
        return self._user.name

    @property
    def email_address(self) -> object:
        """Get user email from composed user."""
        return self._user.email_address

    def validate_user(self) -> bool:
        """Comprehensive user validation using shared domain validation."""
        FlextMixins.clear_validation_errors(self)

        # First validate using shared domain rules
        domain_validation = self._user.validate_domain_rules()
        if domain_validation.is_failure:
            FlextMixins.add_validation_error(
                self,
                f"Domain validation failed: {domain_validation.error}",
            )

        # Additional role validation
        valid_roles = ["user", "admin", "moderator"]
        if self.role not in valid_roles:
            FlextMixins.add_validation_error(
                self, f"Invalid role. Must be one of: {valid_roles}"
            )

        validation_errors = FlextMixins.get_validation_errors(self)
        is_valid = len(validation_errors) == 0
        if is_valid:
            FlextMixins.mark_valid(self)

        if is_valid:
            FlextMixins.log_info(self, "User validation successful", username=self.name)
        else:
            FlextMixins.log_info(
                self,
                "User validation failed",
                username=self.name,
                errors=validation_errors,
            )

        return is_valid

    def promote_to_admin(self) -> bool:
        """Promote user to admin with validation and logging."""
        if not self.validate_user():
            FlextMixins.log_error(self, "Cannot promote invalid user to admin")
            return False

        if self.role == "admin":
            FlextMixins.log_info(self, "User is already admin", username=self.name)
            return False

        old_role = self.role
        # Note: In a real system with immutable entities, you would use copy_with
        # and return a new instance or update via repository
        # For this demonstration, we'll simulate the role change
        try:
            # This is a demonstration - in production use proper state management
            self.role = "admin"
            FlextMixins.update_timestamp(self)
        except (RuntimeError, ValueError, TypeError) as e:
            FlextMixins.log_error(self, f"Failed to update role: {e}")
            return False

        FlextMixins.log_info(
            self,
            "User promoted",
            username=self.name,
            old_role=old_role,
            new_role=self.role,
        )
        log_domain_operation(
            "user_promoted",
            "AdvancedUser",
            self.get_id(),
            old_role=old_role,
            new_role=self.role,
        )
        return True

    @property
    def created_at(self) -> float:
        """Get creation timestamp."""
        return FlextMixins.get_created_at(self)

    @property
    def updated_at(self) -> float:
        """Get last update timestamp."""
        return FlextMixins.get_updated_at(self)

    @property
    def is_valid(self) -> bool:
        """Check if user is valid."""
        return FlextMixins.is_valid(self)

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors."""
        return FlextMixins.get_validation_errors(self)

    def get_user_info(self) -> Mapping[str, object]:
        """Get comprehensive user information."""
        created_timestamp = self._user.created_at.root.timestamp()
        age_seconds = time.time() - created_timestamp

        return {
            "id": self.get_id(),
            "username": self.name,
            "email": self._user.email_address.email,
            "age": self._user.age.value,
            "status": (
                self._user.status.value
                if hasattr(self._user.status, "value")
                else str(self._user.status)
            ),
            "role": self.role,
            "created_at": self._user.created_at,
            "updated_at": self.updated_at,
            "age_seconds": age_seconds,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class SmartDocument:
    """Document with smart features through FlextMixins composition."""

    def __init__(self, title: str, content: str, category: str = "general") -> None:
        """Initialize SmartDocument.

        Args:
            title: Document title
            content: Document content
            category: Document category

        """
        self.title = title
        self.content = content
        self.category = category

        # Initialize FlextMixins fields
        FlextMixins.create_timestamp_fields(self)

        # Initialize cache fields
        self._cache: dict[str, object] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.view_count = 0

        # Initialize mixins
        # Timestamps are initialized lazily via property access

    def view_document(self) -> Mapping[str, object]:
        """View document with caching and statistics."""
        cache_key = f"view_{self.title}"

        # Check if view is cached
        cached_view = self._cache.get(cache_key)
        if cached_view is not None:
            self._cache_hits += 1
            self.view_count += 1
            return cached_view if isinstance(cached_view, dict) else {}

        # Generate view data
        view_data = {
            "title": self.title,
            "content": self.content[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
            if len(self.content) > MAX_CONTENT_PREVIEW_LENGTH
            else self.content,
            "category": self.category,
            "created_at": FlextMixins.get_created_at(self),
            "word_count": len(self.content.split()),
            "character_count": len(self.content),
        }

        # Cache the view
        self._cache[cache_key] = view_data
        self._cache_misses += 1
        self.view_count += 1

        return view_data

    def update_content(self, new_content: str) -> None:
        """Update content and clear cache."""
        self.content = new_content
        FlextMixins.update_timestamp(self)

        # Clear cached views
        cache_key = f"view_{self.title}"
        if cache_key in self._cache:
            del self._cache[cache_key]

    def compare_with(self, other: SmartDocument) -> Mapping[str, object]:
        """Compare documents using FlextMixins."""
        my_created = FlextMixins.get_created_at(self)
        other_created = FlextMixins.get_created_at(other)

        return {
            "title_match": self.title == other.title,
            "category_match": self.category == other.category,
            "content_length_diff": len(self.content) - len(other.content),
            "age_diff_seconds": (my_created or 0) - (other_created or 0),
            "view_count_diff": self.view_count - other.view_count,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize document to dictionary using FlextMixins."""
        base_dict = {
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "view_count": self.view_count,
        }
        return FlextMixins.to_dict(self, base_dict)


class EnterpriseService:
    """Enterprise service with comprehensive FlextMixins composition."""

    def __init__(self, service_name: str, config: dict[str, object]) -> None:
        """Initialize EnterpriseService with name and configuration.

        Args:
            service_name: Name of the enterprise service
            config: Service configuration dictionary

        """
        self.service_name = service_name
        self.config = config
        self.request_count = 0
        self.error_count = 0

        # Initialize FlextMixins fields
        FlextMixins.ensure_id(self)
        FlextMixins.initialize_validation(self)

        # Initialize cache fields for simple cache functionality
        self._cache: dict[str, object] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        FlextMixins.log_info(
            self, "Enterprise service initialized", service_name=service_name
        )

    def validate_service(self) -> bool:
        """Validate service configuration using FlextMixins."""
        FlextMixins.clear_validation_errors(self)

        required_config = ["host", "port", "timeout"]
        for key in required_config:
            if key not in self.config:
                FlextMixins.add_validation_error(
                    self, f"Missing required config: {key}"
                )

        if "timeout" in self.config:
            timeout = self.config["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                FlextMixins.add_validation_error(
                    self, "Timeout must be positive number"
                )

        errors = FlextMixins.get_validation_errors(self)
        is_valid = len(errors) == 0
        if is_valid:
            FlextMixins.mark_valid(self)

        return is_valid

    def process_request(
        self,
        request_id: str,
        data: dict[str, object],
    ) -> dict[str, object]:
        """Process request with comprehensive monitoring using FlextMixins."""
        FlextMixins.log_info(self, "Processing request", request_id=request_id)

        # Validate service first
        if not self.validate_service():
            self.error_count += 1
            FlextMixins.log_error(self, "Service validation failed")
            return {"error": "Service not properly configured"}

        # Check cache
        cache_key = f"request_{request_id}"
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            FlextMixins.log_info(self, "Cache hit for request", request_id=request_id)
            return cached_result if isinstance(cached_result, dict) else {}

        # Time the operation
        FlextMixins.start_timing(self)

        try:
            # Simulate processing
            time.sleep(0.001)

            service_id = getattr(self, "id", "unknown")
            result: dict[str, object] = {
                "request_id": request_id,
                "service_id": service_id,
                "status": "success",
                "data": data,
                "processed_at": FlextUtilities.generate_iso_timestamp(),
            }

            execution_time = FlextMixins.stop_timing(self)
            result["execution_time"] = execution_time

            # Cache successful results
            self._cache[cache_key] = result
            self._cache_misses += 1

            self.request_count += 1
            FlextMixins.log_info(self, "Request completed", request_id=request_id)

            return result

        except (RuntimeError, ValueError, TypeError) as e:
            execution_time = FlextMixins.stop_timing(self)
            self.error_count += 1
            FlextMixins.log_exception(self, "Request failed", request_id=request_id)

            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
            }

    def get_service_metrics(self) -> Mapping[str, object]:
        """Get comprehensive service metrics using FlextMixins."""
        service_id = getattr(self, "id", "unknown")
        return {
            "service_id": service_id,
            "service_name": self.service_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "is_valid": FlextMixins.is_valid(self),
            "validation_errors": FlextMixins.get_validation_errors(self),
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


# =============================================================================
# COMPOSITE MIXIN PATTERNS - Pre-built architectural patterns
# =============================================================================


class DomainEntity(FlextEntityMixin):
    """Domain entity using composite entity mixin."""

    def __init__(self, entity_type: str, data: dict[str, object]) -> None:
        """Initialize DomainEntity with type and data.

        Args:
            entity_type: Type classification of the entity
            data: Entity data dictionary

        """
        super().__init__()
        self.entity_type = entity_type
        self.value = data
        self._domain_events: list[tuple[str, dict[str, object]]] = []

    def clear_domain_events(self) -> None:
        """Clear collected domain events."""
        self._domain_events.clear()

    def get_domain_events(self) -> list[object]:
        """Get collected domain events."""
        return list(self._domain_events)

    def update_data(self, new_data: dict[str, object]) -> None:
        """Update entity data with timestamp tracking."""
        self.value.update(new_data)
        self.update_timestamp()

    def get_entity_info(self) -> Mapping[str, object]:
        """Get comprehensive entity information."""
        return {
            "id": self.get_id(),
            "type": self.entity_type,
            "data": self.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_valid_id": self.has_id(),
        }


class ValueObjectExample(FlextValueObjectMixin):
    """Value object using composite value object mixin."""

    def __init__(self, name: str, value: object, unit: str | None = None) -> None:
        """Initialize ValueObjectExample with name, value and optional unit.

        Args:
            name: Name of the value object
            value: The contained value
            unit: Optional unit of measurement

        """
        super().__init__()
        self.name = name
        self.value = value
        self.unit = unit
        # Validation state is initialized lazily via method calls

    def validate_value(self) -> bool:
        """Validate value object."""
        self.clear_validation_errors()

        if not self.name:
            self.add_validation_error("Name cannot be empty")

        if self.value is None:
            self.add_validation_error("Value cannot be None")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            FlextMixins.mark_valid(self)

        return is_valid

    def to_dict(self) -> dict[str, object]:
        """Serialize value object."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "is_valid": self.is_valid,
        }

    def __str__(self) -> str:
        """Represent string representation."""
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit_str}"


# =============================================================================
# DEMONSTRATION FUNCTIONS - Comprehensive mixin showcases
# =============================================================================


def demonstrate_individual_mixins() -> None:
    """Demonstrate individual mixin patterns."""
    # Timestamp mixin
    doc = TimestampedDocument("Test Document", "Initial content")
    time.sleep(0.1)
    doc.update_content("Updated content")

    # Identifiable mixin
    # Create user using shared domain factory first
    user_result = SharedDomainFactory.create_user("john_doe", "john@example.com", 25)
    if user_result.success:
        shared_user = user_result.value
        user = IdentifiableUser(
            id=shared_user.id,
            name=shared_user.name,
            email_address=shared_user.email_address,
            age=shared_user.age,
            status=shared_user.status,
            phone=shared_user.phone,
            address=shared_user.address,
            version=shared_user.version,
            created_at=shared_user.created_at,
        )
        user.change_username("john_smith")

    # Validatable mixin
    config = ValidatableConfiguration(
        "database",
        {"database_url": "localhost", "timeout": 30},
    )
    config.apply_configuration()

    invalid_config = ValidatableConfiguration("invalid", {"timeout": -1})
    invalid_config.apply_configuration()

    # Loggable mixin
    service = LoggableService("user_service")
    service.process_request("req_001", {"action": "create_user"})

    # Timing mixin
    operation = TimedOperation("data_processing")
    operation.execute_operation(500)

    # Cacheable mixin
    calculator = CacheableCalculator()

    # First calculation (no cache)
    calculator.fibonacci(10)

    # Second calculation (cached)
    calculator.fibonacci(10)

    calculator.get_stats()


def demonstrate_multiple_inheritance() -> None:
    """Demonstrate multiple inheritance patterns."""
    # Advanced user with multiple mixins
    user = AdvancedUser("alice_admin", "alice@company.com", 28, "user")

    # Validate user
    user.validate_user()

    # Promote user
    user.promote_to_admin()

    # Get user info
    user.get_user_info()

    # Smart document with multiple mixins
    doc1 = SmartDocument(
        "AI Research",
        "Artificial Intelligence is transforming technology...",
        "research",
    )
    doc2 = SmartDocument(
        "ML Guide",
        "Machine Learning provides powerful algorithms...",
        "research",
    )

    # View documents (caching demo)
    doc1.view_document()

    doc1.view_document()  # Should hit cache

    # Compare documents
    doc1.compare_with(doc2)

    # Update and view again (cache cleared)
    doc1.update_content("Updated AI research content with more details...")
    doc1.view_document()

    # Enterprise service with all mixins
    service = EnterpriseService(
        "payment_service",
        {
            "host": FlextConstants.Platform.DEFAULT_HOST,
            "port": FlextConstants.Platform.FLEXCORE_PORT,
            "timeout": FlextConstants.DEFAULT_TIMEOUT,
        },
    )

    # Process multiple requests
    for i in range(3):
        request_id = f"req_{i:03d}"
        service.process_request(request_id, {"amount": 100 + i * 10})

    # Get service metrics
    service.get_service_metrics()


def demonstrate_composite_mixins() -> None:
    """Demonstrate composite mixin patterns."""
    # Domain entity using FlextEntityMixin
    entity = DomainEntity(
        "Customer",
        {
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active",
        },
    )

    entity.get_entity_info()

    entity.update_data(
        {"status": "premium", "last_login": FlextUtilities.generate_iso_timestamp()},
    )
    updated_info = entity.get_entity_info()
    data_field = updated_info["data"]
    (len(cast("Sized", data_field)) if hasattr(data_field, "__len__") else 0)

    # Value object using FlextValueObjectMixin

    # Valid value object
    price = ValueObjectExample("Price", 99.99, "USD")
    price.validate_value()

    # Invalid value object
    invalid_value = ValueObjectExample("", None)
    invalid_value.validate_value()

    # Serialization demo
    price.to_dict()


def demonstrate_method_resolution_order() -> None:
    """Demonstrate method resolution order in multiple inheritance."""

    class ComplexClass(
        FlextMixins.Loggable,
        FlextTimingMixin,
        FlextValidatableMixin,
        FlextIdentifiableMixin,
    ):
        """Class with complex multiple inheritance."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            # ID is auto-generated when accessing self.id property
            # Validation state is initialized lazily via method calls
            self.logger.info("Complex class initialized", name=name)

        def perform_operation(self) -> Mapping[str, object]:
            """Operation using multiple mixin capabilities."""
            self.start_timing()

            # Validate state
            self.clear_validation_errors()
            if not self.name:
                self.add_validation_error("Name cannot be empty")

            is_valid = len(self.validation_errors) == 0
            if is_valid:
                FlextMixins.mark_valid(self)

            # Simulate work
            time.sleep(0.001)

            execution_time = self.stop_timing()

            result = {
                "id": self.get_id(),
                "name": self.name,
                "is_valid": is_valid,
                "execution_time": execution_time,
                "mro_length": len(self.__class__.__mro__),
            }

            self.logger.info("Operation completed", name=self.name)
            return result

    # Demonstrate MRO
    complex_obj = ComplexClass("test_object")
    for _i, _cls in enumerate(complex_obj.__class__.__mro__):
        pass

    # Perform operation
    complex_obj.perform_operation()


def demonstrate_performance_characteristics() -> None:
    """Demonstrate performance characteristics of mixins."""

    # Simple class without mixins
    class SimpleClass:
        def __init__(self, name: str) -> None:
            self.name = name

        def operation(self) -> str:
            return f"Simple: {self.name}"

    # Class with single mixin
    class SingleMixinClass(FlextTimestampMixin):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            # Timestamps are initialized lazily via property access

        def operation(self) -> str:
            return f"Single: {self.name}"

    # Class with multiple mixins
    class MultipleMixinClass(
        FlextIdentifiableMixin,
        FlextTimestampMixin,
        FlextMixins.Loggable,
        FlextValidatableMixin,
    ):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            # ID is auto-generated when accessing self.id property
            # Timestamps are initialized lazily via property access
            # Validation state is initialized lazily via method calls

        def operation(self) -> str:
            return f"Multiple: {self.name}"

    # Performance test
    operations = 1000

    # Test simple class
    start_time = time.time()
    for i in range(operations):
        obj = SimpleClass(f"test_{i}")
        obj.operation()
    simple_time = time.time() - start_time

    # Test single mixin class
    start_time = time.time()
    for i in range(operations):
        single_obj = SingleMixinClass(f"test_{i}")
        single_obj.operation()
    single_time = time.time() - start_time

    # Test multiple mixin class
    start_time = time.time()
    for i in range(operations):
        multi_obj = MultipleMixinClass(f"test_{i}")
        multi_obj.operation()
    multiple_time = time.time() - start_time

    # Calculate overhead
    ((single_time - simple_time) / simple_time) * 100
    ((multiple_time - simple_time) / simple_time) * 100


def demonstrate_enterprise_patterns() -> None:
    """Demonstrate enterprise patterns with mixins using railway-oriented programming.

    Refactored from complexity 37 to follow Single Responsibility Principle.
    Each specialized demonstration method handles a specific pattern.
    """
    # Create repository and service instances
    user_repo = _create_enterprise_user_repository()
    order_service = _create_enterprise_order_service(user_repo)

    # Demonstrate each pattern separately
    _demonstrate_repository_pattern(user_repo)
    _demonstrate_service_pattern(order_service)


def _create_enterprise_user_repository() -> UserRepositoryProtocol:
    """Create UserRepository for enterprise patterns demonstration."""

    # Repository pattern with mixins using FlextResult pattern
    class UserRepository(
        FlextMixins.Loggable,
        SimpleCacheMixin,  # For cache methods
        FlextTimingMixin,
    ):
        """Repository with enterprise mixin composition using railway-oriented programming."""

        def __init__(self) -> None:
            super().__init__()
            self.users: dict[str, dict[str, object]] = {}

        def _execute_save_operation(
            self,
            user_id: str,
            user_data: dict[str, object],
        ) -> FlextResult[None]:
            """Execute the actual save operation with error handling."""
            try:
                self.users[user_id] = user_data
                self.cache_set(f"user_{user_id}", user_data)
                return FlextResult.ok(None)
            except (RuntimeError, ValueError, TypeError) as e:
                return FlextResult.fail(f"Failed to save user data: {e}")

        def _log_save_result(
            self,
            user_id: str,
            _start_time: float,
            result: FlextResult[str],
        ) -> FlextResult[None]:
            """Log the save operation result with execution time."""
            _ = self.stop_timing()
            if result.is_success:
                self.logger.info("User saved", user_id=user_id)
                return FlextResult[None].ok(None)
            self.logger.error(
                "Failed to save user",
                user_id=user_id,
                error=result.error,
            )
            return FlextResult[None].fail(result.error or "Save failed")

        def save_user(
            self,
            user_id: str,
            user_data: dict[str, object],
        ) -> FlextResult[None]:
            """Save user with caching and logging using railway-oriented programming."""
            start_time = self.start_timing()

            return self._execute_save_operation(user_id, user_data).flat_map(
                lambda _: self._log_save_result(
                    user_id,
                    start_time,
                    FlextResult.ok("Save completed successfully"),
                ),
            )

        def _check_cache_for_user(
            self,
            user_id: str,
        ) -> FlextResult[dict[str, object] | None]:
            """Check cache for user data."""
            cache_key = f"user:{user_id}"
            cached_user = self.cache_get(cache_key)
            if cached_user is not None:
                self.logger.info("Cache hit for user", user_id=user_id)
                user_dict = cached_user if isinstance(cached_user, dict) else None
                return FlextResult.ok(user_dict)
            return FlextResult.ok(None)

        def _check_storage_for_user(
            self,
            user_id: str,
        ) -> FlextResult[dict[str, object] | None]:
            """Check storage for user data and update cache if found."""
            if user_id in self.users:
                user_data = self.users[user_id]
                cache_key = f"user:{user_id}"
                self.cache_set(cache_key, user_data)
                self.logger.info("User found in storage", user_id=user_id)
                return FlextResult.ok(user_data)
            return FlextResult.ok(None)

        def _handle_user_not_found(
            self,
            user_id: str,
        ) -> FlextResult[dict[str, object] | None]:
            """Handle case when user is not found."""
            self.logger.warning("User not found", user_id=user_id)
            return FlextResult.ok(None)

        def find_user(self, user_id: str) -> FlextResult[dict[str, object] | None]:
            """Find user with caching and logging using railway-oriented programming."""
            self.start_timing()

            # Try cache first, then storage, then handle not found
            cache_result = self._check_cache_for_user(user_id)
            if cache_result.is_success and cache_result.value is not None:
                return cache_result

            storage_result = self._check_storage_for_user(user_id)
            if storage_result.is_success and storage_result.value is not None:
                return storage_result

            return self._handle_user_not_found(user_id)

    return UserRepository()


def _create_enterprise_order_service(
    user_repo: UserRepositoryProtocol,
) -> OrderServiceProtocol:
    """Create OrderService for enterprise patterns demonstration."""

    # Domain service pattern with railway-oriented programming
    class OrderService(
        FlextIdentifiableMixin,
        FlextMixins.Loggable,
        FlextValidatableMixin,
        FlextTimingMixin,
    ):
        """Order service with comprehensive mixins using railway-oriented programming."""

        def __init__(self, user_repo: UserRepositoryProtocol) -> None:
            super().__init__()
            self.user_repo = user_repo
            self.orders: dict[str, dict[str, object]] = {}
            # ID is auto-generated when accessing self.id property
            # Validation state is initialized lazily via method calls

        def _validate_user_exists(self, user_id: str) -> FlextResult[dict[str, object]]:
            """Validate that user exists in repository."""
            user_result = self.user_repo.find_user(user_id)
            if user_result.is_failure:
                return FlextResult.fail(
                    f"Failed to check user existence: {user_result.error}",
                )

            user = user_result.value
            if not user:
                self.logger.error(
                    "Cannot create order: User not found",
                    user_id=user_id,
                )
                return FlextResult.fail(f"User not found: {user_id}")

            return FlextResult.ok(user)

        def _validate_order_items(
            self,
            items: list[dict[str, object]],
        ) -> FlextResult[None]:
            """Validate order items format and content."""
            self.clear_validation_errors()

            if not items:
                self.add_validation_error("Order must have at least one item")

            for item in items:
                if "product_id" not in item or "quantity" not in item:
                    self.add_validation_error("Invalid item format")

            if not self.is_valid:
                self.logger.error(
                    "Order validation failed",
                    errors=self.validation_errors,
                )
                return FlextResult.fail(
                    f"Order validation failed: {'; '.join(self.validation_errors)}",
                )

            return FlextResult.ok(None)

        def _create_order_entity(
            self,
            user_id: str,
            items: list[dict[str, object]],
        ) -> FlextResult[dict[str, object]]:
            """Create the order entity with all required fields."""
            try:
                order_id = FlextUtilities.generate_entity_id()
                order: dict[str, object] = {
                    "order_id": order_id,
                    "user_id": user_id,
                    "items": items,
                    "status": "pending",
                    "created_at": FlextUtilities.generate_iso_timestamp(),
                }
                self.orders[order_id] = order
                return FlextResult.ok(order)
            except (RuntimeError, ValueError, TypeError) as e:
                return FlextResult.fail(f"Failed to create order entity: {e}")

        def _log_order_creation(
            self,
            _start_time: float,
            order: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Log successful order creation with execution time."""
            _ = self.stop_timing()
            self.logger.info("Order created", order_id=order["order_id"])
            return FlextResult.ok(order)

        def create_order(
            self,
            user_id: str,
            items: list[dict[str, object]],
        ) -> FlextResult[dict[str, object]]:
            """Create order with validation and logging using railway-oriented programming."""
            start_time = self.start_timing()
            self.logger.info("Creating order for user", user_id=user_id)

            return (
                self._validate_user_exists(user_id)
                .flat_map(lambda _: self._validate_order_items(items))
                .flat_map(lambda _: self._create_order_entity(user_id, items))
                .flat_map(lambda order: self._log_order_creation(start_time, order))
            )

    return OrderService(user_repo)


def _demonstrate_repository_pattern(user_repo: UserRepositoryProtocol) -> None:
    """Demonstrate repository pattern with FlextResult."""
    # Save users with FlextResult pattern
    user_repo.save_user(
        "user_001",
        {"name": "Alice", "email": "alice@example.com"},
    )
    user_repo.save_user(
        "user_002",
        {"name": "Bob", "email": "bob@example.com"},
    )

    # Find users (cache demo) with FlextResult pattern
    user1_result = user_repo.find_user("user_001")  # From storage
    user_repo.find_user("user_001")  # From cache

    if user1_result.is_success and user1_result.value:
        pass


def _demonstrate_service_pattern(order_service: OrderServiceProtocol) -> None:
    """Demonstrate service pattern with FlextResult."""
    # Create valid order with FlextResult pattern
    order_result = order_service.create_order(
        "user_001",
        [
            {"product_id": "prod_001", "quantity": 2},
            {"product_id": "prod_002", "quantity": 1},
        ],
    )

    if order_result.is_success and order_result.value:
        order = order_result.value
        items = order["items"]
        len(cast("Sized", items)) if hasattr(items, "__len__") else 0

    # Try invalid order with FlextResult pattern
    invalid_order_result = order_service.create_order("user_999", [])
    if invalid_order_result.is_failure:
        pass


def main() -> None:
    """Run comprehensive FlextMixins demonstration."""
    examples = [
        ("Individual Mixin Patterns", demonstrate_individual_mixins),
        ("Multiple Inheritance Composition", demonstrate_multiple_inheritance),
        ("Composite Mixin Patterns", demonstrate_composite_mixins),
        ("Method Resolution Order", demonstrate_method_resolution_order),
        ("Performance Characteristics", demonstrate_performance_characteristics),
        ("Enterprise Architecture Patterns", demonstrate_enterprise_patterns),
    ]

    run_example_demonstration(
        "🔧 FLEXT MIXINS - MULTIPLE INHERITANCE & COMPOSITION PATTERNS DEMONSTRATION",
        examples,
    )


if __name__ == "__main__":
    main()
