#!/usr/bin/env python3
"""Multiple inheritance and composition patterns with mixins.

Demonstrates cross-cutting concerns integration, method resolution order,
and enterprise service patterns using mixin combinations.
"""

from __future__ import annotations

import time
from typing import Protocol

from flext_core import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextConstants,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextResult,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextUtilities,
    FlextValidatableMixin,
)
from flext_core.mixins import FlextEntityMixin, FlextValueObjectMixin

from .shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)
from .shared_example_helpers import run_example_demonstration

# =============================================================================
# PROTOCOL DEFINITIONS - Type protocols for enterprise patterns
# =============================================================================


class UserRepositoryProtocol(Protocol):
    """Protocol for user repository interface."""

    def find_user(self, user_id: str) -> FlextResult[dict[str, object] | None]:
        """Find user by ID."""
        ...

    def save_user(
        self, user_id: str, user_data: dict[str, object]
    ) -> FlextResult[None]:
        """Save user data."""
        ...


class OrderServiceProtocol(Protocol):
    """Protocol for order service interface."""

    def create_order(
        self, user_id: str, items: list[dict[str, object]]
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
# INDIVIDUAL MIXIN DEMONSTRATIONS - Single responsibility patterns
# =============================================================================


class TimestampedDocument(FlextTimestampMixin):
    """Document with automatic timestamp tracking."""

    def __init__(self, title: str, content: str) -> None:
        """Initialize TimestampedDocument.

        Args:
            title: Document title
            content: Document content

        """
        super().__init__()
        self.title = title
        self.content = content
        # Timestamps are initialized lazily via property access

    def update_content(self, new_content: str) -> None:
        """Update document content with timestamp tracking."""
        self.content = new_content
        self._update_timestamp()
        print(f"📝 Document '{self.title}' updated at {self.updated_at}")

    def get_age_seconds(self) -> float:
        """Get document age in seconds."""
        if self.created_at:
            return time.time() - self.created_at
        return 0.0

    def __str__(self) -> str:
        """Strings representation with timestamps."""
        age = self.get_age_seconds()
        return f"Document('{self.title}', age: {age:.2f}s)"


class IdentifiableUser(SharedUser, FlextIdentifiableMixin, FlextLoggableMixin):
    """Enhanced user with unique identification using shared domain models."""

    def change_username(self, new_username: str) -> bool:
        """Change username with ID validation and logging.

        Note: Since SharedUser is immutable, this demonstrates the pattern
        but returns a success indicator rather than mutating the instance.
        """
        if not self.has_id():
            print("❌ Cannot change username: User has no valid ID")
            return False

        old_username = self.name

        # Use shared domain copy_with method for immutable updates
        update_result = self.copy_with(name=new_username)
        if update_result.is_failure:
            self.logger.error("Failed to update username", error=update_result.error)
            return False

        # In a real system, you would return the new instance or update via repository
        self.logger.info(
            "Username change validated",
            old_name=old_username,
            new_name=new_username,
        )
        print(f"👤 User {self.id}: '{old_username}' → '{new_username}' (validated)")
        print("   Note: SharedUser is immutable - use copy_with() for actual updates")
        return True

    def __str__(self) -> str:
        """Return string representation with ID."""
        return f"User({self.name}, ID: {self.id}, Email: {self.email_address.email})"


class ValidatableConfiguration(FlextValidatableMixin):
    """Configuration with validation state tracking."""

    def __init__(self, config_name: str, settings: dict[str, object]) -> None:
        """Initialize ValidatableConfiguration.

        Args:
            config_name: Configuration name
            settings: Configuration settings

        """
        super().__init__()
        self.config_name = config_name
        self.settings = settings
        # Validation state is initialized lazily via method calls

    def validate_configuration(self) -> bool:
        """Validate configuration settings."""
        self.clear_validation_errors()

        # Required settings validation
        required_keys = ["database_url", "api_key", "timeout"]
        for key in required_keys:
            if key not in self.settings:
                self.add_validation_error(f"Missing required setting: {key}")

        # Value validation
        if "timeout" in self.settings:
            timeout = self.settings["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                self.add_validation_error("Timeout must be a positive number")

        # Set validation status
        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self.mark_valid()

        return is_valid

    def apply_configuration(self) -> bool:
        """Apply configuration if valid."""
        if not self.validate_configuration():
            print(f"❌ Configuration '{self.config_name}' validation failed:")
            for error in self.validation_errors:
                print(f"   - {error}")
            return False

        print(f"✅ Configuration '{self.config_name}' applied successfully")
        return True

    def __str__(self) -> str:
        """Strings representation with validation status."""
        status = "Valid" if self.is_valid else "Invalid"
        return f"Config({self.config_name}, {status})"


class LoggableService(FlextLoggableMixin):
    """Service with structured logging capabilities."""

    def __init__(self, service_name: str) -> None:
        """Initialize LoggableService.

        Args:
            service_name: Service name

        """
        super().__init__()
        self.service_name = service_name

    def process_request(
        self,
        request_id: str,
        _data: dict[str, object],
    ) -> dict[str, object]:
        """Process request with comprehensive logging."""
        self.logger.info("Processing request", request_id=request_id)

        try:
            # Simulate processing
            time.sleep(0.001)  # Minimal processing time

            result: dict[str, object] = {
                "request_id": request_id,
                "status": "success",
                "processed_at": FlextUtilities.generate_iso_timestamp(),
                "service": self.service_name,
            }

            self.logger.info("Request completed successfully", request_id=request_id)
            return result

        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.exception("Request failed", request_id=request_id)
            error_result: dict[str, object] = {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
            }
            return error_result

    def health_check(self) -> dict[str, object]:
        """Perform health check with logging."""
        self.logger.debug("Performing health check")

        health_status: dict[str, object] = {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": FlextUtilities.generate_iso_timestamp(),
        }

        self.logger.info("Health check completed")
        return health_status


class TimedOperation(FlextTimingMixin):
    """Operation with execution timing."""

    def __init__(self, operation_name: str) -> None:
        """Initialize TimedOperation.

        Args:
            operation_name: Operation name

        """
        super().__init__()
        self.operation_name = operation_name

    def _get_execution_time_seconds(self, start_time: float) -> float:
        """Convert execution time from milliseconds to seconds."""
        return self._get_execution_time_ms(start_time) / 1000.0

    def execute_operation(self, complexity: int = 1000) -> dict[str, object]:
        """Execute operation with timing measurement."""
        print(f"⏱️ Starting operation: {self.operation_name}")

        start_time = self._start_timing()

        # Simulate operation complexity
        total = 0
        for i in range(complexity):
            total += i * 2

        execution_time = self._get_execution_time_seconds(start_time)

        result = {
            "operation": self.operation_name,
            "result": total,
            "execution_time": execution_time,
            "complexity": complexity,
        }

        print(f"✅ Operation completed in {execution_time:.4f}s")
        return result


class CacheableCalculator(FlextCacheableMixin, SimpleCacheMixin):
    """Calculator with result caching using SimpleCacheMixin for cache methods."""

    def __init__(self) -> None:
        """Initialize CacheableCalculator."""
        super().__init__()
        self.calculation_count = 0

    def fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number with caching."""
        cache_key = f"fib_{n}"

        # Check cache first
        cached_result = self.cache_get(cache_key)
        if cached_result is not None:
            cached_int = (
                int(cached_result)
                if isinstance(cached_result, (int, float, str))
                else 0
            )
            print(f"💰 Cache hit for fib({n}): {cached_int}")
            return cached_int

        # Calculate if not cached
        print(f"🔢 Calculating fib({n})")
        self.calculation_count += 1

        result = n if n <= 1 else self.fibonacci(n - 1) + self.fibonacci(n - 2)

        # Cache the result
        self.cache_set(cache_key, result)
        return result

    def get_stats(self) -> dict[str, object]:
        """Get calculation statistics."""
        return {
            "calculations_performed": self.calculation_count,
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
        }


# =============================================================================
# MULTIPLE INHERITANCE PATTERNS - Complex behavioral composition
# =============================================================================


class AdvancedUser(
    FlextTimestampMixin,
    FlextValidatableMixin,
    FlextLoggableMixin,
):
    """Enhanced user with multiple behavioral mixins using shared domain models."""

    def __init__(self, username: str, email: str, age: int, role: str = "user") -> None:
        """Initialize AdvancedUser using shared domain factory.

        Args:
            username: User username
            email: User email
            age: User age
            role: User role

        """
        # Initialize mixins first
        super().__init__()

        # Create user using shared domain factory
        user_result = SharedDomainFactory.create_user(username, email, age)
        if user_result.is_failure:
            msg: str = f"Failed to create user: {user_result.error}"
            raise ValueError(msg)

        shared_user = user_result.data
        if shared_user is None:
            error_msg = "User creation returned None data"
            raise ValueError(error_msg)

        # Store user data using composition
        self._user: SharedUser = shared_user
        self.role = role

        # Initialize mixins
        # Timestamps are initialized lazily via property access
        # Validation state is initialized lazily via method calls

        self.logger.info("Advanced user created", username=username, role=role)
        log_domain_operation(
            "advanced_user_created",
            "AdvancedUser",
            self._user.id,
            role=role,
        )

    @property
    def id(self) -> str:
        """Get user ID from composed user."""
        return self._user.id

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
        self.clear_validation_errors()

        # First validate using shared domain rules
        domain_validation = self._user.validate_domain_rules()
        if domain_validation.is_failure:
            self.add_validation_error(
                f"Domain validation failed: {domain_validation.error}",
            )

        # Additional role validation
        valid_roles = ["user", "admin", "moderator"]
        if self.role not in valid_roles:
            self.add_validation_error(f"Invalid role. Must be one of: {valid_roles}")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self.mark_valid()

        if is_valid:
            self.logger.info("User validation successful", username=self.name)
        else:
            self.logger.warning(
                "User validation failed",
                username=self.name,
                errors=self.validation_errors,
            )

        return is_valid

    def promote_to_admin(self) -> bool:
        """Promote user to admin with validation and logging."""
        if not self.validate_user():
            self.logger.error("Cannot promote invalid user to admin")
            return False

        if self.role == "admin":
            self.logger.warning("User is already admin", username=self.name)
            return False

        old_role = self.role
        # Note: In a real system with immutable entities, you would use copy_with
        # and return a new instance or update via repository
        # For this demonstration, we'll simulate the role change
        try:
            # This is a demonstration - in production use proper state management
            self.role = "admin"
            self._update_timestamp()
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.exception("Failed to update role", error=str(e))
            return False

        self.logger.info(
            "User promoted",
            username=self.name,
            old_role=old_role,
            new_role=self.role,
        )
        log_domain_operation(
            "user_promoted",
            "AdvancedUser",
            self.id,
            old_role=old_role,
            new_role=self.role,
        )
        return True

    def get_user_info(self) -> dict[str, object]:
        """Get comprehensive user information."""
        created_timestamp = self._user.created_at.timestamp()
        age_seconds = time.time() - created_timestamp

        return {
            "id": self.id,
            "username": self.name,
            "email": self._user.email_address.email,
            "age": self._user.age.value,
            "status": self._user.status.value,
            "role": self.role,
            "created_at": self._user.created_at,
            "updated_at": self.updated_at,
            "age_seconds": age_seconds,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class SmartDocument(
    FlextTimestampMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
    FlextCacheableMixin,
    SimpleCacheMixin,
):
    """Document with smart features through mixin composition."""

    def __init__(self, title: str, content: str, category: str = "general") -> None:
        """Initialize SmartDocument.

        Args:
            title: Document title
            content: Document content
            category: Document category

        """
        super().__init__()
        self.title = title
        self.content = content
        self.category = category
        self.view_count = 0

        # Initialize mixins
        # Timestamps are initialized lazily via property access

    def view_document(self) -> dict[str, object]:
        """View document with caching and statistics."""
        cache_key = f"view_{self.title}"

        # Check if view is cached
        cached_view = self.cache_get(cache_key)
        if cached_view is not None:
            print(f"📄 Cached view for: {self.title}")
            self.view_count += 1
            return cached_view if isinstance(cached_view, dict) else {}

        # Generate view data
        view_data = {
            "title": self.title,
            "content": self.content[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
            if len(self.content) > MAX_CONTENT_PREVIEW_LENGTH
            else self.content,
            "category": self.category,
            "created_at": self.created_at,
            "word_count": len(self.content.split()),
            "character_count": len(self.content),
        }

        # Cache the view
        self.cache_set(cache_key, view_data)
        self.view_count += 1

        print(f"📄 Generated view for: {self.title}")
        return view_data

    def update_content(self, new_content: str) -> None:
        """Update content and clear cache."""
        self.content = new_content
        self._update_timestamp()

        # Clear cached views
        cache_key = f"view_{self.title}"
        cached_view = self.cache_get(cache_key)
        if cached_view is not None:
            self.cache_remove(cache_key)

        print(f"📝 Content updated for: {self.title}")

    def compare_with(self, other: SmartDocument) -> dict[str, object]:
        """Compare documents using comparable mixin."""
        return {
            "title_match": self.title == other.title,
            "category_match": self.category == other.category,
            "content_length_diff": len(self.content) - len(other.content),
            "age_diff_seconds": (self.created_at or 0) - (other.created_at or 0),
            "view_count_diff": self.view_count - other.view_count,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize document to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "view_count": self.view_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class EnterpriseService(
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    SimpleCacheMixin,  # For cache methods
):
    """Enterprise service with comprehensive mixin composition."""

    def __init__(self, service_name: str, config: dict[str, object]) -> None:
        """Initialize EnterpriseService with name and configuration.

        Args:
            service_name: Name of the enterprise service
            config: Service configuration dictionary

        """
        super().__init__()
        self.service_name = service_name
        self.config = config
        self.request_count = 0
        self.error_count = 0

        # Initialize all mixins
        self.generate_id()
        # Validation state is initialized lazily via method calls

        self.logger.info("Enterprise service initialized", service_name=service_name)

    def _get_execution_time_seconds(self, start_time: float) -> float:
        """Convert execution time from milliseconds to seconds."""
        return self._get_execution_time_ms(start_time) / 1000.0

    def validate_service(self) -> bool:
        """Validate service configuration."""
        self.clear_validation_errors()

        required_config = ["host", "port", "timeout"]
        for key in required_config:
            if key not in self.config:
                self.add_validation_error(f"Missing required config: {key}")

        if "timeout" in self.config:
            timeout = self.config["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                self.add_validation_error("Timeout must be positive number")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self.mark_valid()

        return is_valid

    def process_request(
        self,
        request_id: str,
        data: dict[str, object],
    ) -> dict[str, object]:
        """Process request with comprehensive monitoring."""
        self.logger.info("Processing request", request_id=request_id)

        # Validate service first
        if not self.validate_service():
            self.error_count += 1
            self.logger.error("Service validation failed")
            return {"error": "Service not properly configured"}

        # Check cache
        cache_key = f"request_{request_id}"
        cached_result = self.cache_get(cache_key)
        if cached_result is not None:
            self.logger.info("Cache hit for request", request_id=request_id)
            return cached_result if isinstance(cached_result, dict) else {}

        # Time the operation
        start_time = self._start_timing()

        try:
            # Simulate processing
            time.sleep(0.001)

            result: dict[str, object] = {
                "request_id": request_id,
                "service_id": self.id,
                "status": "success",
                "data": data,
                "processed_at": FlextUtilities.generate_iso_timestamp(),
            }

            execution_time = self._get_execution_time_seconds(start_time)
            result["execution_time"] = execution_time

            # Cache successful results
            self.cache_set(cache_key, result)

            self.request_count += 1
            self.logger.info("Request completed", request_id=request_id)

            return result

        except (RuntimeError, ValueError, TypeError) as e:
            execution_time = self._get_execution_time_seconds(start_time)
            self.error_count += 1
            self.logger.exception("Request failed", request_id=request_id)

            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
            }

    def get_service_metrics(self) -> dict[str, object]:
        """Get comprehensive service metrics."""
        return {
            "service_id": self.id,
            "service_name": self.service_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
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
        self.data = data
        self._domain_events: list[tuple[str, dict[str, object]]] = []

    def clear_domain_events(self) -> None:
        """Clear collected domain events."""
        self._domain_events.clear()

    def get_domain_events(self) -> list[object]:
        """Get collected domain events."""
        return list(self._domain_events)

    def update_data(self, new_data: dict[str, object]) -> None:
        """Update entity data with timestamp tracking."""
        self.data.update(new_data)
        self._update_timestamp()
        print(f"🔄 Entity {self.entity_type} updated")

    def get_entity_info(self) -> dict[str, object]:
        """Get comprehensive entity information."""
        return {
            "id": self.id,
            "type": self.entity_type,
            "data": self.data,
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
            self.mark_valid()

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
    print("\n🔧 Individual Mixins Demonstration")
    print("=" * 50)

    # Timestamp mixin
    print("📋 Timestamp Mixin:")
    doc = TimestampedDocument("Test Document", "Initial content")
    print(f"  📄 Created: {doc}")
    time.sleep(0.1)
    doc.update_content("Updated content")
    print(f"  📄 After update: {doc}")

    # Identifiable mixin
    print("\n📋 Identifiable Mixin:")
    # Create user using shared domain factory first
    user_result = SharedDomainFactory.create_user("john_doe", "john@example.com", 25)
    if user_result.success:
        shared_user = user_result.data
        if shared_user is None:
            print("❌ User creation returned None data")
            return
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
        print(f"  👤 User: {user}")
        user.change_username("john_smith")
        print(f"  👤 Updated: {user}")
    else:
        print(f"  ❌ Failed to create user: {user_result.error}")

    # Validatable mixin
    print("\n📋 Validatable Mixin:")
    config = ValidatableConfiguration(
        "database",
        {"database_url": "localhost", "timeout": 30},
    )
    config.apply_configuration()

    invalid_config = ValidatableConfiguration("invalid", {"timeout": -1})
    invalid_config.apply_configuration()

    # Loggable mixin
    print("\n📋 Loggable Mixin:")
    service = LoggableService("user_service")
    result = service.process_request("req_001", {"action": "create_user"})
    print(f"  📊 Result: {result['status']}")

    # Timing mixin
    print("\n📋 Timing Mixin:")
    operation = TimedOperation("data_processing")
    result = operation.execute_operation(500)
    print(f"  ⚡ Execution time: {result['execution_time']:.4f}s")

    # Cacheable mixin
    print("\n📋 Cacheable Mixin:")
    calculator = CacheableCalculator()

    # First calculation (no cache)
    result1 = calculator.fibonacci(10)
    print(f"  🔢 fib(10) = {result1}")

    # Second calculation (cached)
    result2 = calculator.fibonacci(10)
    print(f"  🔢 fib(10) = {result2}")

    stats = calculator.get_stats()
    print(f"  📊 Calculator stats: {stats}")


def demonstrate_multiple_inheritance() -> None:
    """Demonstrate multiple inheritance patterns."""
    print("\n🏗️ Multiple Inheritance Demonstration")
    print("=" * 50)

    # Advanced user with multiple mixins
    print("📋 Advanced User (4 mixins):")
    user = AdvancedUser("alice_admin", "alice@company.com", 28, "user")

    # Validate user
    is_valid = user.validate_user()
    print(f"  ✅ User validation: {is_valid}")

    # Promote user
    promoted = user.promote_to_admin()
    print(f"  🚀 Promotion successful: {promoted}")

    # Get user info
    info = user.get_user_info()
    print(
        f"  📊 User info: ID={info['id']}, Role={info['role']}, "
        f"Valid={info['is_valid']}",
    )

    # Smart document with multiple mixins
    print("\n📋 Smart Document (4 mixins):")
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
    view1 = doc1.view_document()
    print(f"  📄 Document 1 word count: {view1['word_count']}")

    doc1.view_document()  # Should hit cache

    # Compare documents
    comparison = doc1.compare_with(doc2)
    print(f"  🔍 Documents comparison: category_match={comparison['category_match']}")

    # Update and view again (cache cleared)
    doc1.update_content("Updated AI research content with more details...")
    view1_updated = doc1.view_document()
    print(f"  📄 Updated document word count: {view1_updated['word_count']}")

    # Enterprise service with all mixins
    print("\n📋 Enterprise Service (5 mixins):")
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
        result = service.process_request(request_id, {"amount": 100 + i * 10})
        print(
            f"  📦 Request {request_id}: {result['status']} in "
            f"{result.get('execution_time', 0):.4f}s",
        )

    # Get service metrics
    metrics = service.get_service_metrics()
    print(
        f"  📊 Service metrics: {metrics['request_count']} requests, "
        f"{metrics['error_rate']:.2%} error rate",
    )


def demonstrate_composite_mixins() -> None:
    """Demonstrate composite mixin patterns."""
    print("\n🏢 Composite Mixins Demonstration")
    print("=" * 50)

    # Domain entity using FlextEntityMixin
    print("📋 Domain Entity (FlextEntityMixin):")
    entity = DomainEntity(
        "Customer",
        {
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active",
        },
    )

    entity_info = entity.get_entity_info()
    print(f"  🏬 Entity created: {entity_info['type']} (ID: {entity_info['id']})")

    entity.update_data(
        {"status": "premium", "last_login": FlextUtilities.generate_iso_timestamp()},
    )
    updated_info = entity.get_entity_info()
    data_field = updated_info["data"]
    field_count = len(data_field) if hasattr(data_field, "__len__") else 0
    print(f"  🔄 Entity updated: {field_count} fields")

    # Value object using FlextValueObjectMixin
    print("\n📋 Value Objects (FlextValueObjectMixin):")

    # Valid value object
    price = ValueObjectExample("Price", 99.99, "USD")
    is_valid = price.validate_value()
    print(f"  💰 {price} - Valid: {is_valid}")

    # Invalid value object
    invalid_value = ValueObjectExample("", None)
    is_invalid = invalid_value.validate_value()
    print(f"  ❌ Invalid value object - Valid: {is_invalid}")
    print(f"     Errors: {invalid_value.validation_errors}")

    # Serialization demo
    price_dict = price.to_dict()
    print(f"  📋 Serialized price: {price_dict}")


def demonstrate_method_resolution_order() -> None:
    """Demonstrate method resolution order in multiple inheritance."""
    print("\n🔗 Method Resolution Order Demonstration")
    print("=" * 50)

    class ComplexClass(
        FlextLoggableMixin,
        FlextTimingMixin,
        FlextValidatableMixin,
        FlextIdentifiableMixin,
    ):
        """Class with complex multiple inheritance."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.generate_id()
            # Validation state is initialized lazily via method calls
            self.logger.info("Complex class initialized", name=name)

        def _get_execution_time_seconds(self, start_time: float) -> float:
            """Convert execution time from milliseconds to seconds."""
            return self._get_execution_time_ms(start_time) / 1000.0

        def perform_operation(self) -> dict[str, object]:
            """Operation using multiple mixin capabilities."""
            start_time = self._start_timing()

            # Validate state
            self.clear_validation_errors()
            if not self.name:
                self.add_validation_error("Name cannot be empty")

            is_valid = len(self.validation_errors) == 0
            if is_valid:
                self.mark_valid()

            # Simulate work
            time.sleep(0.001)

            execution_time = self._get_execution_time_seconds(start_time)

            result = {
                "id": self.id,
                "name": self.name,
                "is_valid": is_valid,
                "execution_time": execution_time,
                "mro_length": len(self.__class__.__mro__),
            }

            self.logger.info("Operation completed", name=self.name)
            return result

    # Demonstrate MRO
    complex_obj = ComplexClass("test_object")
    print("📋 Method Resolution Order:")
    for i, cls in enumerate(complex_obj.__class__.__mro__):
        print(f"  {i}: {cls.__name__}")

    # Perform operation
    result = complex_obj.perform_operation()
    print(
        f"  📊 Operation result: Valid={result['is_valid']}, "
        f"Time={result['execution_time']:.4f}s",
    )


def demonstrate_performance_characteristics() -> None:
    """Demonstrate performance characteristics of mixins."""
    print("\n⚡ Performance Characteristics Demonstration")
    print("=" * 50)

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
        FlextLoggableMixin,
        FlextValidatableMixin,
    ):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.generate_id()
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

    print(f"📋 Performance Comparison ({operations} operations):")
    print(f"  🔹 Simple class: {simple_time:.4f}s ({operations / simple_time:.0f}/s)")
    print(f"  🔹 Single mixin: {single_time:.4f}s ({operations / single_time:.0f}/s)")
    print(
        f"  🔹 Multiple mixins: {multiple_time:.4f}s "
        f"({operations / multiple_time:.0f}/s)",
    )

    # Calculate overhead
    single_overhead = ((single_time - simple_time) / simple_time) * 100
    multiple_overhead = ((multiple_time - simple_time) / simple_time) * 100

    print("📊 Overhead Analysis:")
    print(f"  🔹 Single mixin overhead: {single_overhead:.1f}%")
    print(f"  🔹 Multiple mixins overhead: {multiple_overhead:.1f}%")


def demonstrate_enterprise_patterns() -> None:
    """Demonstrate enterprise patterns with mixins using railway-oriented programming.

    Refactored from complexity 37 to follow Single Responsibility Principle.
    Each specialized demonstration method handles a specific pattern.
    """
    print("\n🏭 Enterprise Patterns Demonstration")
    print("=" * 50)

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
        FlextLoggableMixin,
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

        def _get_execution_time_seconds(self, start_time: float) -> float:
            """Convert execution time from milliseconds to seconds."""
            return self._get_execution_time_ms(start_time) / 1000.0

        def _log_save_result(
            self,
            user_id: str,
            start_time: float,
            result: FlextResult[None],
        ) -> FlextResult[None]:
            """Log the save operation result with execution time."""
            _ = self._get_execution_time_seconds(start_time)
            if result.is_success:
                self.logger.info("User saved", user_id=user_id)
            else:
                self.logger.error("Failed to save user", user_id=user_id, error=result.error)
            return result

        def save_user(
            self, user_id: str, user_data: dict[str, object]
        ) -> FlextResult[None]:
            """Save user with caching and logging using railway-oriented programming."""
            start_time = self._start_timing()

            return self._execute_save_operation(user_id, user_data).flat_map(
                lambda _: self._log_save_result(
                    user_id, start_time, FlextResult.ok(None)
                )
            )

        def _check_cache_for_user(
            self, user_id: str
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
            self, user_id: str
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
            self, user_id: str
        ) -> FlextResult[dict[str, object] | None]:
            """Handle case when user is not found."""
            self.logger.warning("User not found", user_id=user_id)
            return FlextResult.ok(None)

        def find_user(self, user_id: str) -> FlextResult[dict[str, object] | None]:
            """Find user with caching and logging using railway-oriented programming."""
            self._start_timing()

            # Try cache first, then storage, then handle not found
            cache_result = self._check_cache_for_user(user_id)
            if cache_result.is_success and cache_result.data is not None:
                return cache_result

            storage_result = self._check_storage_for_user(user_id)
            if storage_result.is_success and storage_result.data is not None:
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
        FlextLoggableMixin,
        FlextValidatableMixin,
        FlextTimingMixin,
    ):
        """Order service with comprehensive mixins using railway-oriented programming."""

        def __init__(self, user_repo: UserRepositoryProtocol) -> None:
            super().__init__()
            self.user_repo = user_repo
            self.orders: dict[str, dict[str, object]] = {}
            self.generate_id()
            # Validation state is initialized lazily via method calls

        def _get_execution_time_seconds(self, start_time: float) -> float:
            """Convert execution time from milliseconds to seconds."""
            return self._get_execution_time_ms(start_time) / 1000.0

        def _validate_user_exists(self, user_id: str) -> FlextResult[dict[str, object]]:
            """Validate that user exists in repository."""
            user_result = self.user_repo.find_user(user_id)
            if user_result.is_failure:
                return FlextResult.fail(
                    f"Failed to check user existence: {user_result.error}"
                )

            user = user_result.data
            if not user:
                self.logger.error("Cannot create order: User not found", user_id=user_id)
                return FlextResult.fail(f"User not found: {user_id}")

            return FlextResult.ok(user)

        def _validate_order_items(
            self, items: list[dict[str, object]]
        ) -> FlextResult[None]:
            """Validate order items format and content."""
            self.clear_validation_errors()

            if not items:
                self.add_validation_error("Order must have at least one item")

            for item in items:
                if "product_id" not in item or "quantity" not in item:
                    self.add_validation_error("Invalid item format")

            if not self.is_valid:
                self.logger.error("Order validation failed", errors=self.validation_errors)
                return FlextResult.fail(
                    f"Order validation failed: {'; '.join(self.validation_errors)}"
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
            start_time: float,
            order: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Log successful order creation with execution time."""
            _ = self._get_execution_time_seconds(start_time)
            self.logger.info("Order created", order_id=order["order_id"])
            return FlextResult.ok(order)

        def create_order(
            self,
            user_id: str,
            items: list[dict[str, object]],
        ) -> FlextResult[dict[str, object]]:
            """Create order with validation and logging using railway-oriented programming."""
            start_time = self._start_timing()
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
    print("📋 Enterprise Repository Pattern:")

    # Save users with FlextResult pattern
    save_result_1 = user_repo.save_user(
        "user_001", {"name": "Alice", "email": "alice@example.com"}
    )
    save_result_2 = user_repo.save_user(
        "user_002", {"name": "Bob", "email": "bob@example.com"}
    )
    print(
        f"  💾 Save results: Alice={save_result_1.is_success}, Bob={save_result_2.is_success}"
    )

    # Find users (cache demo) with FlextResult pattern
    user1_result = user_repo.find_user("user_001")  # From storage
    user_repo.find_user("user_001")  # From cache

    if user1_result.is_success and user1_result.data:
        print(f"  👤 Found user: {user1_result.data['name']}")
    else:
        print("  ❌ User not found or error occurred")


def _demonstrate_service_pattern(order_service: OrderServiceProtocol) -> None:
    """Demonstrate service pattern with FlextResult."""
    print("\n📋 Enterprise Service Pattern:")

    # Create valid order with FlextResult pattern
    order_result = order_service.create_order(
        "user_001",
        [
            {"product_id": "prod_001", "quantity": 2},
            {"product_id": "prod_002", "quantity": 1},
        ],
    )

    if order_result.is_success and order_result.data:
        order = order_result.data
        items = order["items"]
        item_count = len(items) if hasattr(items, "__len__") else 0
        print(
            f"  📦 Order created: {order['order_id']} with {item_count} items",
        )
    else:
        print(f"  ❌ Order creation failed: {order_result.error}")

    # Try invalid order with FlextResult pattern
    invalid_order_result = order_service.create_order("user_999", [])
    if invalid_order_result.is_failure:
        print(f"  ❌ Invalid order result: {invalid_order_result.error}")
    else:
        print("  ⚠️ Unexpected success for invalid order")


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
