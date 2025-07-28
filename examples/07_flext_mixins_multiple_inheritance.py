#!/usr/bin/env python3
"""FLEXT Mixins - Multiple Inheritance and Composition Patterns Example.

Demonstrates comprehensive mixin usage with multiple inheritance patterns,
composition strategies, and enterprise-grade cross-cutting concerns integration.

Features demonstrated:
- Individual mixin patterns for specific behaviors
- Multiple inheritance composition with method resolution order
- Composite mixin patterns for common architectural use cases
- Cross-cutting concerns: logging, timing, caching, validation
- Enterprise service patterns with mixin combinations
- Performance characteristics of mixin overhead
- Method resolution order and conflict resolution
- Mixin design patterns and best practices
"""

from __future__ import annotations

import time
from typing import Any

from flext_core.mixins import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextEntityMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextValueObjectMixin,
)
from flext_core.utilities import FlextUtilities

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
        self._initialize_timestamps()

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


class IdentifiableUser(FlextIdentifiableMixin):
    """User with unique identification."""

    def __init__(self, username: str, email: str, user_id: str | None = None) -> None:
        """Initialize IdentifiableUser.

        Args:
            username: User username
            email: User email
            user_id: User ID

        """
        super().__init__()
        self.username = username
        self.email = email
        self._initialize_id(user_id)

    def change_username(self, new_username: str) -> bool:
        """Change username with ID validation."""
        if not self.has_id():
            print("❌ Cannot change username: User has no valid ID")
            return False

        old_username = self.username
        self.username = new_username
        print(f"👤 User {self.id}: '{old_username}' → '{new_username}'")
        return True

    def __str__(self) -> str:
        """Strings representation with ID."""
        return f"User({self.username}, ID: {self.id})"


class ValidatableConfiguration(FlextValidatableMixin):
    """Configuration with validation state tracking."""

    def __init__(self, config_name: str, settings: dict[str, Any]) -> None:
        """Initialize ValidatableConfiguration.

        Args:
            config_name: Configuration name
            settings: Configuration settings

        """
        super().__init__()
        self.config_name = config_name
        self.settings = settings
        self._initialize_validation()

    def validate_configuration(self) -> bool:
        """Validate configuration settings."""
        self._clear_validation_errors()

        # Required settings validation
        required_keys = ["database_url", "api_key", "timeout"]
        for key in required_keys:
            if key not in self.settings:
                self._add_validation_error(f"Missing required setting: {key}")

        # Value validation
        if "timeout" in self.settings:
            timeout = self.settings["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                self._add_validation_error("Timeout must be a positive number")

        # Set validation status
        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self._mark_valid()

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

    def process_request(self, request_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Process request with comprehensive logging."""
        self.logger.info("Processing request %s , request_id=request_id", request_id)

        try:
            # Simulate processing
            time.sleep(0.001)  # Minimal processing time

            result = {
                "request_id": request_id,
                "status": "success",
                "processed_at": FlextUtilities.generate_iso_timestamp(),
                "service": self.service_name,
            }

            self.logger.info(f"Request {request_id} completed successfully", **result)
            return result

        except Exception as e:
            self.logger.exception(
                f"Request {request_id} failed",
                error=str(e),
                request_id=request_id,
            )
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
            }

    def health_check(self) -> dict[str, Any]:
        """Perform health check with logging."""
        self.logger.debug("Performing health check")

        health_status = {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": FlextUtilities.generate_iso_timestamp(),
        }

        self.logger.info("Health check completed", **health_status)
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

    def execute_operation(self, complexity: int = 1000) -> dict[str, Any]:
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


class CacheableCalculator(FlextCacheableMixin):
    """Calculator with result caching."""

    def __init__(self) -> None:
        """Initialize CacheableCalculator."""
        super().__init__()
        self.calculation_count = 0

    def fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number with caching."""
        cache_key = f"fib_{n}"

        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            print(f"💰 Cache hit for fib({n}): {cached_result}")
            return cached_result

        # Calculate if not cached
        print(f"🔢 Calculating fib({n})")
        self.calculation_count += 1

        result = n if n <= 1 else self.fibonacci(n - 1) + self.fibonacci(n - 2)

        # Cache the result
        self._cache_set(cache_key, result)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get calculation statistics."""
        return {
            "calculations_performed": self.calculation_count,
            "cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
        }


# =============================================================================
# MULTIPLE INHERITANCE PATTERNS - Complex behavioral composition
# =============================================================================


class AdvancedUser(
    FlextIdentifiableMixin,
    FlextTimestampMixin,
    FlextValidatableMixin,
    FlextLoggableMixin,
):
    """User with multiple behavioral mixins."""

    def __init__(self, username: str, email: str, role: str = "user") -> None:
        """Initialize AdvancedUser.

        Args:
            username: User username
            email: User email
            role: User role

        """
        super().__init__()
        self.username = username
        self.email = email
        self.role = role

        # Initialize all mixins
        self._initialize_id()
        self._initialize_timestamps()
        self._initialize_validation()

        self.logger.info(
            f"Advanced user created: {username}",
            username=username,
            role=role,
        )

    def validate_user(self) -> bool:
        """Comprehensive user validation."""
        self._clear_validation_errors()

        # Username validation
        if not self.username or len(self.username) < 3:
            self._add_validation_error("Username must be at least 3 characters")

        # Email validation
        if not self.email or "@" not in self.email:
            self._add_validation_error("Invalid email format")

        # Role validation
        valid_roles = ["user", "admin", "moderator"]
        if self.role not in valid_roles:
            self._add_validation_error(f"Invalid role. Must be one of: {valid_roles}")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self._mark_valid()

        if is_valid:
            self.logger.info("User validation successful: %s", self.username)
        else:
            self.logger.warning(
                "User validation failed: %s",
                self.username,
                errors=self.validation_errors,
            )

        return is_valid

    def promote_to_admin(self) -> bool:
        """Promote user to admin with validation and logging."""
        if not self.validate_user():
            self.logger.error("Cannot promote invalid user to admin")
            return False

        if self.role == "admin":
            self.logger.warning(f"User {self.username} is already admin")
            return False

        old_role = self.role
        self.role = "admin"
        self._update_timestamp()

        self.logger.info(
            f"User promoted: {self.username}",
            old_role=old_role,
            new_role="admin",
            user_id=self.id,
        )
        return True

    def get_user_info(self) -> dict[str, Any]:
        """Get comprehensive user information."""
        age_seconds = time.time() - (self.created_at or 0)

        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at,
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
        self._initialize_timestamps()

    def view_document(self) -> dict[str, Any]:
        """View document with caching and statistics."""
        cache_key = f"view_{self.title}"

        # Check if view is cached
        cached_view = self._cache_get(cache_key)
        if cached_view is not None:
            print(f"📄 Cached view for: {self.title}")
            self.view_count += 1
            return cached_view

        # Generate view data
        view_data = {
            "title": self.title,
            "content": self.content[:100] + "..."
            if len(self.content) > 100
            else self.content,
            "category": self.category,
            "created_at": self.created_at,
            "word_count": len(self.content.split()),
            "character_count": len(self.content),
        }

        # Cache the view
        self._cache_set(cache_key, view_data)
        self.view_count += 1

        print(f"📄 Generated view for: {self.title}")
        return view_data

    def update_content(self, new_content: str) -> None:
        """Update content and clear cache."""
        self.content = new_content
        self._update_timestamp()

        # Clear cached views
        cache_key = f"view_{self.title}"
        cached_view = self._cache_get(cache_key)
        if cached_view is not None:
            self._cache_remove(cache_key)

        print(f"📝 Content updated for: {self.title}")

    def compare_with(self, other: SmartDocument) -> dict[str, Any]:
        """Compare documents using comparable mixin."""
        if not isinstance(other, SmartDocument):
            return {"error": "Can only compare with other SmartDocument instances"}

        return {
            "title_match": self.title == other.title,
            "category_match": self.category == other.category,
            "content_length_diff": len(self.content) - len(other.content),
            "age_diff_seconds": (self.created_at or 0) - (other.created_at or 0),
            "view_count_diff": self.view_count - other.view_count,
        }

    def to_dict(self) -> dict[str, Any]:
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
    FlextCacheableMixin,
):
    """Enterprise service with comprehensive mixin composition."""

    def __init__(self, service_name: str, config: dict[str, Any]) -> None:
        super().__init__()
        self.service_name = service_name
        self.config = config
        self.request_count = 0
        self.error_count = 0

        # Initialize all mixins
        self._initialize_id()
        self._initialize_validation()

        self.logger.info(f"Enterprise service initialized: {service_name}")

    def validate_service(self) -> bool:
        """Validate service configuration."""
        self._clear_validation_errors()

        required_config = ["host", "port", "timeout"]
        for key in required_config:
            if key not in self.config:
                self._add_validation_error(f"Missing required config: {key}")

        if "timeout" in self.config:
            timeout = self.config["timeout"]
            if not isinstance(timeout, int | float) or timeout <= 0:
                self._add_validation_error("Timeout must be positive number")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self._mark_valid()

        return is_valid

    def process_request(self, request_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Process request with comprehensive monitoring."""
        self.logger.info(f"Processing request: {request_id}")

        # Validate service first
        if not self.validate_service():
            self.error_count += 1
            self.logger.error("Service validation failed", request_id=request_id)
            return {"error": "Service not properly configured"}

        # Check cache
        cache_key = f"request_{request_id}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            self.logger.info(f"Cache hit for request: {request_id}")
            return cached_result

        # Time the operation
        start_time = self._start_timing()

        try:
            # Simulate processing
            time.sleep(0.001)

            result = {
                "request_id": request_id,
                "service_id": self.id,
                "status": "success",
                "data": data,
                "processed_at": FlextUtilities.generate_iso_timestamp(),
            }

            execution_time = self._get_execution_time_seconds(start_time)
            result["execution_time"] = execution_time

            # Cache successful results
            self._cache_set(cache_key, result)

            self.request_count += 1
            self.logger.info(
                f"Request completed: {request_id}",
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            execution_time = self._get_execution_time_seconds(start_time)
            self.error_count += 1
            self.logger.exception(
                f"Request failed: {request_id}",
                error=str(e),
                execution_time=execution_time,
            )

            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
            }

    def get_service_metrics(self) -> dict[str, Any]:
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

    def __init__(self, entity_type: str, data: dict[str, Any]) -> None:
        super().__init__()
        self.entity_type = entity_type
        self.data = data

    def update_data(self, new_data: dict[str, Any]) -> None:
        """Update entity data with timestamp tracking."""
        self.data.update(new_data)
        self._update_timestamp()
        print(f"🔄 Entity {self.entity_type} updated")

    def get_entity_info(self) -> dict[str, Any]:
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

    def __init__(self, name: str, value: Any, unit: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.value = value
        self.unit = unit
        self._initialize_validation()

    def validate_value(self) -> bool:
        """Validate value object."""
        self._clear_validation_errors()

        if not self.name:
            self._add_validation_error("Name cannot be empty")

        if self.value is None:
            self._add_validation_error("Value cannot be None")

        is_valid = len(self.validation_errors) == 0
        if is_valid:
            self._mark_valid()

        return is_valid

    def to_dict(self) -> dict[str, Any]:
        """Serialize value object."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "is_valid": self.is_valid,
        }

    def __str__(self) -> str:
        """String representation."""
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
    user = IdentifiableUser("john_doe", "john@example.com")
    print(f"  👤 User: {user}")
    user.change_username("john_smith")
    print(f"  👤 Updated: {user}")

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
    user = AdvancedUser("alice_admin", "alice@company.com", "user")

    # Validate user
    is_valid = user.validate_user()
    print(f"  ✅ User validation: {is_valid}")

    # Promote user
    promoted = user.promote_to_admin()
    print(f"  🚀 Promotion successful: {promoted}")

    # Get user info
    info = user.get_user_info()
    print(
        f"  📊 User info: ID={info['id']}, Role={info['role']}, Valid={info['is_valid']}",
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
            "host": "localhost",
            "port": 8080,
            "timeout": 30,
        },
    )

    # Process multiple requests
    for i in range(3):
        request_id = f"req_{i:03d}"
        result = service.process_request(request_id, {"amount": 100 + i * 10})
        print(
            f"  📦 Request {request_id}: {result['status']} in {result.get('execution_time', 0):.4f}s",
        )

    # Get service metrics
    metrics = service.get_service_metrics()
    print(
        f"  📊 Service metrics: {metrics['request_count']} requests, {metrics['error_rate']:.2%} error rate",
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
    print(f"  🔄 Entity updated: {len(updated_info['data'])} fields")

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
            self._initialize_id()
            self._initialize_validation()
            self.logger.info(f"Complex class initialized: {name}")

        def perform_operation(self) -> dict[str, Any]:
            """Operation using multiple mixin capabilities."""
            start_time = self._start_timing()

            # Validate state
            self._clear_validation_errors()
            if not self.name:
                self._add_validation_error("Name cannot be empty")

            is_valid = len(self.validation_errors) == 0
            if is_valid:
                self._mark_valid()

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

            self.logger.info(f"Operation completed for {self.name}", **result)
            return result

    # Demonstrate MRO
    complex_obj = ComplexClass("test_object")
    print("📋 Method Resolution Order:")
    for i, cls in enumerate(complex_obj.__class__.__mro__):
        print(f"  {i}: {cls.__name__}")

    # Perform operation
    result = complex_obj.perform_operation()
    print(
        f"  📊 Operation result: Valid={result['is_valid']}, Time={result['execution_time']:.4f}s",
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
            self._initialize_timestamps()

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
            self._initialize_id()
            self._initialize_timestamps()
            self._initialize_validation()

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
        obj = SingleMixinClass(f"test_{i}")
        obj.operation()
    single_time = time.time() - start_time

    # Test multiple mixin class
    start_time = time.time()
    for i in range(operations):
        obj = MultipleMixinClass(f"test_{i}")
        obj.operation()
    multiple_time = time.time() - start_time

    print(f"📋 Performance Comparison ({operations} operations):")
    print(f"  🔹 Simple class: {simple_time:.4f}s ({operations / simple_time:.0f}/s)")
    print(f"  🔹 Single mixin: {single_time:.4f}s ({operations / single_time:.0f}/s)")
    print(
        f"  🔹 Multiple mixins: {multiple_time:.4f}s ({operations / multiple_time:.0f}/s)",
    )

    # Calculate overhead
    single_overhead = ((single_time - simple_time) / simple_time) * 100
    multiple_overhead = ((multiple_time - simple_time) / simple_time) * 100

    print("📊 Overhead Analysis:")
    print(f"  🔹 Single mixin overhead: {single_overhead:.1f}%")
    print(f"  🔹 Multiple mixins overhead: {multiple_overhead:.1f}%")


def demonstrate_enterprise_patterns() -> None:
    """Demonstrate enterprise patterns with mixins."""
    print("\n🏭 Enterprise Patterns Demonstration")
    print("=" * 50)

    # Repository pattern with mixins
    class UserRepository(
        FlextLoggableMixin,
        FlextCacheableMixin,
        FlextTimingMixin,
    ):
        """Repository with enterprise mixin composition."""

        def __init__(self) -> None:
            super().__init__()
            self.users: dict[str, dict[str, Any]] = {}

        def save_user(self, user_id: str, user_data: dict[str, Any]) -> bool:
            """Save user with caching and logging."""
            start_time = self._start_timing()

            try:
                self.users[user_id] = user_data
                self.set_cached(f"user_{user_id}", user_data, ttl=60)

                execution_time = self._get_execution_time_seconds(start_time)
                self.logger.info(
                    f"User saved: {user_id}",
                    execution_time=execution_time,
                )

                return True

            except Exception as e:
                execution_time = self._get_execution_time_seconds(start_time)
                self.logger.exception(f"Failed to save user: {user_id}", error=str(e))
                return False

        def find_user(self, user_id: str) -> dict[str, Any] | None:
            """Find user with caching."""
            # Check cache first
            cache_key = f"user_{user_id}"
            cached_user = self._cache_get(cache_key)
            if cached_user is not None:
                self.logger.info(f"Cache hit for user: {user_id}")
                return cached_user

            # Check storage
            if user_id in self.users:
                user_data = self.users[user_id]
                self._cache_set(cache_key, user_data)
                self.logger.info(f"User found in storage: {user_id}")
                return user_data

            self.logger.warning(f"User not found: {user_id}")
            return None

    # Domain service pattern
    class OrderService(
        FlextIdentifiableMixin,
        FlextLoggableMixin,
        FlextValidatableMixin,
        FlextTimingMixin,
    ):
        """Order service with comprehensive mixins."""

        def __init__(self, user_repo: UserRepository) -> None:
            super().__init__()
            self.user_repo = user_repo
            self.orders: dict[str, dict[str, Any]] = {}
            self._initialize_id()
            self._initialize_validation()

        def create_order(
            self,
            user_id: str,
            items: list[dict[str, Any]],
        ) -> dict[str, Any] | None:
            """Create order with validation and logging."""
            start_time = self._start_timing()
            self.logger.info(f"Creating order for user: {user_id}")

            # Validate user exists
            user = self.user_repo.find_user(user_id)
            if not user:
                self.logger.error(f"Cannot create order: User not found: {user_id}")
                return None

            # Validate items
            self._clear_validation_errors()
            if not items:
                self._add_validation_error("Order must have at least one item")

            for item in items:
                if "product_id" not in item or "quantity" not in item:
                    self._add_validation_error("Invalid item format")

            if not self.is_valid:
                self.logger.error(
                    "Order validation failed",
                    errors=self.validation_errors,
                )
                return None

            # Create order
            order_id = FlextUtilities.generate_entity_id()
            order = {
                "order_id": order_id,
                "user_id": user_id,
                "items": items,
                "status": "pending",
                "created_at": FlextUtilities.generate_iso_timestamp(),
            }

            self.orders[order_id] = order

            execution_time = self._get_execution_time_seconds(start_time)
            self.logger.info(
                f"Order created: {order_id}",
                execution_time=execution_time,
            )

            return order

    # Demonstrate enterprise patterns
    print("📋 Enterprise Repository Pattern:")
    user_repo = UserRepository()

    # Save users
    user_repo.save_user("user_001", {"name": "Alice", "email": "alice@example.com"})
    user_repo.save_user("user_002", {"name": "Bob", "email": "bob@example.com"})

    # Find users (cache demo)
    user1 = user_repo.find_user("user_001")  # From storage
    user_repo.find_user("user_001")  # From cache
    print(f"  👤 Found user: {user1['name'] if user1 else 'None'}")

    print("\n📋 Enterprise Service Pattern:")
    order_service = OrderService(user_repo)

    # Create valid order
    order = order_service.create_order(
        "user_001",
        [
            {"product_id": "prod_001", "quantity": 2},
            {"product_id": "prod_002", "quantity": 1},
        ],
    )

    if order:
        print(
            f"  📦 Order created: {order['order_id']} with {len(order['items'])} items",
        )

    # Try invalid order
    invalid_order = order_service.create_order("user_999", [])
    print(f"  ❌ Invalid order result: {invalid_order}")


def main() -> None:
    """Run comprehensive FlextMixins demonstration."""
    print("=" * 80)
    print("🔧 FLEXT MIXINS - MULTIPLE INHERITANCE & COMPOSITION PATTERNS DEMONSTRATION")
    print("=" * 80)

    # Example 1: Individual Mixins
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Individual Mixin Patterns")
    print("=" * 60)
    demonstrate_individual_mixins()

    # Example 2: Multiple Inheritance
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Multiple Inheritance Composition")
    print("=" * 60)
    demonstrate_multiple_inheritance()

    # Example 3: Composite Mixins
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Composite Mixin Patterns")
    print("=" * 60)
    demonstrate_composite_mixins()

    # Example 4: Method Resolution Order
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Method Resolution Order")
    print("=" * 60)
    demonstrate_method_resolution_order()

    # Example 5: Performance Characteristics
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Performance Characteristics")
    print("=" * 60)
    demonstrate_performance_characteristics()

    # Example 6: Enterprise Patterns
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: Enterprise Architecture Patterns")
    print("=" * 60)
    demonstrate_enterprise_patterns()

    print("\n" + "=" * 80)
    print("🎉 FLEXT MIXINS DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
