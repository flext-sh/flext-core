#!/usr/bin/env python3
"""FLEXT Utilities - Generation, Formatting, and Validation Example.

Demonstrates comprehensive utility functions using FlextUtilities with enterprise-grade
ID generation, data formatting, validation, and type checking capabilities.

Features demonstrated:
- Entity ID generation with different strategies
- UUID generation for correlation tracking
- Timestamp formatting and ISO string generation
- Data validation and type checking
- Safe operations with error handling
- Hash generation and data integrity
- String manipulation and formatting
- Configuration value parsing
- Maximum type safety using flext_core.types
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from decimal import Decimal

# Import shared domain models to reduce duplication
from shared_domain import (
    Money,
    Order as SharedOrder,
    OrderStatus,
    Product as SharedProduct,
    SharedDomainFactory,
    User as SharedUser,
)

from flext_core import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextResult,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTypes,
    FlextUtilities,
    TAnyObject,
    TConfigDict,
    TEntityId,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# =============================================================================
# FORMATTING CONSTANTS - Data conversion and size formatting
# =============================================================================

# Time formatting constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Grade threshold constants
GRADE_A_THRESHOLD = 90
GRADE_B_THRESHOLD = 80

# Byte conversion constants
BYTES_PER_KB = 1024  # Standard bytes per kilobyte for binary calculations

# Age categorization constants
YOUNG_ADULT_AGE_THRESHOLD = 25
ADULT_AGE_THRESHOLD = 40
MIDDLE_AGED_THRESHOLD = 60

# Discount validation constants
MAX_DISCOUNT_PERCENTAGE = 100

# =============================================================================
# UTILITY HELPER FUNCTIONS - To bridge missing methods
# =============================================================================


def generate_prefixed_id(prefix: str, length: int) -> TEntityId:
    """Generate prefixed ID with specified length using TEntityId."""
    base_id = FlextUtilities.generate_id()
    return f"{prefix}-{base_id[:length]}"


def generate_hash_id(data: str) -> TEntityId:
    """Generate hash-based ID from data using TEntityId."""
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def is_email(value: str) -> bool:
    """Validate email format."""
    return isinstance(value, str) and "@" in value and "." in value.split("@")[-1]


def is_url(value: str) -> bool:
    """Validate URL format."""
    return isinstance(value, str) and (value.startswith(("http://", "https://")))


def is_string(value: object) -> bool:
    """Check if value is string using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, str)


def is_int(value: object) -> bool:
    """Check if value is integer using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, int)


def is_list(value: object) -> bool:
    """Check if value is list using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, list)


def is_dict(value: object) -> bool:
    """Check if value is dict using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, dict)


def is_non_empty_string(value: object) -> bool:
    """Check if value is non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


def generate_short_id(length: int = 8) -> TEntityId:
    """Generate short ID with specified length using TEntityId."""
    base_id = FlextUtilities.generate_id()
    return base_id[:length]


def get_age_category(age_value: int) -> str:
    """Get age category for analytics."""
    if age_value < YOUNG_ADULT_AGE_THRESHOLD:
        return "young_adult"
    if age_value < ADULT_AGE_THRESHOLD:
        return "adult"
    if age_value < MIDDLE_AGED_THRESHOLD:
        return "middle_aged"
    return "senior"


def calculate_discount_price(
    product: SharedProduct,
    discount_percentage: int,
) -> FlextResult[Money]:
    """Calculate discounted price using Money operations for shared products."""
    if discount_percentage < 0 or discount_percentage > MAX_DISCOUNT_PERCENTAGE:
        return FlextResult.fail("Discount percentage must be between 0 and 100")

    discount_factor = Decimal(str(100 - discount_percentage)) / Decimal(100)
    return product.price.multiply(discount_factor)


def process_shared_order(order: SharedOrder) -> FlextResult[SharedOrder]:
    """Process shared order with utility function approach."""
    # Validate order can be processed
    if order.status != OrderStatus.PENDING:
        error_msg = f"Cannot process order in {order.status.value} status"
        return FlextResult.fail(error_msg)

    # Calculate total for validation
    total_result = order.calculate_total()
    if total_result.is_failure:
        return FlextResult.fail(f"Cannot process order: {total_result.error}")

    # For demonstration, we'll simulate status update
    log_message = (
        f"✅ Shared order processed successfully: {order.id} "
        f"(Total: {total_result.data.amount} {total_result.data.currency})"
    )
    print(log_message)

    return FlextResult.ok(order)


# =============================================================================
# DOMAIN MODELS - Enhanced with flext-core mixins and shared domain
# =============================================================================

# Import shared domain models to reduce duplication

# Import flext-core mixins for advanced functionality


class UtilityDemoUser(SharedUser, FlextCacheableMixin, FlextSerializableMixin):
    """Enhanced user with utility mixins for demonstrations."""

    def get_cache_key(self) -> str:
        """Get cache key for user."""
        return f"user:{self.id}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 3600  # 1 hour

    def serialize_key(self) -> str:
        """Get serialization key."""
        return f"user_data:{self.id}"

    def to_serializable(self) -> TAnyObject:
        """Convert to serializable format with enhanced data."""
        base_data = {
            "id": self.id,
            "name": self.name,
            "email": self.email_address.email,
            "age": self.age.value,
            "status": self.status.value,
            "phone": self.phone.number if self.phone else None,
            "created_at": self.created_at,
        }
        return {
            **base_data,
            "cache_key": self.get_cache_key(),
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "status_display": self.status.value.title(),
            "age_category": self._get_age_category(),
        }

    def _get_age_category(self) -> str:
        """Get age category for analytics."""
        age_value = self.age.value
        if age_value < YOUNG_ADULT_AGE_THRESHOLD:
            return "young_adult"
        if age_value < ADULT_AGE_THRESHOLD:
            return "adult"
        if age_value < MIDDLE_AGED_THRESHOLD:
            return "middle_aged"
        return "senior"


class UtilityDemoProduct(SharedProduct, FlextComparableMixin, FlextCacheableMixin):
    """Enhanced product with comparison and caching capabilities."""

    def get_comparison_key(self) -> object:
        """Get key for comparison (price for products)."""
        return self.price.amount

    def get_cache_key(self) -> str:
        """Get cache key for product."""
        return f"product:{self.id}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 1800  # 30 minutes

    def calculate_discount_price(self, discount_percentage: int) -> FlextResult[Money]:
        """Calculate discounted price using Money operations."""
        if discount_percentage < 0 or discount_percentage > MAX_DISCOUNT_PERCENTAGE:
            return FlextResult.fail("Discount percentage must be between 0 and 100")

        discount_factor = Decimal(str(100 - discount_percentage)) / Decimal(100)
        return self.price.multiply(discount_factor)


class UtilityDemoOrder(SharedOrder, FlextLoggableMixin, FlextTimestampMixin):
    """Enhanced order with logging and timestamp capabilities."""

    def get_timestamp_fields(self) -> list[str]:
        """Get fields that should be timestamped."""
        return ["created_at", "updated_at", "last_modified"]

    def process_order(self) -> FlextResult[UtilityDemoOrder]:
        """Process order with comprehensive logging."""
        self.logger.info(
            "Processing order",
            order_id=self.id,
            customer_id=self.customer_id,
            item_count=len(self.items),
            current_status=self.status.value,
        )

        # Validate order can be processed
        if self.status != OrderStatus.PENDING:
            error_msg = f"Cannot process order in {self.status.value} status"
            self.logger.error(error_msg, order_id=self.id)
            return FlextResult.fail(error_msg)

        # Calculate total for validation
        total_result = self.calculate_total()
        if total_result.is_failure:
            self.logger.error(
                f"Cannot process order: {total_result.error}",
            )
            return FlextResult.fail(total_result.error)

        # For demonstration, we'll simulate status update
        log_message = (
            f"✅ Shared order processed successfully: {self.id} "
            f"(Total: {total_result.data.amount} {total_result.data.currency})"
        )
        print(log_message)

        return FlextResult.ok(self)


# =============================================================================
# DEMONSTRATION FUNCTIONS - Core utility demonstrations
# =============================================================================


def demonstrate_id_generation() -> None:
    """Demonstrate various ID generation strategies using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 1: ID Generation Strategies")
    print("=" * 60)

    # Basic entity ID generation
    entity_id: TEntityId = FlextUtilities.generate_entity_id()
    log_message = f"🔧 Entity ID: {entity_id}"
    print(log_message)

    # UUID generation
    uuid_id: TEntityId = FlextUtilities.generate_uuid()
    log_message = f"🆔 UUID: {uuid_id}"
    print(log_message)

    # Correlation ID generation
    correlation_id: TEntityId = FlextUtilities.generate_correlation_id()
    log_message = f"🔗 Correlation ID: {correlation_id}"
    print(log_message)

    # Custom prefixed ID
    prefixed_id: TEntityId = generate_prefixed_id("USER", 8)
    log_message = f"🏷️ Prefixed ID: {prefixed_id}"
    print(log_message)

    # Hash-based ID
    hash_id: TEntityId = generate_hash_id("user@example.com")
    log_message = f"🔐 Hash ID: {hash_id}"
    print(log_message)

    # Short ID
    short_id: TEntityId = generate_short_id(6)
    log_message = f"📏 Short ID: {short_id}"
    print(log_message)

    print("✅ ID generation demonstration completed")


def demonstrate_timestamp_generation() -> None:
    """Demonstrate timestamp generation and formatting using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 2: Timestamp Generation")
    print("=" * 60)

    # ISO timestamp generation
    iso_timestamp = FlextUtilities.generate_iso_timestamp()
    log_message = f"📅 ISO Timestamp: {iso_timestamp}"
    print(log_message)

    # Current timestamp
    current_timestamp = FlextUtilities.generate_timestamp()
    log_message = f"⏰ Current Timestamp: {current_timestamp}"
    print(log_message)

    # Session ID generation
    session_id = FlextUtilities.generate_session_id()
    log_message = f"🔑 Session ID: {session_id}"
    print(log_message)

    print("✅ Timestamp generation demonstration completed")


def demonstrate_hash_generation() -> None:
    """Demonstrate hash generation for data integrity using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 3: Hash Generation")
    print("=" * 60)

    # Test data
    test_data: TUserData = {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
    }

    # Generate hash from data
    data_string = json.dumps(test_data, sort_keys=True)
    data_hash = hashlib.sha256(data_string.encode()).hexdigest()
    log_message = f"🔐 Data Hash: {data_hash}"
    print(log_message)

    # Generate hash ID
    hash_id: TEntityId = generate_hash_id(data_string)
    log_message = f"🆔 Hash ID: {hash_id}"
    print(log_message)

    # Verify data integrity
    verification_hash = hashlib.sha256(data_string.encode()).hexdigest()
    is_valid = data_hash == verification_hash
    log_message = f"✅ Data Integrity: {is_valid}"
    print(log_message)

    print("✅ Hash generation demonstration completed")


def demonstrate_type_checking() -> None:
    """Demonstrate type checking using FlextTypes.TypeGuards."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 4: Type Checking with FlextTypes.TypeGuards")
    print("=" * 60)

    # Test values
    test_values: list[TAnyObject] = [
        "Hello World",
        42,
        [1, 2, 3],
        {"key": "value"},
        None,
        True,
    ]

    for value in test_values:
        # Check instance types
        is_str = FlextTypes.TypeGuards.is_instance_of(value, str)
        is_int = FlextTypes.TypeGuards.is_instance_of(value, int)
        is_list = FlextTypes.TypeGuards.is_instance_of(value, list)
        is_dict = FlextTypes.TypeGuards.is_instance_of(value, dict)

        log_message = f"🔍 {value} (type: {type(value).__name__})"
        print(log_message)
        print(f"   String: {is_str}")
        print(f"   Integer: {is_int}")
        print(f"   List: {is_list}")
        print(f"   Dict: {is_dict}")

    # Check callable
    is_callable_func = FlextTypes.TypeGuards.is_callable(len)
    is_callable_str = FlextTypes.TypeGuards.is_callable("not_callable")
    log_message = f"🔧 len() is callable: {is_callable_func}"
    print(log_message)
    log_message = f"🔧 'not_callable' is callable: {is_callable_str}"
    print(log_message)

    # Check dict-like
    is_dict_like_dict = FlextTypes.TypeGuards.is_dict_like({"a": 1})
    is_dict_like_list = FlextTypes.TypeGuards.is_dict_like([1, 2, 3])
    log_message = f"📚 {{'a': 1}} is dict-like: {is_dict_like_dict}"
    print(log_message)
    log_message = f"📚 [1, 2, 3] is dict-like: {is_dict_like_list}"
    print(log_message)

    # Check list-like
    is_list_like_list = FlextTypes.TypeGuards.is_list_like([1, 2, 3])
    is_list_like_str = FlextTypes.TypeGuards.is_list_like("string")
    log_message = f"📋 [1, 2, 3] is list-like: {is_list_like_list}"
    print(log_message)
    log_message = f"📋 'string' is list-like: {is_list_like_str}"
    print(log_message)

    print("✅ Type checking demonstration completed")


def demonstrate_data_validation() -> None:
    """Demonstrate data validation using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 5: Data Validation")
    print("=" * 60)

    # Test data for validation
    test_cases: list[tuple[str, TAnyObject]] = [
        ("Valid Email", "user@example.com"),
        ("Invalid Email", "invalid-email"),
        ("Valid URL", "https://example.com"),
        ("Invalid URL", "not-a-url"),
        ("Empty String", ""),
        ("Non-empty String", "Hello World"),
        ("Integer", 42),
        ("List", [1, 2, 3]),
        ("Dict", {"key": "value"}),
    ]

    for test_name, test_value in test_cases:
        log_message = f"🔍 Testing: {test_name} = {test_value}"
        print(log_message)

        # Email validation
        if is_string(test_value):
            email_valid = is_email(test_value)
            log_message = f"   📧 Email valid: {email_valid}"
            print(log_message)

        # URL validation
        if is_string(test_value):
            url_valid = is_url(test_value)
            log_message = f"   🌐 URL valid: {url_valid}"
            print(log_message)

        # String validation
        string_valid = is_string(test_value)
        log_message = f"   📝 String valid: {string_valid}"
        print(log_message)

        # Non-empty string validation
        if is_string(test_value):
            non_empty_valid = is_non_empty_string(test_value)
            log_message = f"   📝 Non-empty string: {non_empty_valid}"
            print(log_message)

        # Type validations
        int_valid = is_int(test_value)
        list_valid = is_list(test_value)
        dict_valid = is_dict(test_value)

        log_message = f"   🔢 Integer: {int_valid}"
        print(log_message)
        log_message = f"   📋 List: {list_valid}"
        print(log_message)
        log_message = f"   📚 Dict: {dict_valid}"
        print(log_message)

        print()

    print("✅ Data validation demonstration completed")


def demonstrate_safe_operations() -> None:
    """Demonstrate safe operations with error handling using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 6: Safe Operations")
    print("=" * 60)

    # Safe JSON parsing
    def parse_json(json_str: str) -> TAnyObject:
        """Safely parse JSON string using TAnyObject."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            error_message: TErrorMessage = f"JSON parsing failed: {e}"
            print(f"❌ {error_message}")
            return {}

    # Test JSON parsing
    valid_json = '{"name": "John", "age": 30}'
    invalid_json = '{"name": "John", "age": 30'  # Missing closing brace

    valid_result = parse_json(valid_json)
    log_message = f"✅ Valid JSON parsed: {valid_result}"
    print(log_message)

    invalid_result = parse_json(invalid_json)
    log_message = f"❌ Invalid JSON result: {invalid_result}"
    print(log_message)

    # Safe mathematical operations
    def risky_division(a: float, b: float) -> float:
        """Perform division with error handling."""
        try:
            return a / b
        except ZeroDivisionError:
            error_message: TErrorMessage = "Division by zero"
            print(f"❌ {error_message}")
            return 0.0

    # Test division
    safe_result = risky_division(10, 2)
    log_message = f"✅ Safe division: 10 / 2 = {safe_result}"
    print(log_message)

    unsafe_result = risky_division(10, 0)
    log_message = f"❌ Unsafe division: 10 / 0 = {unsafe_result}"
    print(log_message)

    # Safe list access
    def risky_list_access(items: list[TAnyObject], index: int) -> TAnyObject:
        """Safely access list item with error handling."""
        try:
            return items[index]
        except (IndexError, TypeError) as e:
            error_message: TErrorMessage = f"List access failed: {e}"
            print(f"❌ {error_message}")
            return None

    # Test list access
    test_list: list[TAnyObject] = [1, 2, 3, 4, 5]

    safe_access = risky_list_access(test_list, 2)
    log_message = f"✅ Safe list access: items[2] = {safe_access}"
    print(log_message)

    unsafe_access = risky_list_access(test_list, 10)
    log_message = f"❌ Unsafe list access: items[10] = {unsafe_access}"
    print(log_message)

    print("✅ Safe operations demonstration completed")


def demonstrate_string_utilities() -> None:
    """Demonstrate string manipulation utilities using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 7: String Utilities")
    print("=" * 60)

    # Test strings
    test_strings: list[str] = [
        "  hello world  ",
        "UPPERCASE TEXT",
        "lowercase text",
        "camelCaseText",
        "snake_case_text",
        "kebab-case-text",
    ]

    for test_str in test_strings:
        log_message = f"🔍 Original: '{test_str}'"
        print(log_message)

        # Trimming
        trimmed = test_str.strip()
        log_message = f"   ✂️ Trimmed: '{trimmed}'"
        print(log_message)

        # Case conversion
        upper = test_str.upper()
        lower = test_str.lower()
        log_message = f"   🔤 Uppercase: '{upper}'"
        print(log_message)
        log_message = f"   🔤 Lowercase: '{lower}'"
        print(log_message)

        # Length
        length = len(test_str)
        log_message = f"   📏 Length: {length}"
        print(log_message)

        print()

    print("✅ String utilities demonstration completed")


def demonstrate_config_utilities() -> None:
    """Demonstrate configuration value parsing using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 8: Configuration Utilities")
    print("=" * 60)

    # Test configuration values
    config_values: TConfigDict = {
        "string_value": "hello world",
        "int_value": "42",
        "float_value": "3.14",
        "bool_value": "true",
        "list_value": "item1,item2,item3",
        "invalid_int": "not_a_number",
        "invalid_bool": "maybe",
    }

    for key, value in config_values.items():
        log_message = f"🔧 Config: {key} = '{value}'"
        print(log_message)

        # Parse boolean
        def parse_bool(val: object) -> bool:
            """Parse boolean value safely."""
            if isinstance(val, str):
                return val.lower() in {"true", "1", "yes", "on"}
            return bool(val)

        bool_result = parse_bool(value)
        log_message = f"   ✅ Boolean: {bool_result}"
        print(log_message)

        # Parse list
        def parse_list(val: object) -> list[str]:
            """Parse comma-separated list safely."""
            if isinstance(val, str):
                return [item.strip() for item in val.split(",")]
            return []

        list_result = parse_list(value)
        log_message = f"   📋 List: {list_result}"
        print(log_message)

        # Parse integer
        try:
            int_result = int(value)
            log_message = f"   🔢 Integer: {int_result}"
            print(log_message)
        except ValueError:
            log_message = "   ❌ Integer: Invalid"
            print(log_message)

        print()

    print("✅ Configuration utilities demonstration completed")


def demonstrate_formatting_utilities() -> None:
    """Demonstrate data formatting utilities using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 9: Formatting Utilities")
    print("=" * 60)

    # Size formatting
    def format_bytes(bytes_value: int) -> str:
        """Format bytes into human-readable format."""
        if bytes_value == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = math.floor(math.log(bytes_value, BYTES_PER_KB))
        p = math.pow(BYTES_PER_KB, i)
        s = round(bytes_value / p, 2)

        return f"{s} {size_names[i]}"

    # Test size formatting
    test_sizes = [0, 1024, 1048576, 1073741824, 1099511627776]

    for size in test_sizes:
        formatted_size = format_bytes(size)
        log_message = f"📏 {size} bytes = {formatted_size}"
        print(log_message)

    # Percentage formatting
    def format_percentage(value: float, total: float) -> str:
        """Format percentage with error handling."""
        try:
            percentage = (value / total) * 100
            return f"{percentage:.2f}%"
        except ZeroDivisionError:
            return "0.00%"

    # Test percentage formatting
    test_percentages = [(25, 100), (0, 50), (75, 75), (10, 0)]

    for value, total in test_percentages:
        percentage = format_percentage(value, total)
        log_message = f"📊 {value}/{total} = {percentage}"
        print(log_message)

    # Duration formatting
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < SECONDS_PER_MINUTE:
            return f"{seconds:.2f}s"
        if seconds < SECONDS_PER_HOUR:
            minutes = seconds / SECONDS_PER_MINUTE
            return f"{minutes:.2f}m"
        hours = seconds / SECONDS_PER_HOUR
        return f"{hours:.2f}h"

    # Test duration formatting
    test_durations = [30.5, 125.0, 3720.0]

    for duration in test_durations:
        formatted_duration = format_duration(duration)
        log_message = f"⏱️ {duration}s = {formatted_duration}"
        print(log_message)

    print("✅ Formatting utilities demonstration completed")


def demonstrate_enterprise_scenarios() -> None:  # noqa: PLR0912, PLR0915
    """Demonstrate enterprise scenarios using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 10: Enterprise Scenarios")
    print("=" * 60)

    # Scenario 1: User registration with validation
    log_message = "👤 Scenario 1: User Registration"
    print(log_message)

    user_data: TUserData = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "age": 28,
    }

    # Validate user data
    validation_errors: list[TErrorMessage] = []

    if not is_non_empty_string(user_data.get("name")):
        validation_errors.append("Name is required")

    if not is_email(user_data.get("email", "")):
        validation_errors.append("Valid email is required")

    if not is_int(user_data.get("age")):
        validation_errors.append("Age must be a number")

    if validation_errors:
        log_message = f"❌ Validation failed: {validation_errors}"
        print(log_message)
    else:
        log_message = "✅ User data validation passed"
        print(log_message)

    # Scenario 2: Order processing with ID generation
    log_message = "🛒 Scenario 2: Order Processing"
    print(log_message)

    order_items: list[TAnyObject] = [
        {"name": "Product A", "price": 29.99, "quantity": 2},
        {"name": "Product B", "price": 15.50, "quantity": 1},
    ]

    order_id: TEntityId = FlextUtilities.generate_entity_id()
    customer_id: TEntityId = FlextUtilities.generate_entity_id()

    {
        "id": order_id,
        "customer_id": customer_id,
        "items": order_items,
        "created_at": FlextUtilities.generate_iso_timestamp(),
    }

    log_message = f"✅ Order created: {order_id}"
    print(log_message)

    # Scenario 3: Configuration parsing
    log_message = "⚙️ Scenario 3: Configuration Parsing"
    print(log_message)

    raw_config: TConfigDict = {
        "database_url": "postgresql://localhost:5432/mydb",
        "max_connections": "10",
        "debug_mode": "true",
        "allowed_hosts": "localhost,127.0.0.1,example.com",
    }

    parsed_config: TConfigDict = {}

    # Parse configuration values
    for key, value in raw_config.items():
        if key == "max_connections":
            try:
                parsed_config[key] = int(value)
            except ValueError:
                parsed_config[key] = 5  # Default value
        elif key == "debug_mode":
            parsed_config[key] = value.lower() == "true"
        elif key == "allowed_hosts":
            parsed_config[key] = [host.strip() for host in value.split(",")]
        else:
            parsed_config[key] = value

    log_message = f"✅ Configuration parsed: {parsed_config}"
    print(log_message)

    # Scenario 4: Data transformation pipeline
    log_message = "🔄 Scenario 4: Data Transformation Pipeline"
    print(log_message)

    raw_data: list[TAnyObject] = [
        {"id": "1", "name": "Alice", "score": "85"},
        {"id": "2", "name": "Bob", "score": "92"},
        {"id": "3", "name": "Charlie", "score": "78"},
    ]

    transformed_data: list[TAnyObject] = []

    for item in raw_data:
        if is_dict(item) and "score" in item:
            try:
                score = int(item["score"])
                transformed_item: TAnyObject = {
                    "id": item["id"],
                    "name": item["name"],
                    "score": score,
                    "grade": (
                        "A"
                        if score >= GRADE_A_THRESHOLD
                        else "B"
                        if score >= GRADE_B_THRESHOLD
                        else "C"
                    ),
                }
                transformed_data.append(transformed_item)
            except (ValueError, TypeError):
                log_message = f"⚠️ Skipping invalid score: {item}"
                print(log_message)

    log_message = f"✅ Transformed {len(transformed_data)} records"
    print(log_message)

    print("✅ Enterprise scenarios demonstration completed")


def demonstrate_performance_benchmarks() -> None:
    """Demonstrate performance benchmarks using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 11: Performance Benchmarks")
    print("=" * 60)

    # Benchmark ID generation
    log_message = "🏃 Benchmarking ID Generation"
    print(log_message)

    start_time = time.time()
    _ids: list[TEntityId] = [FlextUtilities.generate_entity_id() for _ in range(1000)]
    end_time = time.time()

    generation_time = end_time - start_time
    log_message = f"✅ Generated 1000 IDs in {generation_time:.4f}s"
    print(log_message)

    # Benchmark type checking
    log_message = "🏃 Benchmarking Type Checking"
    print(log_message)

    test_objects: list[TAnyObject] = ["string", 42, [1, 2, 3], {"key": "value"}] * 250

    start_time = time.time()
    _type_results: list[bool] = [
        FlextTypes.TypeGuards.is_instance_of(obj, str) for obj in test_objects
    ]
    end_time = time.time()

    checking_time = end_time - start_time
    log_message = f"✅ Checked 1000 objects in {checking_time:.4f}s"
    print(log_message)

    # Benchmark hash generation
    log_message = "🏃 Benchmarking Hash Generation"
    print(log_message)

    test_strings = [f"data_{i}" for i in range(1000)]

    start_time = time.time()
    _hashes: list[TEntityId] = [generate_hash_id(test_str) for test_str in test_strings]
    end_time = time.time()

    hash_time = end_time - start_time
    log_message = f"✅ Generated 1000 hashes in {hash_time:.4f}s"
    print(log_message)

    # Performance summary
    log_message = "\n📊 Performance Summary:"
    print(log_message)
    log_message = f"   ID Generation: {generation_time:.4f}s"
    print(log_message)
    log_message = f"   Type Checking: {checking_time:.4f}s"
    print(log_message)
    log_message = f"   Hash Generation: {hash_time:.4f}s"
    print(log_message)

    print("✅ Performance benchmarks demonstration completed")


def demonstrate_mixin_functionality() -> None:  # noqa: PLR0915
    """Demonstrate enhanced mixin functionality using shared domain models."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 12: Enhanced Mixin Functionality")
    print("=" * 60)

    # Create enhanced domain objects using shared domain and mixins

    # Create enhanced user using factory
    user_result = SharedDomainFactory.create_user(
        name="Alice Johnson",
        email="alice@example.com",
        age=28,
    )

    if user_result.is_failure:
        print(f"❌ Failed to create user: {user_result.error}")
        return

    shared_user = user_result.data

    # Use shared user directly with utility functions
    user = shared_user

    log_message = "👤 Shared User Utility Capabilities:"
    print(log_message)

    # Demonstrate caching capabilities using utility functions
    cache_key = f"user:{user.id}"
    cache_ttl = 3600  # 1 hour
    log_message = f"   🗂️ Cache Key: {cache_key} (TTL: {cache_ttl}s)"
    print(log_message)

    # Demonstrate serialization capabilities using shared user
    serialized_data: TAnyObject = {
        "id": user.id,
        "name": user.name,
        "email": user.email_address.email,
        "age": user.age.value,
        "status": user.status.value,
        "phone": user.phone.number if user.phone else None,
        "created_at": str(user.created_at) if user.created_at else None,
        "cache_key": cache_key,
        "serialized_at": FlextUtilities.generate_iso_timestamp(),
        "status_display": user.status.value.title(),
        "age_category": get_age_category(user.age.value),
    }
    log_message = f"   📤 Serialized Data: {len(str(serialized_data))} characters"
    print(log_message)
    log_message = f"   📊 Age Category: {serialized_data.get('age_category')}"
    print(log_message)

    print()

    # Create enhanced product
    product_result = SharedDomainFactory.create_product(
        name="Gaming Laptop",
        description="High-performance gaming laptop with RTX graphics",
        price_amount="1299.99",
        currency="USD",
        category="Electronics",
    )

    if product_result.is_failure:
        print(f"❌ Failed to create product: {product_result.error}")
        return

    shared_product = product_result.data

    # Use shared product directly with utility functions
    product = shared_product

    log_message = "🛍️ Shared Product Utility Capabilities:"
    print(log_message)

    # Demonstrate comparison capabilities using shared product
    comparison_key = product.price.amount
    log_message = f"   ⚖️ Comparison Key (Price): ${comparison_key}"
    print(log_message)

    # Demonstrate discount calculation using utility function
    discount_result = calculate_discount_price(product, 15)
    if discount_result.is_success:
        discount_price = discount_result.data
        log_message = f"   💰 Original Price: ${product.price.amount}"
        print(log_message)
        log_message = f"   🏷️ 15% Discount Price: ${discount_price.amount}"
        print(log_message)

    # Demonstrate cache capabilities using shared product
    product_cache_key = f"product:{product.id}"
    product_cache_ttl = 1800  # 30 minutes
    log_message = f"   🗂️ Cache Key: {product_cache_key} (TTL: {product_cache_ttl}s)"
    print(log_message)

    print()

    # Create enhanced order with multiple items
    order_items = [
        {
            "product_id": product.id,
            "product_name": product.name,
            "quantity": 1,
            "unit_price": "1299.99",
            "currency": "USD",
        },
        {
            "product_id": "prod_002",
            "product_name": "Gaming Mouse",
            "quantity": 2,
            "unit_price": "79.99",
            "currency": "USD",
        },
    ]

    order_result = SharedDomainFactory.create_order(
        customer_id=user.id,
        items=order_items,
    )

    if order_result.is_failure:
        print(f"❌ Failed to create order: {order_result.error}")
        return

    shared_order = order_result.data

    # Use shared order directly with utility functions
    order = shared_order

    log_message = "📦 Shared Order Utility Capabilities:"
    print(log_message)

    # Demonstrate timestamp capabilities using shared order
    timestamp_fields = ["created_at", "updated_at", "last_modified"]
    log_message = f"   ⏰ Timestamp Fields: {timestamp_fields}"
    print(log_message)

    # Demonstrate order processing with utility function
    process_result = process_shared_order(order)
    if process_result.is_success:
        processed_order = process_result.data
        log_message = f"   ✅ Order Processed: {processed_order.id}"
        print(log_message)
        log_message = f"   📊 Status: {processed_order.status.value}"
        print(log_message)
    else:
        log_message = f"   ❌ Processing Failed: {process_result.error}"
        print(log_message)

    print("✅ Enhanced mixin functionality demonstration completed")


def main() -> None:
    """Run comprehensive FlextUtilities demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT UTILITIES - GENERATION, FORMATTING, AND VALIDATION DEMONSTRATION")
    print("=" * 80)

    # Run all demonstrations
    demonstrate_id_generation()
    demonstrate_timestamp_generation()
    demonstrate_hash_generation()
    demonstrate_type_checking()
    demonstrate_data_validation()
    demonstrate_safe_operations()
    demonstrate_string_utilities()
    demonstrate_config_utilities()
    demonstrate_formatting_utilities()
    demonstrate_enterprise_scenarios()
    demonstrate_performance_benchmarks()
    demonstrate_mixin_functionality()

    print("\n" + "=" * 80)
    print("🎉 FLEXT UTILITIES DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
