# FlextConstants Implementation Guide

**Version**: 0.9.0  
**Module**: `flext_core.constants`  
**Target Audience**: Senior Developers, Software Architects, Platform Engineers

## Quick Start

This guide provides step-by-step implementation patterns for using FlextConstants across FLEXT ecosystem libraries, from basic constant usage to advanced library-specific extensions with comprehensive domain organization and type-safe constant management.

**Prerequisite**: Ensure `flext-core` is installed and available in your environment.

---

## 🚀 Basic Implementation

### Step 1: Import and Basic Usage

```python
from flext_core import FlextConstants
import re
from typing import Final

# Basic constant access patterns
def demonstrate_basic_usage():
    """Demonstrate basic FlextConstants usage patterns."""

    # System information access
    system_info = {
        "name": FlextConstants.Core.NAME,
        "version": FlextConstants.Core.VERSION,
        "ecosystem_size": FlextConstants.Core.ECOSYSTEM_SIZE,
        "architecture": FlextConstants.Core.ARCHITECTURE
    }
    print(f"✅ System Info: {system_info}")

    # Network configuration
    network_config = {
        "default_timeout": FlextConstants.Network.DEFAULT_TIMEOUT,
        "connection_timeout": FlextConstants.Network.CONNECTION_TIMEOUT,
        "http_port": FlextConstants.Network.HTTP_PORT,
        "https_port": FlextConstants.Network.HTTPS_PORT
    }
    print(f"✅ Network Config: {network_config}")

    # Error handling with structured codes
    error_info = {
        "validation_error": FlextConstants.Errors.VALIDATION_ERROR,
        "business_rule_violation": FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
        "authentication_failed": FlextConstants.Errors.AUTHENTICATION_FAILED,
        "error_message": FlextConstants.Errors.MESSAGES.get(
            FlextConstants.Errors.VALIDATION_ERROR, "Unknown error"
        )
    }
    print(f"✅ Error Info: {error_info}")

demonstrate_basic_usage()
```

### Step 2: Input Validation with FlextConstants

```python
from flext_core import FlextConstants, FlextResult

def validate_user_input(user_data: dict) -> FlextResult[dict]:
    """Comprehensive user data validation using FlextConstants."""

    # Name validation
    name = user_data.get("name", "")
    if not (FlextConstants.Validation.MIN_NAME_LENGTH <=
           len(name) <=
           FlextConstants.Validation.MAX_NAME_LENGTH):
        return FlextResult.fail(
            f"Name must be between {FlextConstants.Validation.MIN_NAME_LENGTH} "
            f"and {FlextConstants.Validation.MAX_NAME_LENGTH} characters",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )

    # Email validation
    email = user_data.get("email", "")
    if len(email) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
        return FlextResult.fail(
            f"Email exceeds maximum length of {FlextConstants.Validation.MAX_EMAIL_LENGTH}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )

    # Email pattern validation
    if not re.match(FlextConstants.Patterns.EMAIL_PATTERN, email):
        return FlextResult.fail(
            "Invalid email format",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )

    # Age validation
    age = user_data.get("age", 0)
    if not (FlextConstants.Validation.MIN_AGE <= age <= FlextConstants.Validation.MAX_AGE):
        return FlextResult.fail(
            f"Age must be between {FlextConstants.Validation.MIN_AGE} "
            f"and {FlextConstants.Validation.MAX_AGE}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )

    # Phone validation
    phone = user_data.get("phone", "")
    if phone and not (FlextConstants.Validation.MIN_PHONE_LENGTH <=
                     len(phone) <=
                     FlextConstants.Validation.MAX_PHONE_LENGTH):
        return FlextResult.fail(
            f"Phone must be between {FlextConstants.Validation.MIN_PHONE_LENGTH} "
            f"and {FlextConstants.Validation.MAX_PHONE_LENGTH} digits",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )

    return FlextResult.ok(user_data)

def test_user_validation():
    """Test user validation with various scenarios."""

    # Valid user data
    valid_user = {
        "name": "Alice Johnson",
        "email": "alice.johnson@example.com",
        "age": 28,
        "phone": "1234567890"
    }

    result = validate_user_input(valid_user)
    if result.success:
        print(f"✅ Valid user data: {result.value}")
    else:
        print(f"❌ Validation failed: {result.error}")

    # Invalid user data - name too short
    invalid_user = {
        "name": "A",  # Too short
        "email": "alice@example.com",
        "age": 28,
        "phone": "1234567890"
    }

    result = validate_user_input(invalid_user)
    if result.success:
        print(f"Unexpected success: {result.value}")
    else:
        print(f"❌ Expected validation failure: {result.error}")

    # Invalid user data - age too young
    invalid_age_user = {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 16,  # Too young
        "phone": "1234567890"
    }

    result = validate_user_input(invalid_age_user)
    if result.success:
        print(f"Unexpected success: {result.value}")
    else:
        print(f"❌ Expected age validation failure: {result.error}")

test_user_validation()
```

### Step 3: Error Handling with Structured Codes

```python
from flext_core import FlextConstants, FlextResult
from typing import Dict, object

class ErrorHandler:
    """Comprehensive error handling using FlextConstants error codes."""

    @staticmethod
    def categorize_error(error_code: str) -> Dict[str, object]:
        """Categorize error based on FlextConstants error code structure."""

        error_info = {
            "code": error_code,
            "message": FlextConstants.Errors.MESSAGES.get(error_code, "Unknown error"),
            "category": "unknown",
            "severity": "medium",
            "should_retry": False,
            "should_alert": False
        }

        # Business error range (1000-1999)
        if error_code.startswith("FLEXT_1"):
            error_info.update({
                "category": "business",
                "severity": "medium",
                "should_retry": False,
                "should_alert": True
            })

        # Technical error range (2000-2999)
        elif error_code.startswith("FLEXT_2"):
            error_info.update({
                "category": "technical",
                "severity": "high",
                "should_retry": True,
                "should_alert": True
            })

        # Validation error range (3000-3999)
        elif error_code.startswith("FLEXT_3"):
            error_info.update({
                "category": "validation",
                "severity": "low",
                "should_retry": False,
                "should_alert": False
            })

        # Security error range (4000-4999)
        elif error_code.startswith("FLEXT_4"):
            error_info.update({
                "category": "security",
                "severity": "critical",
                "should_retry": False,
                "should_alert": True
            })

        return error_info

    @staticmethod
    def handle_business_operation_error(operation_name: str, error: Exception) -> FlextResult[None]:
        """Handle business operation errors with appropriate categorization."""

        error_type = type(error).__name__

        if error_type == "ValueError":
            return FlextResult.fail(
                f"Business rule violation in {operation_name}: {str(error)}",
                error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION
            )
        elif error_type in ["ConnectionError", "TimeoutError"]:
            return FlextResult.fail(
                f"Connection error in {operation_name}: {str(error)}",
                error_code=FlextConstants.Errors.CONNECTION_ERROR
            )
        elif error_type == "PermissionError":
            return FlextResult.fail(
                f"Authorization denied for {operation_name}: {str(error)}",
                error_code=FlextConstants.Errors.AUTHORIZATION_DENIED
            )
        else:
            return FlextResult.fail(
                f"Generic error in {operation_name}: {str(error)}",
                error_code=FlextConstants.Errors.GENERIC_ERROR
            )

    @staticmethod
    def create_error_response(error_code: str, context: Dict[str, object] = None) -> Dict[str, object]:
        """Create standardized error response using FlextConstants."""

        error_info = ErrorHandler.categorize_error(error_code)

        response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": error_info["message"],
                "category": error_info["category"],
                "severity": error_info["severity"],
                "timestamp": "2024-01-01T12:00:00Z"  # Would use actual timestamp
            }
        }

        if context:
            response["error"]["context"] = context

        # Add retry information for technical errors
        if error_info["should_retry"]:
            response["error"]["retry_info"] = {
                "can_retry": True,
                "max_retries": FlextConstants.Defaults.MAX_RETRIES,
                "retry_delay": "exponential_backoff"
            }

        return response

def test_error_handling():
    """Test comprehensive error handling patterns."""

    error_handler = ErrorHandler()

    print("Testing error categorization...")

    # Test different error categories
    test_errors = [
        FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
        FlextConstants.Errors.CONNECTION_ERROR,
        FlextConstants.Errors.VALIDATION_ERROR,
        FlextConstants.Errors.AUTHORIZATION_DENIED
    ]

    for error_code in test_errors:
        error_info = error_handler.categorize_error(error_code)
        print(f"✅ Error {error_code}: {error_info['category']} ({error_info['severity']})")

    print("\nTesting business operation error handling...")

    # Simulate business operation errors
    try:
        raise ValueError("Invalid business rule: negative quantity not allowed")
    except Exception as e:
        result = error_handler.handle_business_operation_error("process_order", e)
        print(f"❌ Business error: {result.error} (Code: {result.error_code})")

    try:
        raise PermissionError("User lacks required permission")
    except Exception as e:
        result = error_handler.handle_business_operation_error("delete_resource", e)
        print(f"❌ Permission error: {result.error} (Code: {result.error_code})")

    print("\nTesting error response creation...")

    # Create standardized error responses
    business_error_response = error_handler.create_error_response(
        FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
        {"operation": "create_order", "user_id": "user123"}
    )
    print(f"✅ Business error response: {business_error_response}")

    technical_error_response = error_handler.create_error_response(
        FlextConstants.Errors.CONNECTION_ERROR,
        {"service": "payment_gateway", "timeout": 30}
    )
    print(f"✅ Technical error response: {technical_error_response}")

test_error_handling()
```

---

## 🏗️ Advanced Implementation

### Step 1: Library-Specific Constants Extension

```python
from flext_core import FlextConstants
from typing import Final

class FlextMeltanoConstants(FlextConstants):
    """Meltano-specific constants extending FlextConstants hierarchically."""

    class Extractors:
        """Data extraction constants for Meltano taps."""

        # Batch processing settings
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        LARGE_BATCH_SIZE: Final[int] = 5000
        MAX_BATCH_SIZE: Final[int] = 10000

        # Discovery and schema settings
        DISCOVERY_TIMEOUT: Final[int] = 60
        SCHEMA_CACHE_TTL: Final[int] = 1800  # 30 minutes
        MAX_PARALLEL_STREAMS: Final[int] = 4

        # Connection and retry settings
        CONNECTION_POOL_SIZE: Final[int] = 5
        MAX_CONNECTION_RETRIES: Final[int] = 3
        CONNECTION_RETRY_DELAY: Final[float] = 2.0

        # Stream processing limits
        MAX_RECORD_SIZE: Final[int] = 1024 * 1024  # 1MB
        BUFFER_SIZE: Final[int] = 64 * 1024  # 64KB
        MAX_STREAM_DURATION: Final[int] = 3600  # 1 hour

    class Targets:
        """Data loading constants for Meltano targets."""

        # Batch loading settings
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 5000
        MIN_BATCH_SIZE: Final[int] = 100

        # Error handling and retry
        MAX_LOAD_RETRIES: Final[int] = 3
        RETRY_EXPONENTIAL_BASE: Final[float] = 2.0
        RETRY_MAX_DELAY: Final[int] = 60  # seconds

        # Data validation
        MAX_FIELD_NAME_LENGTH: Final[int] = 128
        MAX_TABLE_NAME_LENGTH: Final[int] = 64
        RESERVED_FIELD_NAMES: Final[tuple[str, ...]] = (
            "_sdc_extracted_at", "_sdc_received_at", "_sdc_deleted_at",
            "_sdc_table_version", "_sdc_batched_at", "_sdc_sequence"
        )

        # Performance optimization
        PARALLEL_LOAD_THREADS: Final[int] = 4
        LOAD_TIMEOUT: Final[int] = 300  # 5 minutes
        MEMORY_LIMIT_MB: Final[int] = 512

    class Singer:
        """Singer ecosystem constants for tap and target operations."""

        # Message types
        RECORD_MESSAGE: Final[str] = "RECORD"
        SCHEMA_MESSAGE: Final[str] = "SCHEMA"
        STATE_MESSAGE: Final[str] = "STATE"
        ACTIVATE_VERSION_MESSAGE: Final[str] = "ACTIVATE_VERSION"

        # Replication methods
        FULL_TABLE: Final[str] = "FULL_TABLE"
        INCREMENTAL: Final[str] = "INCREMENTAL"
        LOG_BASED: Final[str] = "LOG_BASED"

        # Stream selection
        INCLUSION_AUTOMATIC: Final[str] = "automatic"
        INCLUSION_AVAILABLE: Final[str] = "available"
        INCLUSION_UNSUPPORTED: Final[str] = "unsupported"

        # Data types
        TYPE_STRING: Final[str] = "string"
        TYPE_INTEGER: Final[str] = "integer"
        TYPE_NUMBER: Final[str] = "number"
        TYPE_BOOLEAN: Final[str] = "boolean"
        TYPE_OBJECT: Final[str] = "object"
        TYPE_ARRAY: Final[str] = "array"
        TYPE_NULL: Final[str] = "null"

    class DataQuality:
        """Data quality and validation constants."""

        # Quality thresholds
        MIN_DATA_QUALITY_SCORE: Final[float] = 0.95  # 95%
        MAX_NULL_PERCENTAGE: Final[float] = 0.1  # 10%
        MAX_DUPLICATE_PERCENTAGE: Final[float] = 0.05  # 5%

        # Anomaly detection
        ANOMALY_DETECTION_WINDOW: Final[int] = 24  # hours
        ANOMALY_THRESHOLD: Final[float] = 2.0  # standard deviations
        MIN_SAMPLES_FOR_DETECTION: Final[int] = 100

        # Data freshness
        MAX_DATA_AGE_HOURS: Final[int] = 24
        STALE_DATA_WARNING_HOURS: Final[int] = 6
        CRITICAL_DATA_AGE_HOURS: Final[int] = 48

class FlextMeltanoService:
    """Example service using FlextMeltanoConstants."""

    def __init__(self):
        self.extractor_config = {
            "batch_size": FlextMeltanoConstants.Extractors.DEFAULT_BATCH_SIZE,
            "discovery_timeout": FlextMeltanoConstants.Extractors.DISCOVERY_TIMEOUT,
            "max_parallel_streams": FlextMeltanoConstants.Extractors.MAX_PARALLEL_STREAMS,
            "connection_pool_size": FlextMeltanoConstants.Extractors.CONNECTION_POOL_SIZE
        }

        self.target_config = {
            "batch_size": FlextMeltanoConstants.Targets.DEFAULT_BATCH_SIZE,
            "max_retries": FlextMeltanoConstants.Targets.MAX_LOAD_RETRIES,
            "parallel_threads": FlextMeltanoConstants.Targets.PARALLEL_LOAD_THREADS,
            "load_timeout": FlextMeltanoConstants.Targets.LOAD_TIMEOUT
        }

    def validate_stream_config(self, stream_config: dict) -> FlextResult[dict]:
        """Validate stream configuration using Meltano constants."""

        # Validate batch size
        batch_size = stream_config.get("batch_size", 0)
        if not (FlextMeltanoConstants.Targets.MIN_BATCH_SIZE <=
               batch_size <=
               FlextMeltanoConstants.Targets.MAX_BATCH_SIZE):
            return FlextResult.fail(
                f"Batch size must be between {FlextMeltanoConstants.Targets.MIN_BATCH_SIZE} "
                f"and {FlextMeltanoConstants.Targets.MAX_BATCH_SIZE}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        # Validate field names
        table_name = stream_config.get("table_name", "")
        if len(table_name) > FlextMeltanoConstants.Targets.MAX_TABLE_NAME_LENGTH:
            return FlextResult.fail(
                f"Table name exceeds maximum length of {FlextMeltanoConstants.Targets.MAX_TABLE_NAME_LENGTH}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        # Check for reserved field names
        fields = stream_config.get("fields", [])
        for field in fields:
            if field.get("name") in FlextMeltanoConstants.Targets.RESERVED_FIELD_NAMES:
                return FlextResult.fail(
                    f"Field name '{field.get('name')}' is reserved",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR
                )

        return FlextResult.ok(stream_config)

    def create_singer_message(self, message_type: str, data: dict) -> dict:
        """Create Singer message using constants."""

        valid_types = [
            FlextMeltanoConstants.Singer.RECORD_MESSAGE,
            FlextMeltanoConstants.Singer.SCHEMA_MESSAGE,
            FlextMeltanoConstants.Singer.STATE_MESSAGE,
            FlextMeltanoConstants.Singer.ACTIVATE_VERSION_MESSAGE
        ]

        if message_type not in valid_types:
            raise ValueError(f"Invalid message type. Must be one of: {valid_types}")

        return {
            "type": message_type,
            "data": data,
            "version": FlextConstants.Core.VERSION,
            "timestamp": "2024-01-01T12:00:00Z"
        }

    def assess_data_quality(self, quality_metrics: dict) -> FlextResult[dict]:
        """Assess data quality using FlextMeltano quality constants."""

        overall_score = quality_metrics.get("overall_score", 0.0)
        null_percentage = quality_metrics.get("null_percentage", 0.0)
        duplicate_percentage = quality_metrics.get("duplicate_percentage", 0.0)

        issues = []

        # Check overall quality score
        if overall_score < FlextMeltanoConstants.DataQuality.MIN_DATA_QUALITY_SCORE:
            issues.append(f"Overall quality score {overall_score:.2%} below minimum {FlextMeltanoConstants.DataQuality.MIN_DATA_QUALITY_SCORE:.2%}")

        # Check null percentage
        if null_percentage > FlextMeltanoConstants.DataQuality.MAX_NULL_PERCENTAGE:
            issues.append(f"Null percentage {null_percentage:.2%} exceeds maximum {FlextMeltanoConstants.DataQuality.MAX_NULL_PERCENTAGE:.2%}")

        # Check duplicate percentage
        if duplicate_percentage > FlextMeltanoConstants.DataQuality.MAX_DUPLICATE_PERCENTAGE:
            issues.append(f"Duplicate percentage {duplicate_percentage:.2%} exceeds maximum {FlextMeltanoConstants.DataQuality.MAX_DUPLICATE_PERCENTAGE:.2%}")

        if issues:
            return FlextResult.fail(
                f"Data quality issues found: {'; '.join(issues)}",
                error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION
            )

        return FlextResult.ok({
            "quality_score": overall_score,
            "status": "passed",
            "issues": []
        })

def test_meltano_constants():
    """Test Meltano-specific constants usage."""

    service = FlextMeltanoService()

    print("Testing Meltano service configuration...")
    print(f"✅ Extractor config: {service.extractor_config}")
    print(f"✅ Target config: {service.target_config}")

    print("\nTesting stream validation...")

    # Valid stream configuration
    valid_stream = {
        "table_name": "users",
        "batch_size": 1000,
        "fields": [
            {"name": "id", "type": "integer"},
            {"name": "name", "type": "string"},
            {"name": "email", "type": "string"}
        ]
    }

    result = service.validate_stream_config(valid_stream)
    if result.success:
        print(f"✅ Stream validation passed: {result.value['table_name']}")
    else:
        print(f"❌ Stream validation failed: {result.error}")

    # Invalid stream - batch size too large
    invalid_stream = {
        "table_name": "orders",
        "batch_size": 15000,  # Exceeds MAX_BATCH_SIZE
        "fields": [{"name": "order_id", "type": "string"}]
    }

    result = service.validate_stream_config(invalid_stream)
    if result.success:
        print(f"Unexpected success: {result.value}")
    else:
        print(f"❌ Expected validation failure: {result.error}")

    print("\nTesting Singer message creation...")

    # Create different Singer messages
    record_message = service.create_singer_message(
        FlextMeltanoConstants.Singer.RECORD_MESSAGE,
        {"stream": "users", "record": {"id": 1, "name": "Alice"}}
    )
    print(f"✅ Record message: {record_message}")

    schema_message = service.create_singer_message(
        FlextMeltanoConstants.Singer.SCHEMA_MESSAGE,
        {"stream": "users", "schema": {"type": "object"}}
    )
    print(f"✅ Schema message: {schema_message}")

    print("\nTesting data quality assessment...")

    # Good quality data
    good_quality = {
        "overall_score": 0.98,
        "null_percentage": 0.02,
        "duplicate_percentage": 0.01
    }

    quality_result = service.assess_data_quality(good_quality)
    if quality_result.success:
        print(f"✅ Data quality passed: {quality_result.value}")
    else:
        print(f"❌ Data quality failed: {quality_result.error}")

    # Poor quality data
    poor_quality = {
        "overall_score": 0.85,  # Below minimum
        "null_percentage": 0.15,  # Too high
        "duplicate_percentage": 0.08  # Too high
    }

    quality_result = service.assess_data_quality(poor_quality)
    if quality_result.success:
        print(f"Unexpected success: {quality_result.value}")
    else:
        print(f"❌ Expected quality failure: {quality_result.error}")

test_meltano_constants()
```

### Step 2: Configuration Management with Environment-Specific Constants

```python
from flext_core import FlextConstants, FlextResult
from enum import StrEnum
from typing import Dict, object, Final

class FlextWebConstants(FlextConstants):
    """Web service constants extending FlextConstants."""

    class HTTP:
        """HTTP-specific configuration constants."""

        # Request handling
        DEFAULT_REQUEST_TIMEOUT: Final[int] = 30
        MAX_REQUEST_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
        MAX_HEADER_SIZE: Final[int] = 8192  # 8KB

        # CORS configuration
        CORS_MAX_AGE: Final[int] = 86400  # 24 hours
        CORS_ALLOWED_METHODS: Final[tuple[str, ...]] = (
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"
        )
        CORS_ALLOWED_HEADERS: Final[tuple[str, ...]] = (
            "Content-Type", "Authorization", "X-Requested-With", "Accept"
        )

        # Security headers
        SECURITY_HEADERS: Final[FlextTypes.Core.Headers] = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }

    class Sessions:
        """Session management constants."""

        # Session configuration
        DEFAULT_SESSION_TIMEOUT: Final[int] = 1800  # 30 minutes
        MAX_SESSION_TIMEOUT: Final[int] = 86400  # 24 hours
        SESSION_COOKIE_NAME: Final[str] = "flext_session"
        CSRF_TOKEN_NAME: Final[str] = "flext_csrf_token"

        # Security settings
        CSRF_TOKEN_LENGTH: Final[int] = 32
        SESSION_ID_LENGTH: Final[int] = 32
        SECURE_COOKIE_ATTRIBUTES: Final[FlextTypes.Core.Dict] = {
            "secure": True,
            "httponly": True,
            "samesite": "strict"
        }

    class RateLimiting:
        """Rate limiting configuration constants."""

        # Default rate limits
        DEFAULT_RATE_LIMIT: Final[int] = 100  # requests per minute
        BURST_RATE_LIMIT: Final[int] = 200
        API_RATE_LIMIT: Final[int] = 1000

        # Window configurations
        RATE_LIMIT_WINDOW: Final[int] = 60  # seconds
        RATE_LIMIT_PRECISION: Final[int] = 10  # sub-windows

        # Rate limit headers
        RATE_LIMIT_HEADERS: Final[FlextTypes.Core.Headers] = {
            "X-RateLimit-Limit": "X-RateLimit-Limit",
            "X-RateLimit-Remaining": "X-RateLimit-Remaining",
            "X-RateLimit-Reset": "X-RateLimit-Reset"
        }

class FlextConfigManager:
    """Configuration management using FlextConstants with environment support."""

    def __init__(self, environment: str = None):
        self.environment = environment or FlextConstants.Config.DEFAULT_ENVIRONMENT
        self.config_cache: Dict[str, object] = {}

    def get_environment_config(self) -> Dict[str, object]:
        """Get environment-specific configuration."""

        base_config = {
            "core": {
                "name": FlextConstants.Core.NAME,
                "version": FlextConstants.Core.VERSION,
                "architecture": FlextConstants.Core.ARCHITECTURE
            },
            "network": {
                "default_timeout": FlextConstants.Network.DEFAULT_TIMEOUT,
                "connection_timeout": FlextConstants.Network.CONNECTION_TIMEOUT
            },
            "performance": {
                "batch_size": FlextConstants.Performance.DEFAULT_BATCH_SIZE,
                "cache_ttl": FlextConstants.Performance.CACHE_TTL,
                "pool_size": FlextConstants.Performance.POOL_SIZE
            }
        }

        # Environment-specific overrides
        if self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION:
            production_overrides = {
                "logging": {
                    "level": FlextConstants.Config.LogLevel.WARNING,
                    "format": "json"
                },
                "validation": {
                    "level": FlextConstants.Config.ValidationLevel.STRICT
                },
                "security": {
                    "enforce_https": True,
                    "session_timeout": FlextWebConstants.Sessions.DEFAULT_SESSION_TIMEOUT,
                    "rate_limit": FlextWebConstants.RateLimiting.API_RATE_LIMIT
                },
                "performance": {
                    "cache_ttl": 3600,  # 1 hour in production
                    "pool_size": 20,  # Larger pool for production
                    "optimization_enabled": True
                }
            }
            self._merge_config(base_config, production_overrides)

        elif self.environment == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT:
            development_overrides = {
                "logging": {
                    "level": FlextConstants.Config.LogLevel.DEBUG,
                    "format": "console"
                },
                "validation": {
                    "level": FlextConstants.Config.ValidationLevel.LOOSE
                },
                "security": {
                    "enforce_https": False,
                    "session_timeout": 3600,  # 1 hour for development
                    "rate_limit": FlextWebConstants.RateLimiting.DEFAULT_RATE_LIMIT * 10  # Relaxed
                },
                "performance": {
                    "cache_ttl": 60,  # Short TTL for development
                    "pool_size": 5,  # Smaller pool for development
                    "optimization_enabled": False
                }
            }
            self._merge_config(base_config, development_overrides)

        elif self.environment == FlextConstants.Config.ConfigEnvironment.TEST:
            test_overrides = {
                "logging": {
                    "level": FlextConstants.Config.LogLevel.ERROR,
                    "format": "minimal"
                },
                "validation": {
                    "level": FlextConstants.Config.ValidationLevel.STRICT
                },
                "security": {
                    "enforce_https": False,
                    "session_timeout": 300,  # 5 minutes for tests
                    "rate_limit": 9999999  # Unlimited for tests
                },
                "performance": {
                    "cache_ttl": 1,  # Very short TTL for tests
                    "pool_size": 1,  # Minimal pool for tests
                    "optimization_enabled": False
                }
            }
            self._merge_config(base_config, test_overrides)

        return base_config

    def _merge_config(self, base: Dict[str, object], overrides: Dict[str, object]) -> None:
        """Recursively merge configuration overrides."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def validate_configuration(self, config: Dict[str, object]) -> FlextResult[Dict[str, object]]:
        """Validate configuration against FlextConstants constraints."""

        # Validate logging level
        log_level = config.get("logging", {}).get("level")
        if log_level:
            valid_levels = [level.value for level in FlextConstants.Config.LogLevel]
            if log_level not in valid_levels:
                return FlextResult.fail(
                    f"Invalid log level '{log_level}'. Valid levels: {valid_levels}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR
                )

        # Validate validation level
        validation_level = config.get("validation", {}).get("level")
        if validation_level:
            valid_validation_levels = [level.value for level in FlextConstants.Config.ValidationLevel]
            if validation_level not in valid_validation_levels:
                return FlextResult.fail(
                    f"Invalid validation level '{validation_level}'. Valid levels: {valid_validation_levels}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR
                )

        # Validate performance settings
        performance_config = config.get("performance", {})

        batch_size = performance_config.get("batch_size", 0)
        if batch_size and not (FlextConstants.Validation.MIN_BATCH_SIZE <=
                              batch_size <=
                              FlextConstants.Validation.MAX_BATCH_SIZE):
            return FlextResult.fail(
                f"Batch size {batch_size} outside valid range "
                f"[{FlextConstants.Validation.MIN_BATCH_SIZE}, {FlextConstants.Validation.MAX_BATCH_SIZE}]",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        pool_size = performance_config.get("pool_size", 0)
        if pool_size and pool_size > FlextConstants.Performance.MAX_CONNECTIONS:
            return FlextResult.fail(
                f"Pool size {pool_size} exceeds maximum {FlextConstants.Performance.MAX_CONNECTIONS}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        # Validate security settings
        security_config = config.get("security", {})
        session_timeout = security_config.get("session_timeout", 0)
        if session_timeout and session_timeout > FlextWebConstants.Sessions.MAX_SESSION_TIMEOUT:
            return FlextResult.fail(
                f"Session timeout {session_timeout} exceeds maximum {FlextWebConstants.Sessions.MAX_SESSION_TIMEOUT}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        return FlextResult.ok(config)

    def get_web_service_config(self) -> Dict[str, object]:
        """Get web service specific configuration."""

        return {
            "http": {
                "request_timeout": FlextWebConstants.HTTP.DEFAULT_REQUEST_TIMEOUT,
                "max_request_size": FlextWebConstants.HTTP.MAX_REQUEST_SIZE,
                "max_header_size": FlextWebConstants.HTTP.MAX_HEADER_SIZE,
                "security_headers": FlextWebConstants.HTTP.SECURITY_HEADERS
            },
            "cors": {
                "max_age": FlextWebConstants.HTTP.CORS_MAX_AGE,
                "allowed_methods": list(FlextWebConstants.HTTP.CORS_ALLOWED_METHODS),
                "allowed_headers": list(FlextWebConstants.HTTP.CORS_ALLOWED_HEADERS)
            },
            "sessions": {
                "timeout": FlextWebConstants.Sessions.DEFAULT_SESSION_TIMEOUT,
                "cookie_name": FlextWebConstants.Sessions.SESSION_COOKIE_NAME,
                "csrf_token_name": FlextWebConstants.Sessions.CSRF_TOKEN_NAME,
                "security_attributes": FlextWebConstants.Sessions.SECURE_COOKIE_ATTRIBUTES
            },
            "rate_limiting": {
                "default_limit": FlextWebConstants.RateLimiting.DEFAULT_RATE_LIMIT,
                "burst_limit": FlextWebConstants.RateLimiting.BURST_RATE_LIMIT,
                "api_limit": FlextWebConstants.RateLimiting.API_RATE_LIMIT,
                "window_seconds": FlextWebConstants.RateLimiting.RATE_LIMIT_WINDOW
            }
        }

def test_configuration_management():
    """Test configuration management with environment-specific settings."""

    print("Testing environment-specific configuration...")

    # Test different environments
    environments = [
        FlextConstants.Config.ConfigEnvironment.DEVELOPMENT,
        FlextConstants.Config.ConfigEnvironment.PRODUCTION,
        FlextConstants.Config.ConfigEnvironment.TEST
    ]

    for env in environments:
        config_manager = FlextConfigManager(env)
        env_config = config_manager.get_environment_config()

        print(f"\n✅ {env.upper()} Configuration:")
        print(f"   Log Level: {env_config['logging']['level']}")
        print(f"   Validation Level: {env_config['validation']['level']}")
        print(f"   Pool Size: {env_config['performance']['pool_size']}")
        print(f"   Cache TTL: {env_config['performance']['cache_ttl']}")
        print(f"   Rate Limit: {env_config['security']['rate_limit']}")

        # Validate configuration
        validation_result = config_manager.validate_configuration(env_config)
        if validation_result.success:
            print(f"   ✅ Configuration validation passed")
        else:
            print(f"   ❌ Configuration validation failed: {validation_result.error}")

    print("\nTesting web service configuration...")

    config_manager = FlextConfigManager(FlextConstants.Config.ConfigEnvironment.PRODUCTION)
    web_config = config_manager.get_web_service_config()

    print(f"✅ Web Service Configuration:")
    print(f"   Request Timeout: {web_config['http']['request_timeout']}s")
    print(f"   Max Request Size: {web_config['http']['max_request_size'] / (1024*1024):.0f}MB")
    print(f"   Session Timeout: {web_config['sessions']['timeout']}s")
    print(f"   Default Rate Limit: {web_config['rate_limiting']['default_limit']} req/min")
    print(f"   CORS Max Age: {web_config['cors']['max_age']}s")

test_configuration_management()
```

---

This comprehensive implementation guide demonstrates how to effectively leverage FlextConstants across all FLEXT ecosystem scenarios, from basic constant usage to advanced library-specific extensions with complete environment-specific configuration management for production-ready applications.
