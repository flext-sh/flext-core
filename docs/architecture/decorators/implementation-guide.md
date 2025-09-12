# FlextDecorators Implementation Guide

**Version**: 0.9.0
**Module**: `flext_core.decorators`
**Target Audience**: Senior Developers, Software Architects, Platform Engineers

## Quick Start

This guide provides step-by-step implementation patterns for applying FlextDecorators across FLEXT ecosystem services, from basic decorator usage to enterprise-grade function enhancement with comprehensive cross-cutting concerns.

**Prerequisite**: Ensure `flext-core` is installed and available in your environment.

---

## 🚀 Basic Implementation

### Step 1: Import and Basic Setup

```python
from flext_core import FlextDecorators, FlextResult, FlextConstants
import time
import random
from typing import Dict, List, Optional

# Basic decorator system configuration
config_result = FlextDecorators.configure_decorators_system({
    "environment": "development",
    "decorator_level": "loose",
    "enable_reliability_decorators": True,
    "enable_validation_decorators": True,
    "enable_performance_monitoring": True,
    "log_level": "DEBUG"
})

if not config_result.success:
    raise Exception(f"Failed to configure decorators: {config_result.error}")

print("✅ FlextDecorators configured successfully")
```

### Step 2: Reliability Layer Implementation

```python
# Safe execution with automatic FlextResult wrapping
@FlextDecorators.Reliability.safe_result
def parse_json_data(json_string: str) -> dict:
    """Parse JSON with automatic error handling."""
    import json
    return json.loads(json_string)  # May raise JSONDecodeError

# Retry functionality with exponential backoff
@FlextDecorators.Reliability.retry(
    max_attempts=3,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError)
)
def connect_to_external_service() -> dict:
    """Connect to external service with retry logic."""
    # Simulate occasional connection failures
    if random.random() < 0.6:  # 60% failure rate for demo
        raise ConnectionError("Network unavailable")

    return {"status": "connected", "service": "external-api"}

# Timeout protection for long-running operations
@FlextDecorators.Reliability.timeout(seconds=5.0)
def long_running_calculation() -> int:
    """Calculation with timeout protection."""
    # Simulate long calculation
    time.sleep(3.0)  # Will complete within timeout
    return 42

# Combined reliability patterns
@FlextDecorators.Reliability.safe_result
@FlextDecorators.Reliability.retry(max_attempts=3)
@FlextDecorators.Reliability.timeout(seconds=10.0)
def robust_external_call(api_endpoint: str) -> dict:
    """External API call with comprehensive reliability."""
    # Simulate external API call
    if random.random() < 0.3:
        raise ConnectionError("API temporarily unavailable")

    return {
        "endpoint": api_endpoint,
        "data": {"result": "success"},
        "timestamp": time.time()
    }

# Test reliability decorators
def test_reliability_patterns():
    """Test reliability decorator patterns."""

    # Test safe_result with success
    result1 = parse_json_data('{"key": "value"}')
    if result1.success:
        print(f"✅ JSON parsed successfully: {result1.value}")

    # Test safe_result with failure
    result2 = parse_json_data("invalid json")
    if not result2.success:
        print(f"❌ JSON parsing failed: {result2.error}")

    # Test retry functionality
    try:
        connection = connect_to_external_service()
        print(f"✅ Connected after retries: {connection}")
    except RuntimeError as e:
        print(f"❌ Connection failed after all retries: {e}")

    # Test timeout protection
    try:
        result = long_running_calculation()
        print(f"✅ Calculation completed: {result}")
    except TimeoutError as e:
        print(f"⏱️ Calculation timed out: {e}")

    # Test combined patterns
    api_result = robust_external_call("/api/v1/data")
    if api_result.success:
        print(f"✅ Robust API call succeeded: {api_result.value}")
    else:
        print(f"❌ Robust API call failed: {api_result.error}")

test_reliability_patterns()
```

### Step 3: Validation Layer Implementation

```python
# Input validation with custom predicates
@FlextDecorators.Validation.validate_input(
    lambda data: isinstance(data, dict) and "user_id" in data and "amount" in data,
    "Input must be dict with user_id and amount fields"
)
def process_payment(payment_data: dict) -> dict:
    """Process payment with input validation."""
    user_id = payment_data["user_id"]
    amount = float(payment_data["amount"])

    return {
        "transaction_id": f"txn_{int(time.time())}",
        "user_id": user_id,
        "amount": amount,
        "status": "processed"
    }

# Type validation with runtime checking
@FlextDecorators.Validation.validate_types(
    arg_types=[str, int],
    return_type=str
)
def format_user_message(name: str, age: int) -> str:
    """Format user message with type validation."""
    return f"User {name} is {age} years old"

# Input sanitization for security
@FlextDecorators.Validation.sanitize_input(
    lambda text: text.strip().lower() if isinstance(text, str) else text
)
@FlextDecorators.Validation.validate_input(
    lambda text: isinstance(text, str) and len(text) > 0,
    "Text must be non-empty string"
)
def process_search_query(query: str) -> dict:
    """Process search query with sanitization and validation."""
    # Input automatically sanitized and validated
    return {
        "query": query,
        "results": f"Search results for: {query}",
        "count": random.randint(1, 100)
    }

# Complex validation with multiple constraints
def validate_order_data(data: object) -> bool:
    """Complex order validation logic."""
    if not isinstance(data, dict):
        return False

    required_fields = ["customer_id", "items", "total_amount"]
    if not all(field in data for field in required_fields):
        return False

    if not isinstance(data["items"], list) or len(data["items"]) == 0:
        return False

    if not isinstance(data["total_amount"], (int, float)) or data["total_amount"] <= 0:
        return False

    return True

@FlextDecorators.Validation.validate_input(
    validate_order_data,
    "Order data must contain customer_id, items (non-empty list), and positive total_amount"
)
def process_customer_order(order_data: dict) -> dict:
    """Process customer order with comprehensive validation."""
    order_id = f"order_{int(time.time())}"

    return {
        "order_id": order_id,
        "customer_id": order_data["customer_id"],
        "item_count": len(order_data["items"]),
        "total": order_data["total_amount"],
        "status": "confirmed"
    }

# Test validation decorators
def test_validation_patterns():
    """Test validation decorator patterns."""

    # Test input validation - success
    try:
        result = process_payment({"user_id": "user123", "amount": 50.00})
        print(f"✅ Payment processed: {result}")
    except ValueError as e:
        print(f"❌ Payment validation failed: {e}")

    # Test input validation - failure
    try:
        result = process_payment({"user_id": "user123"})  # Missing amount
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"❌ Payment validation correctly failed: {e}")

    # Test type validation - success
    try:
        message = format_user_message("Alice", 30)
        print(f"✅ Message formatted: {message}")
    except TypeError as e:
        print(f"❌ Type validation failed: {e}")

    # Test type validation - failure
    try:
        message = format_user_message("Alice", "thirty")  # Wrong type
        print(f"Unexpected success: {message}")
    except TypeError as e:
        print(f"❌ Type validation correctly failed: {e}")

    # Test input sanitization
    result = process_search_query("  Python PROGRAMMING  ")
    print(f"✅ Search query sanitized: '{result['query']}'")  # Should be 'python programming'

    # Test complex validation - success
    valid_order = {
        "customer_id": "cust_123",
        "items": [{"id": 1, "name": "Product A"}],
        "total_amount": 99.99
    }
    try:
        result = process_customer_order(valid_order)
        print(f"✅ Order processed: {result}")
    except ValueError as e:
        print(f"❌ Order validation failed: {e}")

test_validation_patterns()
```

### Step 4: Performance Layer Implementation

```python
# Performance monitoring with thresholds
@FlextDecorators.Performance.monitor(
    threshold=1.0,  # Warn if execution takes >1 second
    log_slow=True,
    collect_metrics=True
)
def database_query_simulation(query: str) -> list[dict]:
    """Database query with performance monitoring."""
    # Simulate variable database response times
    execution_time = random.uniform(0.1, 2.0)
    time.sleep(execution_time)

    return [
        {"id": 1, "name": "Record 1", "query": query},
        {"id": 2, "name": "Record 2", "query": query}
    ]

# Caching with TTL and size limits
@FlextDecorators.Performance.cache(
    max_size=100,   # Maximum 100 cached results
    ttl=300        # 5-minute expiration
)
def expensive_calculation(x: int, y: int) -> dict:
    """Expensive calculation with caching."""
    # Simulate expensive computation
    time.sleep(0.5)

    result = x ** y
    return {
        "input_x": x,
        "input_y": y,
        "result": result,
        "computed_at": time.time()
    }

# Combined performance patterns
@FlextDecorators.Performance.monitor(threshold=0.5, collect_metrics=True)
@FlextDecorators.Performance.cache(max_size=50, ttl=600)  # 10-minute cache
def fetch_user_profile(user_id: str) -> dict:
    """Fetch user profile with monitoring and caching."""
    # Simulate API call or database query
    time.sleep(0.3)

    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "profile": {"role": "standard", "active": True},
        "fetched_at": time.time()
    }

# Rate limiting for API protection
@FlextDecorators.Performance.throttle(calls_per_second=2.0)
def rate_limited_operation(data: str) -> dict:
    """Operation with rate limiting."""
    return {
        "processed": data,
        "timestamp": time.time()
    }

# Test performance decorators
def test_performance_patterns():
    """Test performance decorator patterns."""

    print("Testing performance monitoring...")

    # Test monitoring - fast operation
    result1 = database_query_simulation("SELECT * FROM users LIMIT 10")
    print(f"✅ Fast query completed: {len(result1)} records")

    # Test monitoring - slow operation (will trigger warning)
    result2 = database_query_simulation("SELECT * FROM large_table")
    print(f"⚠️ Slow query completed: {len(result2)} records")

    print("\nTesting caching...")

    # Test caching - first call (cache miss)
    start_time = time.time()
    result1 = expensive_calculation(2, 10)
    first_duration = time.time() - start_time
    print(f"✅ First call (cache miss): {result1['result']} in {first_duration:.3f}s")

    # Test caching - second call (cache hit)
    start_time = time.time()
    result2 = expensive_calculation(2, 10)
    second_duration = time.time() - start_time
    print(f"✅ Second call (cache hit): {result2['result']} in {second_duration:.3f}s")

    print(f"Cache speedup: {first_duration/second_duration:.1f}x faster")

    print("\nTesting user profile caching...")

    # Test combined patterns
    profile1 = fetch_user_profile("user123")
    print(f"✅ Profile fetched: {profile1['name']}")

    profile2 = fetch_user_profile("user123")  # Should be cached
    print(f"✅ Profile cached: {profile2['name']}")

    print("\nTesting rate limiting...")

    # Test rate limiting
    for i in range(3):
        start = time.time()
        result = rate_limited_operation(f"data_{i}")
        duration = time.time() - start
        print(f"✅ Operation {i+1}: processed in {duration:.3f}s")

test_performance_patterns()
```

---

## 🏗️ Advanced Implementation

### Step 1: Observability Layer with Structured Logging

```python
# Execution logging with correlation IDs
@FlextDecorators.Observability.log_execution(
    include_args=True,   # Log function arguments
    include_result=True  # Log function results
)
def process_business_data(business_data: dict) -> dict:
    """Process business data with execution logging."""
    processed_data = {
        "original": business_data,
        "processed": True,
        "processing_time": time.time(),
        "result_count": len(business_data.get("items", []))
    }

    return processed_data

# Security-conscious logging (no sensitive data)
@FlextDecorators.Observability.log_execution(
    include_args=False,  # Don't log sensitive arguments
    include_result=False # Don't log sensitive results
)
def authenticate_user(username: str, password: str, session_data: dict) -> dict:
    """Authenticate user with security-conscious logging."""
    # Simulate authentication logic
    is_valid = len(password) > 6 and "@" in username

    return {
        "authenticated": is_valid,
        "user_id": username.split("@")[0] if is_valid else None,
        "session_id": f"sess_{int(time.time())}" if is_valid else None
    }

# Distributed tracing with correlation IDs
@FlextDecorators.Observability.trace(correlation_id="order_processing")
@FlextDecorators.Observability.log_execution()
def process_order_workflow(order_data: dict) -> dict:
    """Process order with distributed tracing."""

    # Step 1: Validate order
    validation_result = validate_order_step(order_data)

    # Step 2: Process payment
    payment_result = process_payment_step(order_data)

    # Step 3: Create shipment
    shipment_result = create_shipment_step(order_data)

    return {
        "order_id": f"order_{int(time.time())}",
        "validation": validation_result,
        "payment": payment_result,
        "shipment": shipment_result,
        "status": "completed"
    }

def validate_order_step(order_data: dict) -> dict:
    """Order validation step with tracing."""
    return {"status": "validated", "timestamp": time.time()}

def process_payment_step(order_data: dict) -> dict:
    """Payment processing step with tracing."""
    return {"status": "charged", "timestamp": time.time()}

def create_shipment_step(order_data: dict) -> dict:
    """Shipment creation step with tracing."""
    return {"status": "shipped", "timestamp": time.time()}

# Metrics collection for monitoring
@FlextDecorators.Observability.metrics(
    metric_name="api_request_duration",
    tags={"endpoint": "/api/v1/users", "method": "GET"}
)
@FlextDecorators.Observability.log_execution()
def api_endpoint_handler(request_data: dict) -> dict:
    """API endpoint with metrics collection."""
    # Simulate API processing
    time.sleep(random.uniform(0.1, 0.5))

    return {
        "status": "success",
        "data": {"users": ["user1", "user2", "user3"]},
        "request_id": request_data.get("request_id", "unknown")
    }

# Test observability patterns
def test_observability_patterns():
    """Test observability decorator patterns."""

    print("Testing execution logging...")

    # Test business data processing with full logging
    business_data = {
        "customer": "ACME Corp",
        "items": [{"id": 1, "name": "Widget"}],
        "total": 100.00
    }

    result = process_business_data(business_data)
    print(f"✅ Business data processed: {result['result_count']} items")

    # Test authentication with security-conscious logging
    auth_result = authenticate_user(
        "user@example.com",
        "secure_password123",
        {"ip": "192.168.1.100"}
    )
    print(f"✅ Authentication completed: {auth_result['authenticated']}")

    print("\nTesting distributed tracing...")

    # Test order workflow with tracing
    order_data = {
        "customer_id": "cust_456",
        "items": [{"id": 1, "qty": 2}],
        "total_amount": 250.00
    }

    workflow_result = process_order_workflow(order_data)
    print(f"✅ Order workflow completed: {workflow_result['status']}")

    print("\nTesting metrics collection...")

    # Test API endpoint with metrics
    request_data = {"request_id": "req_789", "user_id": "user123"}
    api_result = api_endpoint_handler(request_data)
    print(f"✅ API request processed: {api_result['status']}")

test_observability_patterns()
```

### Step 2: Lifecycle Management and API Evolution

```python
# Deprecation with migration guidance
@FlextDecorators.Lifecycle.deprecated(
    version="1.5.0",
    reason="Use calculate_advanced_tax() for improved accuracy and performance",
    removal_version="2.0.0"
)
def calculate_simple_tax(amount: float) -> float:
    """Legacy tax calculation (deprecated)."""
    return amount * 0.08

# New implementation with enhanced features
@FlextDecorators.Lifecycle.version("1.6.0")
def calculate_advanced_tax(amount: float, tax_region: str = "default") -> dict:
    """Advanced tax calculation with regional support."""

    tax_rates = {
        "default": 0.08,
        "california": 0.0875,
        "new_york": 0.08375,
        "texas": 0.0625
    }

    rate = tax_rates.get(tax_region, tax_rates["default"])
    tax_amount = amount * rate

    return {
        "original_amount": amount,
        "tax_rate": rate,
        "tax_amount": tax_amount,
        "total_amount": amount + tax_amount,
        "region": tax_region
    }

# Experimental API with warnings
@FlextDecorators.Lifecycle.experimental(
    "Machine learning tax prediction - API may change in future versions"
)
@FlextDecorators.Lifecycle.version("1.7.0-beta")
def predict_tax_optimization(financial_data: dict) -> dict:
    """Experimental ML-based tax optimization."""
    # Simulate ML prediction
    optimization_score = random.uniform(0.1, 0.3)

    return {
        "current_tax": financial_data.get("current_tax", 0),
        "optimization_potential": optimization_score,
        "confidence": random.uniform(0.7, 0.95),
        "recommendations": ["Consider tax-deferred investments", "Review deductions"]
    }

# Legacy compatibility layer
@FlextDecorators.Lifecycle.deprecated_alias(
    old_name="get_tax_rate",
    replacement="calculate_advanced_tax(amount, region).tax_rate"
)
def get_tax_rate(region: str = "default") -> float:
    """Legacy tax rate getter (deprecated alias)."""
    tax_rates = {"default": 0.08, "california": 0.0875}
    return tax_rates.get(region, 0.08)

# Test lifecycle management
def test_lifecycle_patterns():
    """Test lifecycle management decorator patterns."""

    print("Testing deprecation warnings...")

    # Test deprecated function (will generate warning)
    old_tax = calculate_simple_tax(100.00)
    print(f"⚠️ Old tax calculation: ${old_tax:.2f}")

    # Test new implementation
    new_tax = calculate_advanced_tax(100.00, "california")
    print(f"✅ New tax calculation: ${new_tax['total_amount']:.2f} (CA)")

    print("\nTesting experimental features...")

    # Test experimental API (will generate warning)
    financial_data = {"income": 75000, "current_tax": 6000}
    optimization = predict_tax_optimization(financial_data)
    print(f"🧪 Tax optimization potential: {optimization['optimization_potential']:.2%}")

    print("\nTesting deprecated aliases...")

    # Test deprecated alias (will generate warning)
    rate = get_tax_rate("california")
    print(f"⚠️ Legacy tax rate: {rate:.4f}")

test_lifecycle_patterns()
```

### Step 3: Enterprise Decorator Composition

```python
# Enterprise decorator for critical business functions
@FlextDecorators.Integration.create_enterprise_decorator(
    # Validation configuration
    with_validation=True,
    validator=lambda data: isinstance(data, dict) and "user_id" in data and "amount" in data,

    # Reliability configuration
    with_retry=True,
    max_retries=5,
    with_timeout=True,
    timeout_seconds=30.0,

    # Performance configuration
    with_caching=True,
    cache_size=200,
    with_monitoring=True,
    monitor_threshold=2.0,

    # Observability configuration
    with_logging=True
)
def process_financial_transaction(transaction_data: dict) -> dict:
    """Critical financial transaction with enterprise-grade protection."""

    user_id = transaction_data["user_id"]
    amount = float(transaction_data["amount"])
    transaction_type = transaction_data.get("type", "payment")

    # Simulate external financial processing
    if random.random() < 0.1:  # 10% chance of temporary failure
        raise ConnectionError("Financial service temporarily unavailable")

    # Process transaction
    transaction_id = f"txn_{int(time.time())}_{random.randint(1000, 9999)}"

    return {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "amount": amount,
        "type": transaction_type,
        "status": "completed",
        "processed_at": time.time()
    }

# Service layer with comprehensive enhancement
@FlextDecorators.Integration.create_enterprise_decorator(
    with_validation=True,
    validator=lambda req: isinstance(req, dict) and "action" in req,
    with_retry=True,
    max_retries=3,
    with_monitoring=True,
    monitor_threshold=1.0,
    with_logging=True
)
def service_request_processor(service_request: dict) -> dict:
    """Service request processor with enterprise enhancements."""

    action = service_request["action"]
    parameters = service_request.get("parameters", {})

    # Simulate service processing
    processing_time = random.uniform(0.1, 1.5)
    time.sleep(processing_time)

    if action == "error_test" and random.random() < 0.3:
        raise RuntimeError("Simulated service error")

    return {
        "request_id": f"req_{int(time.time())}",
        "action": action,
        "result": f"Processed {action} with {len(parameters)} parameters",
        "processing_time": processing_time,
        "status": "success"
    }

# Data processing pipeline with validation and monitoring
@FlextDecorators.Integration.create_enterprise_decorator(
    with_validation=True,
    validator=lambda data: isinstance(data, list) and len(data) > 0,
    with_monitoring=True,
    monitor_threshold=5.0,  # 5 second threshold for large datasets
    with_caching=True,
    cache_size=50,
    with_logging=True
)
def process_data_pipeline(dataset: list) -> dict:
    """Data processing pipeline with comprehensive enhancements."""

    # Simulate data processing
    processed_records = []

    for i, record in enumerate(dataset):
        if isinstance(record, dict):
            processed_record = {
                "id": record.get("id", i),
                "data": record.get("data", "processed"),
                "processed_at": time.time()
            }
            processed_records.append(processed_record)

        # Simulate processing time
        if i % 100 == 0:  # Checkpoint every 100 records
            time.sleep(0.1)

    return {
        "total_records": len(dataset),
        "processed_records": len(processed_records),
        "processing_rate": len(processed_records) / len(dataset),
        "pipeline_id": f"pipeline_{int(time.time())}",
        "status": "completed"
    }

# Test enterprise decorator composition
def test_enterprise_composition():
    """Test enterprise decorator composition patterns."""

    print("Testing enterprise financial transaction...")

    # Test financial transaction with full enterprise stack
    transaction_data = {
        "user_id": "user_789",
        "amount": 1250.50,
        "type": "payment",
        "merchant": "ACME Store"
    }

    try:
        result = process_financial_transaction(transaction_data)
        print(f"✅ Financial transaction completed: {result['transaction_id']}")
        print(f"   Amount: ${result['amount']:.2f}, Status: {result['status']}")
    except Exception as e:
        print(f"❌ Financial transaction failed: {e}")

    print("\nTesting service request processing...")

    # Test service processing with enterprise enhancements
    service_requests = [
        {"action": "create_user", "parameters": {"name": "Alice", "email": "alice@example.com"}},
        {"action": "update_profile", "parameters": {"user_id": "123", "bio": "Updated bio"}},
        {"action": "generate_report", "parameters": {"type": "monthly", "format": "pdf"}}
    ]

    for request in service_requests:
        try:
            result = service_request_processor(request)
            print(f"✅ Service request processed: {result['action']} -> {result['status']}")
        except Exception as e:
            print(f"❌ Service request failed: {request['action']} -> {e}")

    print("\nTesting data pipeline processing...")

    # Test data pipeline with large dataset
    sample_dataset = [
        {"id": i, "data": f"record_{i}", "category": "test"}
        for i in range(250)
    ]

    try:
        pipeline_result = process_data_pipeline(sample_dataset)
        print(f"✅ Data pipeline completed:")
        print(f"   Records: {pipeline_result['processed_records']}/{pipeline_result['total_records']}")
        print(f"   Rate: {pipeline_result['processing_rate']:.2%}")
        print(f"   Pipeline ID: {pipeline_result['pipeline_id']}")
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")

test_enterprise_composition()
```

---

## 🎯 Production Implementation Patterns

### 1. Microservice Endpoint Enhancement

```python
from flext_core import FlextDecorators, FlextResult
from typing import Dict

# API endpoint with comprehensive enterprise enhancements
class UserMicroservice:

    @FlextDecorators.Integration.create_enterprise_decorator(
        # Security and validation
        with_validation=True,
        validator=lambda req: (
            isinstance(req, dict) and
            "user_id" in req and
            len(req.get("user_id", "")) > 0
        ),

        # Reliability patterns
        with_retry=True,
        max_retries=3,
        with_timeout=True,
        timeout_seconds=15.0,

        # Performance optimization
        with_caching=True,
        cache_size=500,  # Cache 500 user profiles
        with_monitoring=True,
        monitor_threshold=1.0,  # Alert if >1 second

        # Observability
        with_logging=True
    )
    def get_user_profile(self, request: Dict[str, object]) -> Dict[str, object]:
        """Get user profile with enterprise-grade enhancements."""

        user_id = request["user_id"]
        include_preferences = request.get("include_preferences", False)

        # Simulate database lookup
        time.sleep(0.3)  # Database query simulation

        profile_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "created_at": "2024-01-15T10:30:00Z",
            "last_login": time.time()
        }

        if include_preferences:
            # Additional query for preferences
            time.sleep(0.2)
            profile_data["preferences"] = {
                "theme": "dark",
                "notifications": True,
                "language": "en"
            }

        return {
            "status": "success",
            "data": profile_data,
            "retrieved_at": time.time()
        }

    @FlextDecorators.Integration.create_enterprise_decorator(
        with_validation=True,
        validator=lambda req: (
            isinstance(req, dict) and
            "user_data" in req and
            isinstance(req["user_data"], dict) and
            "email" in req["user_data"]
        ),
        with_retry=True,
        max_retries=2,  # Fewer retries for write operations
        with_monitoring=True,
        monitor_threshold=2.0,  # More time allowed for writes
        with_logging=True
    )
    def create_user_account(self, request: Dict[str, object]) -> Dict[str, object]:
        """Create user account with validation and monitoring."""

        user_data = request["user_data"]

        # Simulate account creation process
        time.sleep(0.8)  # Database write simulation

        new_user_id = f"user_{int(time.time())}"

        return {
            "status": "created",
            "user_id": new_user_id,
            "email": user_data["email"],
            "created_at": time.time()
        }

# Test microservice with enterprise decorators
def test_microservice_patterns():
    """Test microservice implementation with enterprise decorators."""

    service = UserMicroservice()

    print("Testing user profile retrieval...")

    # Test profile retrieval - success case
    try:
        profile_request = {
            "user_id": "12345",
            "include_preferences": True
        }

        result = service.get_user_profile(profile_request)
        print(f"✅ User profile retrieved: {result['data']['name']}")
        print(f"   Preferences included: {'preferences' in result['data']}")

        # Second call should be cached (faster)
        start_time = time.time()
        cached_result = service.get_user_profile(profile_request)
        cache_duration = time.time() - start_time
        print(f"✅ Cached profile retrieved in {cache_duration:.3f}s")

    except Exception as e:
        print(f"❌ Profile retrieval failed: {e}")

    print("\nTesting user account creation...")

    # Test account creation
    try:
        create_request = {
            "user_data": {
                "name": "Alice Johnson",
                "email": "alice.johnson@example.com",
                "role": "customer"
            }
        }

        result = service.create_user_account(create_request)
        print(f"✅ User account created: {result['user_id']}")
        print(f"   Email: {result['email']}, Status: {result['status']}")

    except Exception as e:
        print(f"❌ Account creation failed: {e}")

    # Test validation failure
    print("\nTesting validation failure...")

    try:
        invalid_request = {"user_id": ""}  # Empty user_id should fail
        result = service.get_user_profile(invalid_request)
        print(f"Unexpected success: {result}")
    except ValueError as e:
        print(f"✅ Validation correctly failed: {e}")

test_microservice_patterns()
```

### 2. ETL Pipeline Enhancement

```python
# ETL operations with comprehensive reliability and monitoring
class DataPipelineService:

    @FlextDecorators.Integration.create_enterprise_decorator(
        with_validation=True,
        validator=lambda config: (
            isinstance(config, dict) and
            "source_connection" in config and
            "query" in config
        ),
        with_retry=True,
        max_retries=5,  # More retries for external data sources
        with_timeout=True,
        timeout_seconds=300,  # 5 minutes for large extracts
        with_monitoring=True,
        monitor_threshold=30.0,  # 30 second threshold for extracts
        with_logging=True
    )
    def extract_data(self, extraction_config: Dict[str, object]) -> Dict[str, object]:
        """Extract data from source with enterprise reliability."""

        source_connection = extraction_config["source_connection"]
        query = extraction_config["query"]
        batch_size = extraction_config.get("batch_size", 1000)

        # Simulate data extraction
        print(f"Extracting from {source_connection}...")

        # Simulate network delays and potential failures
        if random.random() < 0.15:  # 15% chance of transient failure
            raise ConnectionError("Source system temporarily unavailable")

        extraction_time = random.uniform(5.0, 25.0)
        time.sleep(extraction_time)

        # Simulate extracted records
        record_count = random.randint(500, 5000)

        return {
            "status": "extracted",
            "source": source_connection,
            "records_extracted": record_count,
            "batch_size": batch_size,
            "extraction_time": extraction_time,
            "extracted_at": time.time()
        }

    @FlextDecorators.Integration.create_enterprise_decorator(
        with_validation=True,
        validator=lambda data: (
            isinstance(data, dict) and
            "records_extracted" in data and
            data["records_extracted"] > 0
        ),
        with_monitoring=True,
        monitor_threshold=60.0,  # 1 minute threshold for transforms
        with_caching=True,
        cache_size=100,  # Cache transformation rules
        with_logging=True
    )
    def transform_data(self, extraction_result: Dict[str, object]) -> Dict[str, object]:
        """Transform extracted data with validation and caching."""

        records_count = extraction_result["records_extracted"]

        # Simulate data transformation
        print(f"Transforming {records_count} records...")

        # Transform time proportional to record count
        transform_time = (records_count / 1000) * 2.0
        time.sleep(min(transform_time, 10.0))  # Cap at 10 seconds for demo

        # Simulate transformation metrics
        transformed_records = int(records_count * 0.95)  # 95% successful transformation
        rejected_records = records_count - transformed_records

        return {
            "status": "transformed",
            "input_records": records_count,
            "transformed_records": transformed_records,
            "rejected_records": rejected_records,
            "transformation_rate": transformed_records / records_count,
            "transform_time": transform_time,
            "transformed_at": time.time()
        }

    @FlextDecorators.Integration.create_enterprise_decorator(
        with_validation=True,
        validator=lambda data: (
            isinstance(data, dict) and
            "transformed_records" in data and
            data["transformed_records"] > 0
        ),
        with_retry=True,
        max_retries=3,  # Retry for destination failures
        with_timeout=True,
        timeout_seconds=180,  # 3 minutes for loads
        with_monitoring=True,
        monitor_threshold=45.0,  # 45 second threshold for loads
        with_logging=True
    )
    def load_data(self, transformation_result: Dict[str, object]) -> Dict[str, object]:
        """Load transformed data to destination with reliability."""

        records_count = transformation_result["transformed_records"]

        # Simulate data loading
        print(f"Loading {records_count} records to destination...")

        # Simulate potential destination failures
        if random.random() < 0.1:  # 10% chance of load failure
            raise ConnectionError("Destination system temporarily unavailable")

        # Load time proportional to record count
        load_time = (records_count / 1000) * 1.5
        time.sleep(min(load_time, 8.0))  # Cap at 8 seconds for demo

        # Simulate successful load
        loaded_records = int(records_count * 0.98)  # 98% successful load
        failed_records = records_count - loaded_records

        return {
            "status": "loaded",
            "input_records": records_count,
            "loaded_records": loaded_records,
            "failed_records": failed_records,
            "load_success_rate": loaded_records / records_count,
            "load_time": load_time,
            "loaded_at": time.time()
        }

# Test ETL pipeline with enterprise decorators
def test_etl_pipeline():
    """Test ETL pipeline with enterprise decorator enhancements."""

    pipeline = DataPipelineService()

    print("Starting ETL pipeline with enterprise enhancements...\n")

    # Configure extraction
    extraction_config = {
        "source_connection": "postgresql://source-db:5432/analytics",
        "query": "SELECT * FROM user_events WHERE created_at > '2024-01-01'",
        "batch_size": 2000
    }

    try:
        # Extract phase
        print("Phase 1: Data Extraction")
        extract_result = pipeline.extract_data(extraction_config)
        print(f"✅ Extracted {extract_result['records_extracted']} records")
        print(f"   Extraction time: {extract_result['extraction_time']:.2f}s")

        # Transform phase
        print("\nPhase 2: Data Transformation")
        transform_result = pipeline.transform_data(extract_result)
        print(f"✅ Transformed {transform_result['transformed_records']} records")
        print(f"   Transformation rate: {transform_result['transformation_rate']:.1%}")
        print(f"   Transform time: {transform_result['transform_time']:.2f}s")

        # Load phase
        print("\nPhase 3: Data Loading")
        load_result = pipeline.load_data(transform_result)
        print(f"✅ Loaded {load_result['loaded_records']} records")
        print(f"   Load success rate: {load_result['load_success_rate']:.1%}")
        print(f"   Load time: {load_result['load_time']:.2f}s")

        # Pipeline summary
        total_time = (
            extract_result['extraction_time'] +
            transform_result['transform_time'] +
            load_result['load_time']
        )

        print(f"\n🎉 ETL Pipeline completed successfully!")
        print(f"   Total records processed: {load_result['loaded_records']}")
        print(f"   Total processing time: {total_time:.2f}s")
        print(f"   Records per second: {load_result['loaded_records']/total_time:.1f}")

    except Exception as e:
        print(f"❌ ETL Pipeline failed: {e}")
        print("Enterprise decorators provided comprehensive error handling and logging")

test_etl_pipeline()
```

### 3. Configuration Management

```python
# Environment-specific decorator configuration
def configure_decorators_for_environment():
    """Configure decorators based on deployment environment."""

    import os

    environment = os.getenv("ENVIRONMENT", "development")
    print(f"Configuring decorators for {environment} environment...")

    # Create environment-specific configuration
    config_result = FlextDecorators.create_environment_decorators_config(environment)

    if config_result.success:
        config = config_result.value
        print(f"✅ Decorator configuration loaded:")
        print(f"   Environment: {config['environment']}")
        print(f"   Decorator level: {config['decorator_level']}")
        print(f"   Performance monitoring: {config.get('enable_performance_monitoring', False)}")
        print(f"   Observability: {config.get('enable_observability_decorators', False)}")
        print(f"   Caching enabled: {config.get('decorator_caching_enabled', False)}")
        print(f"   Timeout enabled: {config.get('decorator_timeout_enabled', False)}")
        print(f"   Max retry attempts: {config.get('decorator_retry_max_attempts', 0)}")

        return config
    else:
        print(f"❌ Failed to configure decorators: {config_result.error}")
        return None

# Performance optimization based on workload
def optimize_decorator_performance():
    """Optimize decorator performance for high-throughput scenarios."""

    print("Optimizing decorator performance for production workload...")

    perf_config = {
        "performance_level": "high",
        "max_concurrent_decorators": 200,
        "decorator_cache_size": 1000,
        "reliability_optimization": True,
        "validation_optimization": True
    }

    optimization_result = FlextDecorators.optimize_decorators_performance(perf_config)

    if optimization_result.success:
        optimized = optimization_result.value
        print(f"✅ Performance optimization completed:")
        print(f"   Performance level: {optimized['performance_level']}")
        print(f"   Expected throughput: {optimized['expected_throughput_decorators_per_second']} ops/sec")
        print(f"   Target overhead: {optimized['target_decorator_overhead_ms']}ms per decorator")
        print(f"   Memory efficiency target: {optimized['memory_efficiency_target']:.1%}")
        print(f"   Decorator cache size: {optimized['decorator_cache_size']}")
        print(f"   Object pooling enabled: {optimized.get('enable_object_pooling', False)}")

        return optimized
    else:
        print(f"❌ Performance optimization failed: {optimization_result.error}")
        return None

# Test configuration patterns
print("=== Decorator Configuration Management ===\n")
env_config = configure_decorators_for_environment()
print()
perf_config = optimize_decorator_performance()
```

---

This comprehensive implementation guide demonstrates how to effectively leverage FlextDecorators across all enterprise scenarios, from basic reliability patterns to complex ETL pipelines with complete observability, validation, and performance optimization for production-ready applications.
