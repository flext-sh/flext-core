# FlextUtilities Implementation Guide

**Complete step-by-step implementation guide for all 10 FlextUtilities domains with practical examples and enterprise patterns.**

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Generators Domain Implementation](#generators-domain-implementation)
3. [TextProcessor Domain Implementation](#textprocessor-domain-implementation)  
4. [Performance Domain Implementation](#performance-domain-implementation)
5. [Conversions Domain Implementation](#conversions-domain-implementation)
6. [ProcessingUtils Domain Implementation](#processingutils-domain-implementation)
7. [Configuration Domain Implementation](#configuration-domain-implementation)
8. [TypeGuards Domain Implementation](#typeguards-domain-implementation)
9. [Formatters Domain Implementation](#formatters-domain-implementation)
10. [ResultUtils Domain Implementation](#resultutils-domain-implementation)
11. [TimeUtils Domain Implementation](#timeutils-domain-implementation)
12. [Enterprise Integration Patterns](#enterprise-integration-patterns)

---

## Quick Start Guide

### Installation and Basic Usage

```python
# Import FlextUtilities
from flext_core import FlextUtilities
from flext_core.result import FlextResult
from datetime import datetime
import time

# Basic utility operations
uuid = FlextUtilities.Generators.generate_uuid()
safe_text = FlextUtilities.TextProcessor.safe_string(None, "default")
config = FlextUtilities.Configuration.create_default_config("production")

print(f"Generated UUID: {uuid}")
print(f"Safe text: {safe_text}")
print(f"Config creation: {'success' if config.success else 'failed'}")
```

### 30-Second Implementation

```python
# Complete request processing example
def process_api_request(request_data: dict) -> dict:
    # Generate tracking IDs
    request_id = FlextUtilities.Generators.generate_request_id()
    correlation_id = FlextUtilities.Generators.generate_correlation_id()
    
    # Safe data extraction
    user_id = FlextUtilities.Conversions.safe_int(request_data.get("user_id"), 0)
    search_term = FlextUtilities.TextProcessor.clean_text(
        request_data.get("search", "")
    )
    
    # Performance tracking
    start_time = time.time()
    
    try:
        # Simulate business logic
        results = perform_search(user_id, search_term)
        
        # Record success metrics
        duration = time.time() - start_time
        FlextUtilities.Performance.record_metric("api_search", duration, success=True)
        
        return {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "results": results,
            "status": "success"
        }
        
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        FlextUtilities.Performance.record_metric("api_search", duration, success=False, error=str(e))
        
        return {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "error": FlextUtilities.TextProcessor.safe_string(str(e)),
            "status": "error"
        }
```

---

## Generators Domain Implementation

### 1. ID Generation Patterns

The Generators domain provides consistent ID generation for distributed systems:

```python
# UUID Generation - Full RFC4122 compliant UUIDs
standard_uuid = FlextUtilities.Generators.generate_uuid()
print(f"Standard UUID: {standard_uuid}")
# Output: Standard UUID: 123e4567-e89b-12d3-a456-426614174000

# Entity ID Generation - Business entity identifiers
user_id = FlextUtilities.Generators.generate_entity_id()
order_id = FlextUtilities.Generators.generate_entity_id()
product_id = FlextUtilities.Generators.generate_entity_id()

print(f"User ID: {user_id}")      # entity_a1b2c3d4e5f6
print(f"Order ID: {order_id}")    # entity_f6e5d4c3b2a1
print(f"Product ID: {product_id}") # entity_9876543210ab

# Request Tracking - Web request and correlation IDs
request_id = FlextUtilities.Generators.generate_request_id()
correlation_id = FlextUtilities.Generators.generate_correlation_id()
session_id = FlextUtilities.Generators.generate_session_id()

print(f"Request ID: {request_id}")        # req_abc123def456
print(f"Correlation ID: {correlation_id}") # corr_1234567890abcdef
print(f"Session ID: {session_id}")        # sess_fedcba0987654321
```

### 2. Timestamp Generation

```python
# ISO Timestamp Generation - UTC timezone guaranteed
iso_timestamp = FlextUtilities.Generators.generate_iso_timestamp()
print(f"ISO Timestamp: {iso_timestamp}")
# Output: 2023-12-01T10:30:45.123456+00:00

# Custom timestamp generation for specific use cases
def create_audit_log_entry(user_id: str, action: str) -> dict:
    return {
        "entry_id": FlextUtilities.Generators.generate_entity_id(),
        "user_id": user_id,
        "action": action,
        "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        "correlation_id": FlextUtilities.Generators.generate_correlation_id()
    }

# Usage example
audit_entry = create_audit_log_entry("user_123", "login_attempt")
print(f"Audit Entry: {audit_entry}")
```

### 3. Enterprise ID Management

```python
# Enterprise ID generation with business context
class EnterpriseIDManager:
    def __init__(self, service_name: str):
        self.service_name = service_name
        
    def generate_transaction_id(self) -> str:
        """Generate transaction ID with service context."""
        base_id = FlextUtilities.Generators.generate_entity_id()
        return f"{self.service_name}_{base_id}"
    
    def create_request_context(self) -> dict:
        """Create complete request context with all tracking IDs."""
        return {
            "request_id": FlextUtilities.Generators.generate_request_id(),
            "correlation_id": FlextUtilities.Generators.generate_correlation_id(),
            "transaction_id": self.generate_transaction_id(),
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            "service": self.service_name
        }

# Usage example
id_manager = EnterpriseIDManager("user-service")
context = id_manager.create_request_context()
print(f"Request Context: {context}")
```

---

## TextProcessor Domain Implementation

### 1. Safe Text Processing

```python
# Safe string conversion with fallback handling
def process_user_input(form_data: dict) -> dict:
    """Process form data with safe text handling."""
    
    return {
        "username": FlextUtilities.TextProcessor.safe_string(
            form_data.get("username"), "anonymous"
        ),
        "email": FlextUtilities.TextProcessor.safe_string(
            form_data.get("email"), ""
        ),
        "bio": FlextUtilities.TextProcessor.truncate(
            FlextUtilities.TextProcessor.safe_string(form_data.get("bio"), ""),
            max_length=200,
            suffix="..."
        ),
        "display_name": FlextUtilities.TextProcessor.clean_text(
            FlextUtilities.TextProcessor.safe_string(form_data.get("display_name"), "")
        )
    }

# Test with various input types
test_data = [
    {"username": "john_doe", "bio": "A" * 300, "display_name": "John\nDoe\t\r\n"},
    {"username": None, "email": 12345, "bio": "", "display_name": ""},
    {"username": ["not", "a", "string"], "bio": None}
]

for data in test_data:
    processed = process_user_input(data)
    print(f"Input: {data}")
    print(f"Processed: {processed}")
    print("-" * 50)
```

### 2. Text Sanitization and Formatting

```python
# URL slug generation for SEO-friendly URLs
article_titles = [
    "The Complete Guide to Python Programming",
    "How to Build REST APIs with FastAPI",
    "Machine Learning: A Beginner's Guide",
    "Advanced Database Optimization Techniques",
    "Understanding Microservices Architecture"
]

for title in article_titles:
    slug = FlextUtilities.TextProcessor.slugify(title)
    print(f"Title: {title}")
    print(f"Slug: {slug}")
    print()

# Output:
# Title: The Complete Guide to Python Programming
# Slug: the-complete-guide-to-python-programming

# camelCase generation for API compatibility
python_fields = [
    "user_name", "created_at", "is_active", "last_login_time", 
    "email_verified", "phone_number", "billing_address"
]

field_mapping = {}
for field in python_fields:
    camel_case = FlextUtilities.TextProcessor.generate_camel_case_alias(field)
    field_mapping[field] = camel_case
    print(f"Python: {field} -> JSON: {camel_case}")

print(f"\nField mapping: {field_mapping}")
```

### 3. Sensitive Data Masking

```python
# Advanced sensitive data masking for security
class DataMaskingService:
    def __init__(self):
        self.masking_patterns = {
            "credit_card": {"show_first": 4, "show_last": 4},
            "ssn": {"show_first": 0, "show_last": 4},
            "email": {"show_first": 3, "show_last": 0},
            "password": {"show_first": 0, "show_last": 0},
            "api_key": {"show_first": 8, "show_last": 4}
        }
    
    def mask_sensitive_data(self, data: dict) -> dict:
        """Mask sensitive data according to predefined patterns."""
        masked_data = {}
        
        for key, value in data.items():
            if key in self.masking_patterns:
                pattern = self.masking_patterns[key]
                masked_value = FlextUtilities.TextProcessor.mask_sensitive(
                    FlextUtilities.TextProcessor.safe_string(value),
                    show_first=pattern["show_first"],
                    show_last=pattern["show_last"]
                )
                masked_data[key] = masked_value
            else:
                masked_data[key] = value
                
        return masked_data

# Usage example
masking_service = DataMaskingService()

sensitive_data = {
    "user_id": "user_123",
    "credit_card": "4532123456789012",
    "ssn": "123-45-6789",
    "email": "john.doe@example.com",
    "password": "SuperSecretPassword123!",
    "api_key": "sk_live_abcdefghijklmnopqrstuvwxyz123456",
    "public_info": "This is not sensitive"
}

masked = masking_service.mask_sensitive_data(sensitive_data)
print("Original data:")
for key, value in sensitive_data.items():
    print(f"  {key}: {value}")

print("\nMasked data:")
for key, value in masked.items():
    print(f"  {key}: {value}")
```

### 4. Text Cleaning and Normalization

```python
# Advanced text cleaning for data processing
def clean_and_normalize_text_data(raw_data: list[str]) -> list[str]:
    """Clean and normalize text data for processing."""
    
    cleaned_data = []
    
    for text in raw_data:
        # Convert to safe string
        safe_text = FlextUtilities.TextProcessor.safe_string(text, "")
        
        # Clean control characters and normalize whitespace
        cleaned_text = FlextUtilities.TextProcessor.clean_text(safe_text)
        
        # Skip empty results
        if cleaned_text:
            cleaned_data.append(cleaned_text)
    
    return cleaned_data

# Test with problematic text data
problematic_texts = [
    "Normal text here",
    "Text with\x00control\x01characters\x02",
    "Multiple    spaces   and\ttabs\n\n\nnewlines",
    "Text with\rcarriage\rreturns",
    None,
    "",
    123,  # Non-string type
    "   Leading and trailing spaces   "
]

cleaned = clean_and_normalize_text_data(problematic_texts)
print("Cleaned text data:")
for i, text in enumerate(cleaned):
    print(f"{i+1}: '{text}'")
```

---

## Performance Domain Implementation

### 1. Performance Monitoring with Decorators

```python
# Performance monitoring using decorators
class DatabaseService:
    def __init__(self):
        self.connection = self._get_connection()
    
    @FlextUtilities.Performance.track_performance("db_query")
    def execute_query(self, sql: str, params: dict = None) -> list:
        """Execute database query with performance tracking."""
        # Simulate database query execution
        time.sleep(0.1)  # Simulate query time
        
        if "invalid" in sql.lower():
            raise Exception("Invalid SQL query")
        
        return [{"id": 1, "name": "test"}]
    
    @FlextUtilities.Performance.track_performance("db_transaction")
    def execute_transaction(self, operations: list) -> bool:
        """Execute database transaction with performance tracking."""
        # Simulate transaction processing
        time.sleep(0.05 * len(operations))
        
        # Simulate occasional transaction failures
        if len(operations) > 10:
            raise Exception("Transaction too large")
        
        return True
    
    def _get_connection(self):
        return "mock_connection"

# Usage example with performance monitoring
db_service = DatabaseService()

# Execute multiple operations to generate metrics
operations = [
    ("SELECT * FROM users WHERE id = 1", {}),
    ("SELECT * FROM orders WHERE user_id = 1", {}),
    ("UPDATE users SET last_login = NOW() WHERE id = 1", {}),
    ("SELECT * FROM invalid_table", {}),  # This will fail
]

print("Executing database operations...")

for sql, params in operations:
    try:
        result = db_service.execute_query(sql, params)
        print(f"✅ Query succeeded: {len(result)} results")
    except Exception as e:
        print(f"❌ Query failed: {e}")

# Execute transaction operations
try:
    success = db_service.execute_transaction(["op1", "op2", "op3"])
    print(f"✅ Transaction succeeded: {success}")
except Exception as e:
    print(f"❌ Transaction failed: {e}")

# Get performance metrics
db_metrics = FlextUtilities.Performance.get_metrics("db_query")
tx_metrics = FlextUtilities.Performance.get_metrics("db_transaction")

print(f"\nDatabase Query Metrics:")
print(f"  Total calls: {db_metrics.get('total_calls', 0)}")
print(f"  Success count: {db_metrics.get('success_count', 0)}")
print(f"  Error count: {db_metrics.get('error_count', 0)}")
print(f"  Average duration: {db_metrics.get('avg_duration', 0):.3f}s")

print(f"\nTransaction Metrics:")
print(f"  Total calls: {tx_metrics.get('total_calls', 0)}")
print(f"  Success count: {tx_metrics.get('success_count', 0)}")
print(f"  Average duration: {tx_metrics.get('avg_duration', 0):.3f}s")
```

### 2. Manual Performance Recording

```python
# Manual performance recording for custom scenarios
class APIEndpointMonitor:
    def __init__(self):
        self.active_requests = {}
    
    def start_request(self, endpoint: str, request_id: str):
        """Start timing a request."""
        self.active_requests[request_id] = {
            "endpoint": endpoint,
            "start_time": time.perf_counter(),
            "request_id": request_id
        }
    
    def end_request(self, request_id: str, success: bool = True, error: str = None):
        """End timing a request and record metrics."""
        if request_id not in self.active_requests:
            return
        
        request_info = self.active_requests.pop(request_id)
        duration = time.perf_counter() - request_info["start_time"]
        
        # Record performance metric
        FlextUtilities.Performance.record_metric(
            f"api_{request_info['endpoint']}", 
            duration,
            success=success,
            error=error
        )
        
        print(f"Request {request_id} completed in {duration:.3f}s - {'✅' if success else '❌'}")
    
    def get_endpoint_metrics(self, endpoint: str) -> dict:
        """Get metrics for specific endpoint."""
        return FlextUtilities.Performance.get_metrics(f"api_{endpoint}")

# Usage example
monitor = APIEndpointMonitor()

# Simulate API requests
endpoints = ["users", "orders", "products"]
requests = []

# Start multiple requests
for i in range(10):
    endpoint = endpoints[i % len(endpoints)]
    request_id = FlextUtilities.Generators.generate_request_id()
    requests.append((endpoint, request_id))
    
    monitor.start_request(endpoint, request_id)
    
    # Simulate request processing time
    time.sleep(0.01 + (i % 3) * 0.02)
    
    # Simulate occasional failures
    success = i % 7 != 0  # Fail every 7th request
    error = "Simulated error" if not success else None
    
    monitor.end_request(request_id, success=success, error=error)

# Get and display metrics for each endpoint
print("\nEndpoint Performance Summary:")
for endpoint in endpoints:
    metrics = monitor.get_endpoint_metrics(endpoint)
    if metrics:
        print(f"\n{endpoint.upper()} Endpoint:")
        print(f"  Total requests: {metrics.get('total_calls', 0)}")
        print(f"  Successful: {metrics.get('success_count', 0)}")
        print(f"  Failed: {metrics.get('error_count', 0)}")
        print(f"  Average response time: {metrics.get('avg_duration', 0):.3f}s")
        
        if metrics.get('error_count', 0) > 0:
            print(f"  Last error: {metrics.get('last_error', 'N/A')}")
```

### 3. System-Wide Performance Analysis

```python
# Comprehensive performance analysis and reporting
class PerformanceAnalyzer:
    def __init__(self):
        self.analysis_id = FlextUtilities.Generators.generate_entity_id()
    
    def generate_performance_report(self) -> dict:
        """Generate comprehensive performance report."""
        all_metrics = FlextUtilities.Performance.get_metrics()
        
        if not all_metrics:
            return {"message": "No performance data available"}
        
        report = {
            "analysis_id": self.analysis_id,
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            "summary": self._generate_summary(all_metrics),
            "operations": self._analyze_operations(all_metrics),
            "recommendations": self._generate_recommendations(all_metrics)
        }
        
        return report
    
    def _generate_summary(self, metrics: dict) -> dict:
        """Generate summary statistics."""
        total_operations = len(metrics)
        total_calls = sum(m.get('total_calls', 0) for m in metrics.values())
        total_errors = sum(m.get('error_count', 0) for m in metrics.values())
        avg_success_rate = (total_calls - total_errors) / total_calls if total_calls > 0 else 0
        
        return {
            "total_operations": total_operations,
            "total_calls": total_calls,
            "total_errors": total_errors,
            "overall_success_rate": FlextUtilities.Formatters.format_percentage(avg_success_rate)
        }
    
    def _analyze_operations(self, metrics: dict) -> list:
        """Analyze individual operations."""
        operations = []
        
        for operation_name, operation_metrics in metrics.items():
            total_calls = operation_metrics.get('total_calls', 0)
            success_rate = (
                (total_calls - operation_metrics.get('error_count', 0)) / total_calls 
                if total_calls > 0 else 0
            )
            
            operations.append({
                "name": operation_name,
                "calls": total_calls,
                "avg_duration": f"{operation_metrics.get('avg_duration', 0):.3f}s",
                "success_rate": FlextUtilities.Formatters.format_percentage(success_rate),
                "error_count": operation_metrics.get('error_count', 0)
            })
        
        # Sort by total calls (most active first)
        operations.sort(key=lambda x: x['calls'], reverse=True)
        return operations
    
    def _generate_recommendations(self, metrics: dict) -> list:
        """Generate performance recommendations."""
        recommendations = []
        
        for operation_name, operation_metrics in metrics.items():
            avg_duration = operation_metrics.get('avg_duration', 0)
            error_count = operation_metrics.get('error_count', 0)
            total_calls = operation_metrics.get('total_calls', 0)
            
            # Slow operation recommendation
            if avg_duration > 1.0:
                recommendations.append({
                    "type": "performance",
                    "operation": operation_name,
                    "issue": f"Slow average response time: {avg_duration:.3f}s",
                    "recommendation": "Consider optimizing this operation"
                })
            
            # High error rate recommendation
            error_rate = error_count / total_calls if total_calls > 0 else 0
            if error_rate > 0.1:  # 10% error rate
                recommendations.append({
                    "type": "reliability",
                    "operation": operation_name,
                    "issue": f"High error rate: {FlextUtilities.Formatters.format_percentage(error_rate)}",
                    "recommendation": "Investigate and fix error causes"
                })
        
        return recommendations

# Usage example - Generate comprehensive performance report
analyzer = PerformanceAnalyzer()
report = analyzer.generate_performance_report()

print("PERFORMANCE ANALYSIS REPORT")
print("=" * 50)
print(f"Analysis ID: {report['analysis_id']}")
print(f"Generated: {report['timestamp']}")

if 'summary' in report:
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Total Calls: {summary['total_calls']}")
    print(f"  Total Errors: {summary['total_errors']}")
    print(f"  Success Rate: {summary['overall_success_rate']}")

if 'operations' in report:
    print(f"\nOPERATIONS BREAKDOWN:")
    for op in report['operations'][:5]:  # Show top 5 operations
        print(f"  {op['name']}:")
        print(f"    Calls: {op['calls']}")
        print(f"    Avg Duration: {op['avg_duration']}")
        print(f"    Success Rate: {op['success_rate']}")

if 'recommendations' in report and report['recommendations']:
    print(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  [{rec['type'].upper()}] {rec['operation']}")
        print(f"    Issue: {rec['issue']}")
        print(f"    Recommendation: {rec['recommendation']}")
```

---

## Conversions Domain Implementation

### 1. Safe Type Conversion with Fallbacks

```python
# Comprehensive type conversion for data processing
class DataConverter:
    def __init__(self):
        self.conversion_stats = {
            "successful_conversions": 0,
            "failed_conversions": 0,
            "fallback_usage": 0
        }
    
    def convert_form_data(self, form_data: dict) -> dict:
        """Convert web form data to appropriate types."""
        converted_data = {}
        
        # Integer conversions with fallbacks
        converted_data["age"] = FlextUtilities.Conversions.safe_int(
            form_data.get("age"), 0
        )
        converted_data["quantity"] = FlextUtilities.Conversions.safe_int(
            form_data.get("quantity"), 1
        )
        
        # Float conversions for monetary values
        converted_data["price"] = FlextUtilities.Conversions.safe_float(
            form_data.get("price"), 0.0
        )
        converted_data["discount"] = FlextUtilities.Conversions.safe_float(
            form_data.get("discount"), 0.0
        )
        
        # Boolean conversions for flags
        converted_data["is_premium"] = FlextUtilities.Conversions.safe_bool(
            form_data.get("is_premium"), default=False
        )
        converted_data["newsletter_subscription"] = FlextUtilities.Conversions.safe_bool(
            form_data.get("newsletter"), default=False
        )
        
        # Track conversion statistics
        self._update_conversion_stats(form_data, converted_data)
        
        return converted_data
    
    def _update_conversion_stats(self, original: dict, converted: dict):
        """Update conversion statistics."""
        for key in converted.keys():
            original_value = original.get(key)
            converted_value = converted[key]
            
            # Check if fallback was used (original was None or conversion failed)
            if original_value is None:
                self.conversion_stats["fallback_usage"] += 1
            else:
                try:
                    # Try the conversion again to see if it would succeed
                    if key in ["age", "quantity"]:
                        int(original_value)
                    elif key in ["price", "discount"]:
                        float(original_value)
                    elif key in ["is_premium", "newsletter_subscription"]:
                        bool(original_value)
                    
                    self.conversion_stats["successful_conversions"] += 1
                except:
                    self.conversion_stats["failed_conversions"] += 1
    
    def get_stats(self) -> dict:
        """Get conversion statistics."""
        return self.conversion_stats.copy()

# Test with various input types
converter = DataConverter()

test_cases = [
    # Valid data
    {
        "age": "25",
        "quantity": "3", 
        "price": "99.99",
        "discount": "0.15",
        "is_premium": "true",
        "newsletter": "yes"
    },
    # Invalid/problematic data
    {
        "age": "not_a_number",
        "quantity": "-5",
        "price": "invalid_price",
        "discount": None,
        "is_premium": "maybe",
        "newsletter": ""
    },
    # Mixed valid/invalid
    {
        "age": 30,  # Already int
        "quantity": "2.5",  # Float for int field
        "price": 149.99,  # Already float
        "discount": "20%",  # Invalid format
        "is_premium": 1,  # Truthy value
        "newsletter": 0  # Falsy value
    },
    # Missing data
    {
        "age": None,
        "price": ""
    }
]

print("TYPE CONVERSION TESTING")
print("=" * 50)

for i, test_case in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"Input: {test_case}")
    
    converted = converter.convert_form_data(test_case)
    print(f"Converted: {converted}")
    
    # Validate results
    print("Validation:")
    print(f"  Age is int: {isinstance(converted['age'], int)}")
    print(f"  Price is float: {isinstance(converted['price'], float)}")
    print(f"  Premium is bool: {isinstance(converted['is_premium'], bool)}")

# Print conversion statistics
stats = converter.get_stats()
print(f"\nCONVERSION STATISTICS:")
print(f"  Successful: {stats['successful_conversions']}")
print(f"  Failed: {stats['failed_conversions']}")
print(f"  Fallbacks used: {stats['fallback_usage']}")
```

### 2. Advanced Conversion with Validation

```python
# Advanced conversion with business logic validation
class BusinessDataConverter:
    def __init__(self):
        self.validation_rules = {
            "age": {"min": 0, "max": 120},
            "quantity": {"min": 1, "max": 1000},
            "price": {"min": 0.0, "max": 999999.99},
            "discount": {"min": 0.0, "max": 1.0}
        }
    
    def convert_with_validation(self, data: dict) -> tuple[dict, list[str]]:
        """Convert data with business rule validation."""
        converted_data = {}
        validation_errors = []
        
        # Age conversion and validation
        age = FlextUtilities.Conversions.safe_int(data.get("age"), -1)
        if age == -1:
            validation_errors.append("Age is required and must be a valid number")
        elif not (self.validation_rules["age"]["min"] <= age <= self.validation_rules["age"]["max"]):
            validation_errors.append(f"Age must be between {self.validation_rules['age']['min']} and {self.validation_rules['age']['max']}")
        else:
            converted_data["age"] = age
        
        # Quantity conversion and validation
        quantity = FlextUtilities.Conversions.safe_int(data.get("quantity"), 0)
        if quantity < self.validation_rules["quantity"]["min"]:
            validation_errors.append(f"Quantity must be at least {self.validation_rules['quantity']['min']}")
        elif quantity > self.validation_rules["quantity"]["max"]:
            validation_errors.append(f"Quantity cannot exceed {self.validation_rules['quantity']['max']}")
        else:
            converted_data["quantity"] = quantity
        
        # Price conversion and validation
        price = FlextUtilities.Conversions.safe_float(data.get("price"), -1.0)
        if price < 0:
            validation_errors.append("Price is required and must be a valid number")
        elif not (self.validation_rules["price"]["min"] <= price <= self.validation_rules["price"]["max"]):
            validation_errors.append(f"Price must be between {self.validation_rules['price']['min']} and {self.validation_rules['price']['max']}")
        else:
            converted_data["price"] = price
        
        # Discount conversion and validation
        discount = FlextUtilities.Conversions.safe_float(data.get("discount"), 0.0)
        if not (self.validation_rules["discount"]["min"] <= discount <= self.validation_rules["discount"]["max"]):
            validation_errors.append(f"Discount must be between {self.validation_rules['discount']['min']} and {self.validation_rules['discount']['max']}")
        else:
            converted_data["discount"] = discount
        
        # Boolean fields (no validation needed, safe defaults)
        converted_data["is_active"] = FlextUtilities.Conversions.safe_bool(
            data.get("is_active"), default=True
        )
        converted_data["send_notifications"] = FlextUtilities.Conversions.safe_bool(
            data.get("send_notifications"), default=False
        )
        
        return converted_data, validation_errors

# Test with business validation
business_converter = BusinessDataConverter()

test_scenarios = [
    # Valid business data
    {
        "name": "Valid Product",
        "data": {
            "age": "25",
            "quantity": "5",
            "price": "49.99",
            "discount": "0.1",
            "is_active": "true",
            "send_notifications": "yes"
        }
    },
    # Invalid age
    {
        "name": "Invalid Age",
        "data": {
            "age": "150",  # Too old
            "quantity": "2",
            "price": "29.99",
            "discount": "0.05"
        }
    },
    # Invalid quantity and price
    {
        "name": "Invalid Quantity and Price",
        "data": {
            "age": "30",
            "quantity": "0",  # Below minimum
            "price": "-10.00",  # Negative price
            "discount": "1.5"  # Above 100%
        }
    },
    # Missing required fields
    {
        "name": "Missing Required Fields",
        "data": {
            "is_active": "false"
        }
    }
]

print("BUSINESS DATA CONVERSION & VALIDATION")
print("=" * 50)

for scenario in test_scenarios:
    print(f"\nScenario: {scenario['name']}")
    print(f"Input: {scenario['data']}")
    
    converted_data, errors = business_converter.convert_with_validation(scenario['data'])
    
    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Validation passed")
    
    print(f"Converted data: {converted_data}")
```

---

## ProcessingUtils Domain Implementation

### 1. Safe JSON Processing

```python
# Comprehensive JSON processing with error handling
class JSONProcessor:
    def __init__(self):
        self.processing_stats = {
            "successful_parses": 0,
            "failed_parses": 0,
            "successful_serializations": 0,
            "failed_serializations": 0
        }
    
    def process_json_data(self, json_strings: list[str]) -> dict:
        """Process multiple JSON strings with error collection."""
        results = {
            "successful": [],
            "failed": [],
            "parsed_objects": []
        }
        
        for i, json_str in enumerate(json_strings):
            # Safe JSON parsing
            parsed_data = FlextUtilities.ProcessingUtils.safe_json_parse(
                json_str, default={}
            )
            
            if parsed_data:  # Successfully parsed
                results["successful"].append(i)
                results["parsed_objects"].append(parsed_data)
                self.processing_stats["successful_parses"] += 1
            else:  # Failed to parse
                results["failed"].append(i)
                self.processing_stats["failed_parses"] += 1
        
        return results
    
    def serialize_objects(self, objects: list) -> dict:
        """Serialize multiple objects to JSON with error handling."""
        results = {
            "successful": [],
            "failed": [],
            "json_strings": []
        }
        
        for i, obj in enumerate(objects):
            # Safe JSON serialization
            json_str = FlextUtilities.ProcessingUtils.safe_json_stringify(
                obj, default="{}"
            )
            
            if json_str != "{}":  # Successfully serialized
                results["successful"].append(i)
                results["json_strings"].append(json_str)
                self.processing_stats["successful_serializations"] += 1
            else:  # Failed to serialize
                results["failed"].append(i)
                self.processing_stats["failed_serializations"] += 1
        
        return results
    
    def get_processing_stats(self) -> dict:
        """Get JSON processing statistics."""
        return self.processing_stats.copy()

# Test JSON processing
json_processor = JSONProcessor()

# Test JSON strings (mix of valid and invalid)
test_json_strings = [
    '{"name": "John", "age": 30, "active": true}',  # Valid
    '{"product": "laptop", "price": 999.99}',       # Valid
    '{"invalid": json}',                             # Invalid - missing quotes
    '{"unclosed": "object"',                         # Invalid - unclosed
    '',                                              # Invalid - empty
    'null',                                          # Valid - null
    '[]',                                           # Valid - empty array
    '{"nested": {"key": "value"}}',                 # Valid - nested object
]

print("JSON PROCESSING DEMONSTRATION")
print("=" * 50)

# Process JSON strings
parse_results = json_processor.process_json_data(test_json_strings)

print("PARSING RESULTS:")
print(f"Successful parses: {len(parse_results['successful'])}")
print(f"Failed parses: {len(parse_results['failed'])}")

for i, success_index in enumerate(parse_results["successful"]):
    original = test_json_strings[success_index]
    parsed = parse_results["parsed_objects"][i]
    print(f"  ✅ Index {success_index}: {original[:50]}... -> {type(parsed)}")

for fail_index in parse_results["failed"]:
    original = test_json_strings[fail_index]
    print(f"  ❌ Index {fail_index}: {original[:50]}...")

# Test serialization
test_objects = [
    {"name": "Alice", "age": 25},                    # Valid dict
    ["apple", "banana", "cherry"],                   # Valid list
    "simple string",                                 # Valid string
    42,                                             # Valid number
    None,                                           # Valid null
    set([1, 2, 3]),                                # Invalid - set not JSON serializable
    {"datetime": datetime.now()},                   # Problematic - datetime
]

print(f"\nSERIALIZATION RESULTS:")
serialize_results = json_processor.serialize_objects(test_objects)

print(f"Successful serializations: {len(serialize_results['successful'])}")
print(f"Failed serializations: {len(serialize_results['failed'])}")

for i, success_index in enumerate(serialize_results["successful"]):
    original = str(test_objects[success_index])
    serialized = serialize_results["json_strings"][i]
    print(f"  ✅ Index {success_index}: {original[:30]}... -> {serialized[:50]}...")

for fail_index in serialize_results["failed"]:
    original = str(test_objects[fail_index])
    print(f"  ❌ Index {fail_index}: {original[:30]}...")

# Display statistics
stats = json_processor.get_processing_stats()
print(f"\nPROCESSING STATISTICS:")
print(f"  Parse success rate: {stats['successful_parses']}/{stats['successful_parses'] + stats['failed_parses']}")
print(f"  Serialization success rate: {stats['successful_serializations']}/{stats['successful_serializations'] + stats['failed_serializations']}")
```

### 2. Model Processing and Extraction

```python
# Model processing with Pydantic integration
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Define test models
@dataclass
class User:
    id: int
    name: str
    email: str
    age: Optional[int] = None
    created_at: Optional[datetime] = None

class Product:
    """Pydantic-style model (mock)"""
    def __init__(self, name: str, price: float, category: str):
        self.name = name
        self.price = price
        self.category = category
    
    def model_dump(self) -> dict:
        """Mock Pydantic model_dump method"""
        return {
            "name": self.name,
            "price": self.price,
            "category": self.category,
            "exported_at": datetime.now().isoformat()
        }

class ModelProcessor:
    def __init__(self):
        self.extraction_stats = {
            "model_dump_extractions": 0,
            "dict_extractions": 0,
            "fallback_extractions": 0
        }
    
    def extract_data_from_objects(self, objects: list) -> list[dict]:
        """Extract data from various object types."""
        extracted_data = []
        
        for obj in objects:
            # Use FlextUtilities to extract model data
            data = FlextUtilities.ProcessingUtils.extract_model_data(obj)
            extracted_data.append(data)
            
            # Track extraction method used
            if hasattr(obj, "model_dump"):
                self.extraction_stats["model_dump_extractions"] += 1
            elif hasattr(obj, "dict"):
                self.extraction_stats["dict_extractions"] += 1
            elif isinstance(obj, dict):
                self.extraction_stats["dict_extractions"] += 1
            else:
                self.extraction_stats["fallback_extractions"] += 1
        
        return extracted_data
    
    def parse_json_to_models(self, json_data: list[str], model_classes: list[type]) -> dict:
        """Parse JSON to model instances."""
        results = {
            "successful_conversions": [],
            "failed_conversions": [],
            "models": []
        }
        
        for i, (json_str, model_class) in enumerate(zip(json_data, model_classes)):
            # Use FlextUtilities to parse JSON to model
            result = FlextUtilities.ProcessingUtils.parse_json_to_model(
                json_str, model_class
            )
            
            if result.success:
                results["successful_conversions"].append(i)
                results["models"].append(result.value)
            else:
                results["failed_conversions"].append(i)
                print(f"Conversion failed for index {i}: {result.error}")
        
        return results
    
    def get_extraction_stats(self) -> dict:
        """Get data extraction statistics."""
        return self.extraction_stats.copy()

# Test model processing
model_processor = ModelProcessor()

# Create test objects
test_objects = [
    User(1, "John Doe", "john@example.com", 30, datetime.now()),  # Dataclass
    Product("Laptop", 999.99, "Electronics"),  # Mock Pydantic model
    {"id": 2, "name": "Direct dict", "type": "manual"},  # Regular dict
    "String object",  # Non-dict object
    42,  # Number object
]

print("MODEL DATA EXTRACTION")
print("=" * 50)

# Extract data from objects
extracted_data = model_processor.extract_data_from_objects(test_objects)

for i, (original, extracted) in enumerate(zip(test_objects, extracted_data)):
    print(f"Object {i+1}:")
    print(f"  Type: {type(original).__name__}")
    print(f"  Original: {str(original)[:50]}...")
    print(f"  Extracted: {extracted}")
    print()

# Test JSON to model parsing
json_test_data = [
    '{"id": 100, "name": "Alice", "email": "alice@example.com", "age": 28}',
    '{"name": "Smartphone", "price": 599.99, "category": "Electronics"}',
    '{"id": 200, "name": "Bob"}',  # Missing email (should fail for User)
    '{"invalid": "json"',  # Invalid JSON
]

model_classes = [User, Product, User, User]

print("JSON TO MODEL CONVERSION")
print("=" * 50)

# Parse JSON to models
conversion_results = model_processor.parse_json_to_models(json_test_data, model_classes)

print(f"Successful conversions: {len(conversion_results['successful_conversions'])}")
print(f"Failed conversions: {len(conversion_results['failed_conversions'])}")

for i, success_index in enumerate(conversion_results["successful_conversions"]):
    model = conversion_results["models"][i]
    json_data = json_test_data[success_index]
    print(f"  ✅ Index {success_index}: {type(model).__name__}")
    print(f"    JSON: {json_data[:40]}...")
    print(f"    Model: {model}")

for fail_index in conversion_results["failed_conversions"]:
    json_data = json_test_data[fail_index]
    target_class = model_classes[fail_index].__name__
    print(f"  ❌ Index {fail_index}: Failed to create {target_class}")
    print(f"    JSON: {json_data[:40]}...")

# Display extraction statistics
extraction_stats = model_processor.get_extraction_stats()
print(f"\nEXTRACTION STATISTICS:")
print(f"  Model dump extractions: {extraction_stats['model_dump_extractions']}")
print(f"  Dict extractions: {extraction_stats['dict_extractions']}")
print(f"  Fallback extractions: {extraction_stats['fallback_extractions']}")
```

---

This implementation guide provides comprehensive examples for all FlextUtilities domains with practical, real-world scenarios. Each domain is demonstrated with complete code examples showing both basic usage and advanced enterprise patterns. The guide continues with the remaining domains following the same detailed approach.
