# FlextTypeAdapters Implementation Guide

**Complete step-by-step guide for implementing type adaptation, validation, and serialization using FlextTypeAdapters across all architectural layers.**

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Foundation Layer Implementation](#foundation-layer-implementation)
3. [Domain Layer Implementation](#domain-layer-implementation)
4. [Application Layer Implementation](#application-layer-implementation)
5. [Infrastructure Layer Implementation](#infrastructure-layer-implementation)
6. [Utilities Layer Implementation](#utilities-layer-implementation)
7. [Configuration Management](#configuration-management)
8. [Performance Optimization](#performance-optimization)
9. [Error Handling Patterns](#error-handling-patterns)
10. [Testing Strategies](#testing-strategies)

---

## Quick Start Guide

### Installation and Basic Setup

```python
# Import FlextTypeAdapters
from flext_core import FlextTypeAdapters
from flext_core.result import FlextResult
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

# Basic type adapter creation
string_adapter = FlextTypeAdapters.Foundation.create_string_adapter()
int_adapter = FlextTypeAdapters.Foundation.create_integer_adapter()
float_adapter = FlextTypeAdapters.Foundation.create_float_adapter()

# Simple validation example
validation_result = FlextTypeAdapters.Foundation.validate_with_adapter(
    string_adapter, "example_value"
)

if validation_result.success:
    validated_value = validation_result.value
    print(f"✅ Validation successful: {validated_value}")
else:
    print(f"❌ Validation failed: {validation_result.error}")
```

### 30-Second Implementation

```python
# Define your data model
@dataclass
class User:
    id: str
    name: str
    email: str
    age: int
    active: bool = True

# Create adapter
user_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(User)

# Validate data
user_data = {
    "id": "user_123",
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
}

result = FlextTypeAdapters.Foundation.validate_with_adapter(user_adapter, user_data)

if result.success:
    user = result.value  # Type: User
    print(f"Created user: {user.name}")
else:
    print(f"Validation failed: {result.error}")
```

---

## Foundation Layer Implementation

### 1. Primitive Type Adapters

The Foundation layer provides adapters for all primitive types with comprehensive error handling:

```python
# String Adapter with Validation
string_adapter = FlextTypeAdapters.Foundation.create_string_adapter()

# Validate string values
test_values = ["valid_string", "", None, 123]

for value in test_values:
    result = FlextTypeAdapters.Foundation.validate_with_adapter(
        string_adapter, value
    )

    if result.success:
        print(f"✅ '{value}' -> '{result.value}' (type: {type(result.value)})")
    else:
        print(f"❌ '{value}' failed: {result.error} (code: {result.error_code})")

# Integer Adapter with Range Validation
int_adapter = FlextTypeAdapters.Foundation.create_integer_adapter()

# Test various integer inputs
integer_tests = [42, "123", "not_a_number", 3.14, None]

for value in integer_tests:
    result = FlextTypeAdapters.Foundation.validate_with_adapter(
        int_adapter, value
    )

    if result.success:
        print(f"✅ {value} -> {result.value} (converted to int)")
    else:
        print(f"❌ {value} failed: {result.error}")
```

### 2. Custom Type Adapters

Create adapters for your domain-specific types:

```python
# Define custom types
@dataclass
class Address:
    street: str
    city: str
    postal_code: str
    country: str = "US"

@dataclass
class Person:
    first_name: str
    last_name: str
    address: Address
    birth_date: datetime
    phone: Optional[str] = None

# Create adapters for nested types
address_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Address)
person_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Person)

# Test nested type validation
person_data = {
    "first_name": "Jane",
    "last_name": "Smith",
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "postal_code": "12345"
    },
    "birth_date": "1990-01-15T00:00:00Z",
    "phone": "+1-555-123-4567"
}

result = FlextTypeAdapters.Foundation.validate_with_adapter(person_adapter, person_data)

if result.success:
    person = result.value
    print(f"✅ Created person: {person.first_name} {person.last_name}")
    print(f"   Address: {person.address.street}, {person.address.city}")
    print(f"   Phone: {person.phone}")
else:
    print(f"❌ Person validation failed: {result.error}")
```

### 3. Generic Type Support

FlextTypeAdapters supports generic types for collections and optional values:

```python
from typing import List, Dict, Optional, Union

# List of strings adapter
list_str_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(List[str])

# Dictionary adapter
dict_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Dict[str, int])

# Optional type adapter
optional_str_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Optional[str])

# Union type adapter
union_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Union[str, int])

# Test collection validation
test_cases = [
    (list_str_adapter, ["apple", "banana", "cherry"]),
    (list_str_adapter, ["valid", 123, "mixed"]),  # Should fail
    (dict_adapter, {"count": 10, "total": 42}),
    (dict_adapter, {"invalid": "string_value"}),  # Should fail
    (optional_str_adapter, "some_string"),
    (optional_str_adapter, None),
    (union_adapter, "string_value"),
    (union_adapter, 42),
    (union_adapter, [1, 2, 3])  # Should fail
]

for adapter, test_data in test_cases:
    result = FlextTypeAdapters.Foundation.validate_with_adapter(adapter, test_data)

    if result.success:
        print(f"✅ {test_data} validated successfully")
    else:
        print(f"❌ {test_data} failed: {result.error}")
```

---

## Domain Layer Implementation

### 1. Business Entity Validation

The Domain layer implements business-specific validation rules:

```python
# Entity ID Validation with Business Rules
def validate_user_id(user_id: str) -> FlextResult[str]:
    """Validate user ID according to business rules."""
    result = FlextTypeAdapters.Domain.validate_entity_id(user_id)

    if result.success:
        # Additional business validation
        validated_id = result.value

        # Check ID format: must start with 'user_' and have numeric suffix
        if not validated_id.startswith('user_'):
            return FlextResult.failure(
                error="User ID must start with 'user_' prefix",
                error_code="INVALID_USER_ID_FORMAT"
            )

        # Extract numeric part
        try:
            numeric_part = validated_id.split('_', 1)[1]
            int(numeric_part)  # Validate numeric suffix
        except (IndexError, ValueError):
            return FlextResult.failure(
                error="User ID must have numeric suffix after 'user_' prefix",
                error_code="INVALID_USER_ID_NUMERIC_SUFFIX"
            )

        return FlextResult.success(validated_id)

    return result

# Test entity ID validation
test_user_ids = [
    "user_12345",      # Valid
    "user_abc",        # Invalid - non-numeric suffix
    "customer_123",    # Invalid - wrong prefix
    "user_",          # Invalid - missing suffix
    "",               # Invalid - empty
    None              # Invalid - null
]

for user_id in test_user_ids:
    result = validate_user_id(user_id) if user_id is not None else \
             FlextResult.failure("Null user ID", "NULL_USER_ID")

    if result.success:
        print(f"✅ Valid user ID: {result.value}")
    else:
        print(f"❌ Invalid user ID '{user_id}': {result.error}")
```

### 2. Percentage and Range Validation

```python
# Business Percentage Rules
def validate_completion_percentage(percentage: float) -> FlextResult[float]:
    """Validate completion percentage with business rules."""

    # First, validate as general percentage
    result = FlextTypeAdapters.Domain.validate_percentage(percentage)

    if result.success:
        validated_percentage = result.value

        # Business rule: Completion can only be 0, 25, 50, 75, or 100
        allowed_values = [0.0, 25.0, 50.0, 75.0, 100.0]

        if validated_percentage not in allowed_values:
            return FlextResult.failure(
                error=f"Completion percentage must be one of {allowed_values}",
                error_code="INVALID_COMPLETION_PERCENTAGE"
            )

        return FlextResult.success(validated_percentage)

    return result

# Test percentage validation
test_percentages = [0, 25, 50, 75, 100, 30, 85.5, -10, 150]

for percentage in test_percentages:
    result = validate_completion_percentage(float(percentage))

    if result.success:
        print(f"✅ Valid completion: {result.value}%")
    else:
        print(f"❌ Invalid completion {percentage}%: {result.error}")
```

### 3. Network and Infrastructure Validation

```python
# Host and Port Validation for Database Connections
@dataclass
class DatabaseConnection:
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_enabled: bool = True

def validate_database_connection(config: dict) -> FlextResult[DatabaseConnection]:
    """Validate database connection configuration with business rules."""

    # First validate host/port combination
    host = config.get('host')
    port = config.get('port')

    if not host or not port:
        return FlextResult.failure(
            "Host and port are required",
            "MISSING_CONNECTION_INFO"
        )

    # Validate host/port combination
    host_port_result = FlextTypeAdapters.Domain.validate_host_port(host, port)

    if not host_port_result.success:
        return host_port_result

    # Create adapter for full configuration
    db_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(DatabaseConnection)

    # Validate complete configuration
    config_result = FlextTypeAdapters.Foundation.validate_with_adapter(
        db_adapter, config
    )

    if config_result.success:
        db_config = config_result.value

        # Business rule: Production requires SSL
        if not db_config.ssl_enabled and config.get('environment') == 'production':
            return FlextResult.failure(
                "SSL is required for production database connections",
                "SSL_REQUIRED_PRODUCTION"
            )

        return FlextResult.success(db_config)

    return config_result

# Test database connection validation
test_connections = [
    {
        "host": "localhost",
        "port": 5432,
        "database": "myapp",
        "username": "user",
        "password": "secret123",
        "ssl_enabled": True
    },
    {
        "host": "prod-db.company.com",
        "port": 5432,
        "database": "production",
        "username": "prod_user",
        "password": "prod_secret",
        "ssl_enabled": False,
        "environment": "production"  # Should fail - SSL required
    },
    {
        "host": "invalid-host",
        "port": 99999,  # Invalid port
        "database": "test",
        "username": "user",
        "password": "pass"
    }
]

for i, connection_config in enumerate(test_connections):
    result = validate_database_connection(connection_config)

    if result.success:
        db_config = result.value
        print(f"✅ Connection {i+1}: {db_config.host}:{db_config.port}")
        print(f"   SSL: {db_config.ssl_enabled}, DB: {db_config.database}")
    else:
        print(f"❌ Connection {i+1} failed: {result.error}")
```

---

## Application Layer Implementation

### 1. JSON Serialization and Deserialization

The Application layer provides enterprise-grade serialization capabilities:

```python
from datetime import datetime, timezone
from decimal import Decimal

# Define complex business model
@dataclass
class Order:
    id: str
    customer_id: str
    items: List[Dict[str, any]]
    total_amount: Decimal
    order_date: datetime
    status: str = "pending"
    metadata: Optional[Dict[str, str]] = None

# Create adapter
order_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Order)

# Sample order data
sample_order = Order(
    id="order_12345",
    customer_id="customer_67890",
    items=[
        {"product_id": "prod_1", "quantity": 2, "price": 29.99},
        {"product_id": "prod_2", "quantity": 1, "price": 15.50}
    ],
    total_amount=Decimal("75.48"),
    order_date=datetime.now(timezone.utc),
    status="confirmed",
    metadata={"source": "web", "campaign": "summer_sale"}
)

# JSON Serialization with Error Handling
json_result = FlextTypeAdapters.Application.serialize_to_json(
    order_adapter, sample_order
)

if json_result.success:
    json_string = json_result.value
    print("✅ JSON Serialization successful:")
    print(json_string[:200] + "..." if len(json_string) > 200 else json_string)

    # JSON Deserialization
    deserialization_result = FlextTypeAdapters.Application.deserialize_from_json(
        json_string, order_adapter
    )

    if deserialization_result.success:
        reconstructed_order = deserialization_result.value
        print(f"✅ Deserialization successful: {reconstructed_order.id}")
        print(f"   Amount: ${reconstructed_order.total_amount}")
        print(f"   Items: {len(reconstructed_order.items)}")
    else:
        print(f"❌ Deserialization failed: {deserialization_result.error}")

else:
    print(f"❌ Serialization failed: {json_result.error}")
```

### 2. Dictionary Conversion

```python
# Dictionary Serialization and Deserialization
dict_result = FlextTypeAdapters.Application.serialize_to_dict(
    order_adapter, sample_order
)

if dict_result.success:
    order_dict = dict_result.value
    print("✅ Dictionary serialization successful:")
    print(f"   Order ID: {order_dict['id']}")
    print(f"   Customer: {order_dict['customer_id']}")
    print(f"   Status: {order_dict['status']}")

    # Dictionary Deserialization
    dict_deserialization_result = FlextTypeAdapters.Application.deserialize_from_dict(
        order_dict, order_adapter
    )

    if dict_deserialization_result.success:
        reconstructed_order = dict_deserialization_result.value
        print(f"✅ Dict deserialization successful: {reconstructed_order.id}")
    else:
        print(f"❌ Dict deserialization failed: {dict_deserialization_result.error}")
else:
    print(f"❌ Dict serialization failed: {dict_result.error}")
```

### 3. Schema Generation for API Documentation

```python
# Generate JSON Schema for OpenAPI Documentation
schema_result = FlextTypeAdapters.Application.generate_schema(order_adapter)

if schema_result.success:
    order_schema = schema_result.value
    print("✅ Schema generation successful:")
    print(f"   Type: {order_schema.get('type')}")
    print(f"   Properties: {list(order_schema.get('properties', {}).keys())}")
    print(f"   Required: {order_schema.get('required', [])}")

    # Pretty print schema structure
    import json
    schema_json = json.dumps(order_schema, indent=2)
    print("\n📋 Complete Schema:")
    print(schema_json[:500] + "..." if len(schema_json) > 500 else schema_json)
else:
    print(f"❌ Schema generation failed: {schema_result.error}")

# Batch Schema Generation for Multiple Models
@dataclass
class Customer:
    id: str
    name: str
    email: str
    address: Address

@dataclass
class Product:
    id: str
    name: str
    price: Decimal
    category: str
    in_stock: bool = True

# Create adapters
customer_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Customer)
product_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(Product)

# Generate multiple schemas at once
adapters = {
    "Customer": customer_adapter,
    "Product": product_adapter,
    "Order": order_adapter,
    "Address": address_adapter
}

schemas_result = FlextTypeAdapters.Application.generate_multiple_schemas(adapters)

if schemas_result.success:
    all_schemas = schemas_result.value
    print(f"✅ Generated {len(all_schemas)} schemas:")

    for model_name, schema in all_schemas.items():
        properties_count = len(schema.get('properties', {}))
        print(f"   📄 {model_name}: {properties_count} properties")

    # Create OpenAPI specification
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "FLEXT Business API",
            "version": "1.0.0",
            "description": "Auto-generated API documentation using FlextTypeAdapters"
        },
        "components": {
            "schemas": all_schemas
        }
    }

    print("\n🚀 OpenAPI specification ready for use!")

else:
    print(f"❌ Multiple schema generation failed: {schemas_result.error}")
```

### 4. Batch Processing

```python
# Batch Serialization for High-Volume Operations
orders_data = []
for i in range(5):
    order = Order(
        id=f"order_{i+1:03d}",
        customer_id=f"customer_{i+1:03d}",
        items=[{"product_id": f"prod_{i+1}", "quantity": 1, "price": 19.99}],
        total_amount=Decimal("19.99"),
        order_date=datetime.now(timezone.utc),
        status="pending"
    )
    orders_data.append(order)

# Batch JSON serialization
batch_json_result = FlextTypeAdapters.Application.batch_serialize_to_json(
    order_adapter, orders_data
)

if batch_json_result.success:
    json_strings = batch_json_result.value
    print(f"✅ Batch JSON serialization successful: {len(json_strings)} orders")

    # Show first order as example
    print(f"   Sample: {json_strings[0][:100]}...")

    # Batch JSON deserialization
    batch_deserialization_result = FlextTypeAdapters.Application.batch_deserialize_from_json(
        json_strings, order_adapter
    )

    if batch_deserialization_result.success:
        reconstructed_orders = batch_deserialization_result.value
        print(f"✅ Batch deserialization successful: {len(reconstructed_orders)} orders")

        for order in reconstructed_orders[:3]:  # Show first 3
            print(f"   📦 {order.id}: ${order.total_amount}")

    else:
        print(f"❌ Batch deserialization failed: {batch_deserialization_result.error}")

else:
    print(f"❌ Batch serialization failed: {batch_json_result.error}")
```

---

## Infrastructure Layer Implementation

### 1. Custom Adapter Protocols

The Infrastructure layer provides protocol-based interfaces for custom adapters:

```python
# Define custom validation protocol
from typing import Protocol, object

class CustomStringValidator(Protocol):
    """Protocol for custom string validation logic."""

    def validate(self, value: str) -> bool:
        """Validate string according to custom rules."""
        ...

    def get_error_message(self, value: str) -> str:
        """Get descriptive error message for invalid values."""
        ...

# Implement custom validator
class AlphanumericValidator:
    """Validates that strings contain only alphanumeric characters."""

    def validate(self, value: str) -> bool:
        return value.isalnum() and len(value) >= 3

    def get_error_message(self, value: str) -> str:
        if not value.isalnum():
            return f"Value '{value}' must contain only alphanumeric characters"
        if len(value) < 3:
            return f"Value '{value}' must be at least 3 characters long"
        return f"Value '{value}' is invalid"

# Register custom adapter
alphanumeric_validator = AlphanumericValidator()

# Create adapter protocol
adapter_protocol = FlextTypeAdapters.Infrastructure.create_adapter_protocol(
    alphanumeric_validator
)

# Test custom validation
test_values = ["abc123", "test", "12", "hello!", "", "ValidString123"]

for value in test_values:
    is_valid = alphanumeric_validator.validate(value)

    if is_valid:
        print(f"✅ '{value}' passed custom validation")
    else:
        error_msg = alphanumeric_validator.get_error_message(value)
        print(f"❌ '{value}' failed: {error_msg}")
```

### 2. Adapter Registry Management

```python
# Initialize adapter registry
registry = FlextTypeAdapters.Infrastructure.AdapterRegistry()

# Register multiple custom adapters
custom_adapters = {
    "alphanumeric": AlphanumericValidator(),
    "email_strict": EmailStrictValidator(),  # Hypothetical strict email validator
    "password_strong": StrongPasswordValidator(),  # Hypothetical password validator
}

for name, adapter in custom_adapters.items():
    registry_result = FlextTypeAdapters.Infrastructure.register_adapter(name, adapter)

    if registry_result.success:
        print(f"✅ Registered adapter: {name}")
    else:
        print(f"❌ Failed to register {name}: {registry_result.error}")

# List all registered adapters
adapters_result = FlextTypeAdapters.Infrastructure.list_registered_adapters()

if adapters_result.success:
    adapter_names = adapters_result.value
    print(f"\n📋 Registered adapters ({len(adapter_names)}):")
    for name in adapter_names:
        print(f"   • {name}")

# Retrieve and use registered adapter
adapter_result = FlextTypeAdapters.Infrastructure.get_adapter("alphanumeric")

if adapter_result.success:
    retrieved_adapter = adapter_result.value

    # Use retrieved adapter for validation
    test_value = "TestValue123"
    validation_result = retrieved_adapter.validate(test_value)

    print(f"\n🧪 Testing retrieved adapter with '{test_value}': {validation_result}")
else:
    print(f"❌ Failed to retrieve adapter: {adapter_result.error}")
```

### 3. Dependency Injection Integration

```python
from flext_core.protocols import FlextProtocols
from typing import inject

# Define validation service protocol
class ValidationService(FlextProtocols.Base):
    """Protocol for validation services."""

    def validate_user_input(self, data: dict) -> FlextResult[dict]:
        """Validate user input data."""
        ...

    def validate_business_rules(self, entity: any) -> FlextResult[bool]:
        """Validate business rules for entity."""
        ...

# Implement validation service using FlextTypeAdapters
class UserValidationService:
    """User validation service implementation."""

    def __init__(self):
        self.user_adapter = FlextTypeAdapters.Foundation.create_basic_adapter(User)
        self.registry = FlextTypeAdapters.Infrastructure.AdapterRegistry()

    def validate_user_input(self, data: dict) -> FlextResult[dict]:
        """Validate user input with custom and built-in rules."""

        # First, validate basic structure
        basic_result = FlextTypeAdapters.Foundation.validate_with_adapter(
            self.user_adapter, data
        )

        if not basic_result.success:
            return basic_result

        # Then apply custom business rules
        user = basic_result.value

        # Validate username with custom alphanumeric validator
        username_adapter = self.registry.get_adapter("alphanumeric")
        if username_adapter.success:
            username_valid = username_adapter.value.validate(user.name)
            if not username_valid:
                return FlextResult.failure(
                    username_adapter.value.get_error_message(user.name),
                    "INVALID_USERNAME"
                )

        return FlextResult.success(data)

    def validate_business_rules(self, entity: User) -> FlextResult[bool]:
        """Validate business-specific rules for user entity."""

        # Business rule: User must have valid email domain
        allowed_domains = ["company.com", "example.com", "gmail.com"]
        email_domain = entity.email.split("@")[-1] if "@" in entity.email else ""

        if email_domain not in allowed_domains:
            return FlextResult.failure(
                f"Email domain '{email_domain}' is not allowed. Use: {allowed_domains}",
                "INVALID_EMAIL_DOMAIN"
            )

        # Business rule: Age must be reasonable for business context
        if entity.age < 18 or entity.age > 100:
            return FlextResult.failure(
                f"Age {entity.age} is outside allowed range (18-100)",
                "INVALID_AGE_RANGE"
            )

        return FlextResult.success(True)

# Register service with DI container (conceptual)
# container.register(ValidationService, UserValidationService)

# Usage example
validation_service = UserValidationService()

test_user_data = {
    "id": "user_123",
    "name": "JohnDoe",
    "email": "john@company.com",
    "age": 30,
    "active": True
}

# Validate input
input_result = validation_service.validate_user_input(test_user_data)

if input_result.success:
    print("✅ User input validation passed")

    # Create user object for business rule validation
    user_result = FlextTypeAdapters.Foundation.validate_with_adapter(
        validation_service.user_adapter, test_user_data
    )

    if user_result.success:
        user = user_result.value

        # Validate business rules
        business_result = validation_service.validate_business_rules(user)

        if business_result.success:
            print("✅ Business rules validation passed")
            print(f"   User {user.name} is ready for processing")
        else:
            print(f"❌ Business rules failed: {business_result.error}")
    else:
        print(f"❌ User creation failed: {user_result.error}")
else:
    print(f"❌ Input validation failed: {input_result.error}")
```

---

## Configuration Management

### 1. System Configuration

```python
# Configure FlextTypeAdapters system
config = {
    "environment": "production",
    "validation_level": "strict",
    "performance_optimization": True,
    "enable_adapter_caching": True,
    "batch_size_limit": 10000,
    "error_detail_level": "full",
    "json_serialization_mode": "fast",
    "schema_generation_format": "openapi_3.0"
}

# Apply configuration
config_result = FlextTypeAdapters.configure_type_adapters_system(config)

if config_result.success:
    applied_config = config_result.value
    print("✅ FlextTypeAdapters configured successfully")
    print(f"   Environment: {applied_config['environment']}")
    print(f"   Validation Level: {applied_config['validation_level']}")
    print(f"   Caching: {applied_config['enable_adapter_caching']}")
else:
    print(f"❌ Configuration failed: {config_result.error}")

# Get current configuration
current_config_result = FlextTypeAdapters.get_type_adapters_system_config()

if current_config_result.success:
    current_config = current_config_result.value
    print(f"\n📋 Current configuration:")
    for key, value in current_config.items():
        print(f"   {key}: {value}")
```

### 2. Environment-Specific Configuration

```python
# Create environment-specific configurations
environments = ["development", "testing", "staging", "production"]

for env in environments:
    env_config_result = FlextTypeAdapters.create_environment_specific_config(env)

    if env_config_result.success:
        env_config = env_config_result.value
        print(f"✅ {env.title()} configuration:")
        print(f"   Validation Level: {env_config['validation_level']}")
        print(f"   Error Details: {env_config['error_detail_level']}")
        print(f"   Performance Mode: {env_config['performance_optimization']}")
        print()
    else:
        print(f"❌ Failed to create {env} config: {env_config_result.error}")
```

### 3. Performance Optimization

```python
# Optimize for different performance levels
performance_levels = ["minimal", "balanced", "maximum"]

for level in performance_levels:
    perf_config_result = FlextTypeAdapters.optimize_performance(level)

    if perf_config_result.success:
        perf_config = perf_config_result.value
        print(f"✅ {level.title()} performance optimization:")
        print(f"   Batch Size: {perf_config['batch_size_limit']}")
        print(f"   Caching: {perf_config['enable_adapter_caching']}")
        print(f"   Validation Mode: {perf_config['validation_mode']}")
        print()
    else:
        print(f"❌ Failed to optimize for {level}: {perf_config_result.error}")
```

---

This implementation guide provides comprehensive coverage of all FlextTypeAdapters capabilities with practical examples. Each layer builds upon the previous ones, creating a complete type adaptation solution for enterprise applications.
