# FlextHandlers Implementation Guide

**Version**: 0.9.0  
**Target Audience**: FLEXT Developers, Enterprise Architects  
**Implementation Time**: 2-4 weeks per handler ecosystem  
**Complexity**: Intermediate to Advanced

## 📖 Overview

This guide provides comprehensive instructions for implementing enterprise-grade handler systems using `FlextHandlers`. The framework offers **7 architectural layers**, **8 integrated design patterns**, and complete **CQRS infrastructure** for sophisticated request processing, validation, authorization, and event sourcing in enterprise applications.

### Prerequisites

- Understanding of Design Patterns (Chain of Responsibility, Command, Observer, Registry, Factory, Decorator, Strategy, Template Method)
- Familiarity with CQRS (Command Query Responsibility Segregation) and Event Sourcing
- Knowledge of railway-oriented programming with FlextResult patterns
- Experience with enterprise security patterns (authentication, authorization, validation)

### Implementation Benefits

- 🏗️ **Complete Handler Infrastructure** through 7-layer architecture
- ⚡ **Enterprise Performance** with thread-safe operations and comprehensive metrics
- 🔗 **Pattern Integration Excellence** with 8 design patterns and FlextResult integration
- 🛡️ **Security-First Design** with built-in validation, authorization, and access control
- 🌐 **CQRS Excellence** with CommandBus, QueryBus, EventBus, and domain event sourcing

---

## 🚀 Quick Start

### Basic Handler Implementation

```python
from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Create enterprise handler with comprehensive configuration
handler = FlextHandlers.Implementation.BasicHandler("user_service")

# Configure for production environment
config_result = FlextHandlers.Implementation.BasicHandler.create_environment_handler_config("production")
if config_result.success:
    production_config = config_result.value
    configure_result = handler.configure(production_config)
    print(f"Handler configured for {production_config['environment']}")

# Process request with automatic metrics collection
request_data = {
    "action": "create_user",
    "user": {"name": "John Doe", "email": "john@company.com"},
    "metadata": {"source": "api", "version": "2.1.0"}
}

processing_result = handler.handle(request_data)
if processing_result.success:
    response = processing_result.value
    print(f"Request processed successfully: {response}")

    # Access comprehensive metrics
    metrics = handler.get_metrics()
    print(f"Success rate: {metrics['successful_requests'] / metrics['requests_processed'] * 100:.1f}%")
    print(f"Average response time: {metrics['average_processing_time']*1000:.1f}ms")
else:
    print(f"Processing failed: {processing_result.error}")
```

### Complete CQRS Implementation

```python
from dataclasses import dataclass
from typing import Dict, object
import datetime

# Define Commands and Queries
@dataclass
class CreateUserCommand:
    name: str
    email: str
    department: str

    def validate(self) -> FlextResult[None]:
        if not self.name or len(self.name) < 2:
            return FlextResult[None].fail("Name must be at least 2 characters")
        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)

@dataclass
class GetUserQuery:
    user_id: str
    include_details: bool = False

@dataclass
class UserCreatedEvent:
    user_id: str
    user_name: str
    created_at: datetime.datetime

# Command Handler
class UserCommandHandler:
    def __init__(self):
        self.users_db: Dict[str, dict] = {}
        self.user_counter = 1

    def handle_create_user(self, command: CreateUserCommand) -> FlextResult[str]:
        # Validate command
        validation = command.validate()
        if validation.is_failure:
            return FlextResult[str].fail(validation.error)

        # Business logic
        user_id = f"user_{self.user_counter}"
        self.user_counter += 1

        user_data = {
            "id": user_id,
            "name": command.name,
            "email": command.email,
            "department": command.department,
            "created_at": datetime.datetime.now()
        }

        self.users_db[user_id] = user_data
        return FlextResult[str].ok(user_id)

# Query Handler
class UserQueryHandler:
    def __init__(self, users_db: Dict[str, dict]):
        self.users_db = users_db

    def handle_get_user(self, query: GetUserQuery) -> FlextResult[dict]:
        if query.user_id not in self.users_db:
            return FlextResult[dict].fail(f"User {query.user_id} not found")

        user_data = dict(self.users_db[query.user_id])
        if not query.include_details:
            user_data.pop("email", None)  # Remove sensitive data

        return FlextResult[dict].ok(user_data)

# Event Handler
class UserEventHandler:
    def __init__(self):
        self.events_processed = 0

    def handle_user_created(self, event: UserCreatedEvent) -> FlextResult[None]:
        self.events_processed += 1
        print(f"🔔 Welcome notification sent to {event.user_name}")
        return FlextResult[None].ok(None)

# Setup Complete CQRS System
command_bus = FlextHandlers.CQRS.CommandBus()
query_bus = FlextHandlers.CQRS.QueryBus()
event_bus = FlextHandlers.CQRS.EventBus()

# Initialize handlers
command_handler = UserCommandHandler()
query_handler = UserQueryHandler(command_handler.users_db)
event_handler = UserEventHandler()

# Register handlers
command_bus.register(CreateUserCommand, command_handler.handle_create_user)
query_bus.register(GetUserQuery, query_handler.handle_get_user)
event_bus.subscribe("UserCreated", event_handler.handle_user_created)

# Execute CQRS workflow
create_command = CreateUserCommand(
    name="Alice Johnson",
    email="alice@company.com",
    department="Engineering"
)

# Send command
command_result = command_bus.send(create_command)
if command_result.success:
    user_id = command_result.value
    print(f"✅ User created: {user_id}")

    # Publish domain event
    user_event = UserCreatedEvent(
        user_id=user_id,
        user_name=create_command.name,
        created_at=datetime.datetime.now()
    )
    event_bus.publish(user_event)

    # Query the created user
    get_query = GetUserQuery(user_id=user_id, include_details=True)
    query_result = query_bus.execute(get_query)

    if query_result.success:
        user_data = query_result.value
        print(f"✅ User retrieved: {user_data['name']} in {user_data['department']}")
```

---

## 📚 Step-by-Step Implementation

### Step 1: Handler Foundation Setup

#### Understanding the 7-Layer Architecture

```python
from flext_core.handlers import FlextHandlers

# Layer 1: Constants - Configuration and state management
print("=== Constants Layer ===")
print(f"Default timeout: {FlextHandlers.Constants.Handler.DEFAULT_TIMEOUT}ms")
print(f"Max chain handlers: {FlextHandlers.Constants.Handler.MAX_CHAIN_HANDLERS}")
print(f"Handler states: {FlextHandlers.Constants.Handler.States.IDLE}, {FlextHandlers.Constants.Handler.States.PROCESSING}")

# Layer 2: Types - Type system integration
print("\n=== Types Layer ===")
handler_name: FlextHandlers.Types.HandlerTypes.Name = "my_handler"
handler_state: FlextHandlers.Types.HandlerTypes.State = FlextHandlers.Constants.Handler.States.IDLE
handler_metrics: FlextHandlers.Types.HandlerTypes.Metrics = {"requests": 0, "successes": 0}

# Layer 3: Protocols - Contract definitions
print("\n=== Protocols Layer ===")
# Protocols ensure type safety and contract compliance
# Used for: MetricsHandler, ChainableHandler, ValidatingHandler, AuthorizingHandler

# Layer 4: Implementation - Concrete handlers
print("\n=== Implementation Layer ===")
basic_handler = FlextHandlers.Implementation.BasicHandler("example_handler")

# Layer 5: CQRS - Command Query Responsibility Segregation
print("\n=== CQRS Layer ===")
command_bus = FlextHandlers.CQRS.CommandBus()
query_bus = FlextHandlers.CQRS.QueryBus()
event_bus = FlextHandlers.CQRS.EventBus()

# Layer 6: Patterns - Design pattern implementations
print("\n=== Patterns Layer ===")
handler_chain = FlextHandlers.Patterns.HandlerChain("processing_pipeline")
pipeline = FlextHandlers.Patterns.Pipeline("data_pipeline")
middleware = FlextHandlers.Patterns.Middleware("request_middleware")

# Layer 7: Management - Registry and lifecycle
print("\n=== Management Layer ===")
registry = FlextHandlers.Management.HandlerRegistry()
```

#### Creating Custom Handlers with Type Safety

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

# Define custom types for your domain
RequestType = TypeVar('RequestType')
ResponseType = TypeVar('ResponseType')

class OrderProcessingHandler(FlextHandlers.Implementation.AbstractHandler[dict, dict]):
    """Custom order processing handler with comprehensive business logic."""

    def __init__(self):
        self._handler_name = "order_processor"
        self._orders_db = {}
        self._order_counter = 1

    @property
    def handler_name(self) -> str:
        return self._handler_name

    def handle(self, request: dict) -> FlextResult[dict]:
        """Process order request with comprehensive validation and business logic."""

        # Input validation
        if not request.get("customer_id"):
            return FlextResult[dict].fail("Customer ID is required")

        if not request.get("items") or not isinstance(request["items"], list):
            return FlextResult[dict].fail("Order must contain items")

        # Business logic validation
        if len(request["items"]) > 10:
            return FlextResult[dict].fail("Maximum 10 items per order")

        total_amount = sum(item.get("price", 0) * item.get("quantity", 0) for item in request["items"])
        if total_amount <= 0:
            return FlextResult[dict].fail("Order total must be positive")

        # Create order
        order_id = f"order_{self._order_counter}"
        self._order_counter += 1

        order = {
            "id": order_id,
            "customer_id": request["customer_id"],
            "items": request["items"],
            "total_amount": total_amount,
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat(),
            "processing_notes": []
        }

        # Apply business rules
        if total_amount > 1000:
            order["status"] = "requires_approval"
            order["processing_notes"].append("High-value order requires manager approval")

        if len(request["items"]) >= 5:
            order["bulk_discount"] = 0.05  # 5% bulk discount
            order["total_amount"] = total_amount * 0.95
            order["processing_notes"].append("Bulk discount applied (5%)")

        # Store order
        self._orders_db[order_id] = order

        return FlextResult[dict].ok(order)

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can process the message type."""
        return message_type == dict

    def get_order_statistics(self) -> dict:
        """Get order processing statistics."""
        orders = list(self._orders_db.values())
        if not orders:
            return {"total_orders": 0, "total_revenue": 0}

        total_revenue = sum(order["total_amount"] for order in orders)
        avg_order_value = total_revenue / len(orders)

        return {
            "total_orders": len(orders),
            "total_revenue": total_revenue,
            "average_order_value": avg_order_value,
            "orders_requiring_approval": len([o for o in orders if o["status"] == "requires_approval"])
        }

# Usage with comprehensive testing
order_handler = OrderProcessingHandler()

# Test various order scenarios
test_orders = [
    # Valid standard order
    {
        "customer_id": "cust_001",
        "items": [
            {"name": "Laptop", "price": 800, "quantity": 1},
            {"name": "Mouse", "price": 25, "quantity": 2}
        ]
    },
    # High-value order requiring approval
    {
        "customer_id": "cust_002",
        "items": [
            {"name": "Server", "price": 1500, "quantity": 1}
        ]
    },
    # Bulk order with discount
    {
        "customer_id": "cust_003",
        "items": [
            {"name": "Cable", "price": 10, "quantity": 5},
            {"name": "Adapter", "price": 15, "quantity": 3},
            {"name": "Switch", "price": 50, "quantity": 2},
            {"name": "Router", "price": 100, "quantity": 1},
            {"name": "Modem", "price": 80, "quantity": 1}
        ]
    },
    # Invalid order - no items
    {
        "customer_id": "cust_004",
        "items": []
    }
]

print("=== Order Processing Handler Testing ===")
for i, test_order in enumerate(test_orders):
    print(f"\nOrder {i+1}: Customer {test_order.get('customer_id', 'unknown')}")
    result = order_handler.handle(test_order)

    if result.success:
        order = result.value
        print(f"✅ Order processed: {order['id']} (${order['total_amount']:.2f})")
        print(f"   Status: {order['status']}")
        if order.get("processing_notes"):
            for note in order["processing_notes"]:
                print(f"   📝 {note}")
    else:
        print(f"❌ Order failed: {result.error}")

# Display order statistics
stats = order_handler.get_order_statistics()
print(f"\n📊 Order Processing Statistics:")
print(f"   Total orders: {stats['total_orders']}")
print(f"   Total revenue: ${stats['total_revenue']:.2f}")
print(f"   Average order value: ${stats.get('average_order_value', 0):.2f}")
print(f"   Orders requiring approval: {stats['orders_requiring_approval']}")
```

### Step 2: Handler Chain Implementation

#### Building Enterprise Processing Pipelines

```python
class AuthenticationHandler:
    """Authentication handler for the processing chain."""

    def __init__(self, name: str = "authenticator"):
        self.handler_name = name
        self.auth_attempts = 0
        self.failed_auths = 0

    def handle(self, request: dict) -> FlextResult[dict]:
        """Authenticate user request."""
        self.auth_attempts += 1

        # Check for authentication token
        auth_token = request.get("auth_token")
        if not auth_token:
            self.failed_auths += 1
            return FlextResult[dict].fail("Authentication token required")

        # Mock token validation (would integrate with real auth system)
        valid_tokens = {
            "admin_token_123": {"user_id": "admin_001", "role": "admin", "permissions": ["read", "write", "delete"]},
            "manager_token_456": {"user_id": "mgr_002", "role": "manager", "permissions": ["read", "write"]},
            "user_token_789": {"user_id": "user_003", "role": "user", "permissions": ["read"]}
        }

        if auth_token not in valid_tokens:
            self.failed_auths += 1
            return FlextResult[dict].fail("Invalid authentication token")

        # Add user context to request
        user_context = valid_tokens[auth_token]
        enriched_request = dict(request)
        enriched_request["user_context"] = user_context
        enriched_request["authenticated"] = True

        print(f"🔑 User authenticated: {user_context['user_id']} ({user_context['role']})")
        return FlextResult[dict].ok(enriched_request)

    def can_handle(self, message_type: type) -> bool:
        return message_type == dict

class AuthorizationHandler:
    """Authorization handler checking permissions."""

    def __init__(self, name: str = "authorizer"):
        self.handler_name = name
        self.authz_checks = 0
        self.denied_requests = 0

    def handle(self, request: dict) -> FlextResult[dict]:
        """Check user authorization for requested operation."""
        self.authz_checks += 1

        # Get user context from authentication
        user_context = request.get("user_context")
        if not user_context:
            self.denied_requests += 1
            return FlextResult[dict].fail("No user context available - authentication required")

        # Get requested operation
        operation = request.get("operation", "read")
        required_permission = self._map_operation_to_permission(operation)

        # Check permissions
        user_permissions = user_context.get("permissions", [])
        if required_permission not in user_permissions:
            self.denied_requests += 1
            return FlextResult[dict].fail(
                f"User {user_context['user_id']} lacks permission '{required_permission}' for operation '{operation}'"
            )

        print(f"🛡️ Authorization granted for {operation} operation")
        return FlextResult[dict].ok(request)

    def _map_operation_to_permission(self, operation: str) -> str:
        """Map operations to required permissions."""
        operation_map = {
            "read": "read",
            "get": "read",
            "list": "read",
            "create": "write",
            "update": "write",
            "patch": "write",
            "delete": "delete",
            "remove": "delete"
        }
        return operation_map.get(operation.lower(), "read")

    def can_handle(self, message_type: type) -> bool:
        return message_type == dict

class ValidationHandler:
    """Input validation handler."""

    def __init__(self, name: str = "validator"):
        self.handler_name = name
        self.validations = 0
        self.validation_failures = 0

    def handle(self, request: dict) -> FlextResult[dict]:
        """Validate request input data."""
        self.validations += 1

        # Check required fields
        required_fields = ["operation", "endpoint"]
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            self.validation_failures += 1
            return FlextResult[dict].fail(f"Missing required fields: {missing_fields}")

        # Validate operation
        valid_operations = ["read", "create", "update", "delete", "list"]
        operation = request.get("operation")
        if operation not in valid_operations:
            self.validation_failures += 1
            return FlextResult[dict].fail(f"Invalid operation '{operation}'. Valid operations: {valid_operations}")

        # Validate endpoint format
        endpoint = request.get("endpoint")
        if not endpoint.startswith("/api/"):
            self.validation_failures += 1
            return FlextResult[dict].fail("Endpoint must start with '/api/'")

        # Data validation based on operation
        if operation in ["create", "update"]:
            data = request.get("data")
            if not data or not isinstance(data, dict):
                self.validation_failures += 1
                return FlextResult[dict].fail(f"Operation '{operation}' requires valid data object")

        print(f"✅ Input validation passed for {operation} operation")
        return FlextResult[dict].ok(request)

    def can_handle(self, message_type: type) -> bool:
        return message_type == dict

class BusinessLogicHandler:
    """Final business logic processing handler."""

    def __init__(self, name: str = "business_logic"):
        self.handler_name = name
        self.processed_requests = 0
        self.processing_errors = 0

    def handle(self, request: dict) -> FlextResult[dict]:
        """Execute core business logic."""
        self.processed_requests += 1

        try:
            operation = request["operation"]
            endpoint = request["endpoint"]
            user_context = request["user_context"]

            # Mock business logic based on operation
            if operation == "create":
                result = self._handle_create(request)
            elif operation == "update":
                result = self._handle_update(request)
            elif operation == "delete":
                result = self._handle_delete(request)
            elif operation in ["read", "get", "list"]:
                result = self._handle_read(request)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            # Add processing metadata
            result["processed_by"] = self.handler_name
            result["processed_at"] = datetime.datetime.now().isoformat()
            result["user_id"] = user_context["user_id"]

            print(f"🏭 Business logic processed: {operation} at {endpoint}")
            return FlextResult[dict].ok(result)

        except Exception as e:
            self.processing_errors += 1
            return FlextResult[dict].fail(f"Business logic error: {str(e)}")

    def _handle_create(self, request: dict) -> dict:
        """Handle create operations."""
        data = request["data"]
        return {
            "operation": "create",
            "resource_id": f"res_{hash(str(data)) % 10000}",
            "data": data,
            "status": "created"
        }

    def _handle_update(self, request: dict) -> dict:
        """Handle update operations."""
        data = request["data"]
        resource_id = request.get("resource_id", "unknown")
        return {
            "operation": "update",
            "resource_id": resource_id,
            "data": data,
            "status": "updated"
        }

    def _handle_delete(self, request: dict) -> dict:
        """Handle delete operations."""
        resource_id = request.get("resource_id", "unknown")
        return {
            "operation": "delete",
            "resource_id": resource_id,
            "status": "deleted"
        }

    def _handle_read(self, request: dict) -> dict:
        """Handle read operations."""
        endpoint = request["endpoint"]
        return {
            "operation": "read",
            "endpoint": endpoint,
            "data": {"message": f"Data from {endpoint}"},
            "status": "success"
        }

    def can_handle(self, message_type: type) -> bool:
        return message_type == dict

# Build Enterprise Processing Chain
print("=== Building Enterprise Processing Chain ===")
processing_chain = FlextHandlers.Patterns.HandlerChain("enterprise_api_pipeline")

# Create handler instances
auth_handler = AuthenticationHandler()
authz_handler = AuthorizationHandler()
validation_handler = ValidationHandler()
business_handler = BusinessLogicHandler()

# Build chain: Authentication → Authorization → Validation → Business Logic
chain_steps = [
    ("Authentication", auth_handler),
    ("Authorization", authz_handler),
    ("Validation", validation_handler),
    ("Business Logic", business_handler)
]

for step_name, handler in chain_steps:
    add_result = processing_chain.add_handler(handler)
    if add_result.success:
        print(f"✅ {step_name} handler added to chain")
    else:
        print(f"❌ Failed to add {step_name} handler: {add_result.error}")

# Test Enterprise Processing Chain
test_requests = [
    # Valid admin request
    {
        "auth_token": "admin_token_123",
        "operation": "delete",
        "endpoint": "/api/users/123",
        "resource_id": "user_123"
    },

    # Valid manager request
    {
        "auth_token": "manager_token_456",
        "operation": "create",
        "endpoint": "/api/products",
        "data": {"name": "New Product", "price": 99.99}
    },

    # Valid user request
    {
        "auth_token": "user_token_789",
        "operation": "read",
        "endpoint": "/api/profile"
    },

    # Unauthorized request - user trying to delete
    {
        "auth_token": "user_token_789",
        "operation": "delete",
        "endpoint": "/api/users/456",
        "resource_id": "user_456"
    },

    # Invalid request - no auth token
    {
        "operation": "read",
        "endpoint": "/api/data"
    },

    # Invalid request - bad operation
    {
        "auth_token": "admin_token_123",
        "operation": "invalid_op",
        "endpoint": "/api/test"
    }
]

print(f"\n🔗 Processing {len(test_requests)} requests through enterprise chain...")

for i, test_request in enumerate(test_requests):
    print(f"\n--- Request {i+1}: {test_request.get('operation', 'unknown')} at {test_request.get('endpoint', 'unknown')} ---")

    chain_result = processing_chain.handle(test_request)

    if chain_result.success:
        processed_data = chain_result.value
        print("✅ Enterprise chain processing successful!")
        print(f"   Operation: {processed_data.get('operation')}")
        print(f"   Status: {processed_data.get('status')}")
        print(f"   Processed by: {processed_data.get('processed_by')}")
        print(f"   User: {processed_data.get('user_id')}")
        if processed_data.get("resource_id"):
            print(f"   Resource: {processed_data['resource_id']}")
    else:
        print(f"❌ Enterprise chain failed: {chain_result.error}")

# Display comprehensive chain metrics
print(f"\n📊 Enterprise Chain Performance Metrics:")
print(f"Authentication:")
print(f"   - Attempts: {auth_handler.auth_attempts}")
print(f"   - Failures: {auth_handler.failed_auths}")
print(f"   - Success rate: {((auth_handler.auth_attempts - auth_handler.failed_auths) / max(auth_handler.auth_attempts, 1)) * 100:.1f}%")

print(f"Authorization:")
print(f"   - Checks: {authz_handler.authz_checks}")
print(f"   - Denied: {authz_handler.denied_requests}")
print(f"   - Approval rate: {((authz_handler.authz_checks - authz_handler.denied_requests) / max(authz_handler.authz_checks, 1)) * 100:.1f}%")

print(f"Validation:")
print(f"   - Validations: {validation_handler.validations}")
print(f"   - Failures: {validation_handler.validation_failures}")
print(f"   - Success rate: {((validation_handler.validations - validation_handler.validation_failures) / max(validation_handler.validations, 1)) * 100:.1f}%")

print(f"Business Logic:")
print(f"   - Processed: {business_handler.processed_requests}")
print(f"   - Errors: {business_handler.processing_errors}")
print(f"   - Success rate: {((business_handler.processed_requests - business_handler.processing_errors) / max(business_handler.processed_requests, 1)) * 100:.1f}%")

# Get overall chain metrics
chain_metrics = processing_chain.get_chain_metrics()
print(f"\nOverall Chain:")
print(f"   - Total requests: {chain_metrics.get('total_requests', 0)}")
print(f"   - Successful: {chain_metrics.get('successful_requests', 0)}")
print(f"   - Failed: {chain_metrics.get('failed_requests', 0)}")
print(f"   - Success rate: {(chain_metrics.get('successful_requests', 0) / max(chain_metrics.get('total_requests', 1), 1)) * 100:.1f}%")
```

### Step 3: Handler Registry and Management

#### Centralized Handler Management System

```python
from typing import Dict, List, Optional

class EnterpriseHandlerManager:
    """Comprehensive handler management system with lifecycle and monitoring."""

    def __init__(self):
        self.registry = FlextHandlers.Management.HandlerRegistry()
        self.handler_instances: Dict[str, object] = {}
        self.handler_configs: Dict[str, dict] = {}
        self.handler_metrics: Dict[str, dict] = {}

    def register_handler_ecosystem(
        self,
        handlers_config: List[dict]
    ) -> FlextResult[Dict[str, str]]:
        """Register a complete ecosystem of handlers with configuration."""

        registration_results = {}

        for config in handlers_config:
            handler_name = config["name"]
            handler_type = config["type"]
            handler_config = config.get("config", {})

            try:
                # Create handler instance based on type
                if handler_type == "basic":
                    handler_instance = FlextHandlers.Implementation.BasicHandler(handler_name)
                elif handler_type == "validating":
                    validator_func = config.get("validator_function")
                    if not validator_func:
                        return FlextResult[Dict[str, str]].fail(f"Validator function required for {handler_name}")
                    handler_instance = FlextHandlers.Implementation.ValidatingHandler(handler_name, validator_func)
                elif handler_type == "authorizing":
                    auth_func = config.get("auth_function")
                    if not auth_func:
                        return FlextResult[Dict[str, str]].fail(f"Auth function required for {handler_name}")
                    handler_instance = FlextHandlers.Implementation.AuthorizingHandler(handler_name, auth_func)
                elif handler_type == "custom":
                    handler_class = config.get("handler_class")
                    if not handler_class:
                        return FlextResult[Dict[str, str]].fail(f"Handler class required for {handler_name}")
                    handler_instance = handler_class()
                else:
                    return FlextResult[Dict[str, str]].fail(f"Unknown handler type: {handler_type}")

                # Configure handler if config provided
                if handler_config and hasattr(handler_instance, 'configure'):
                    config_result = handler_instance.configure(handler_config)
                    if config_result.is_failure:
                        return FlextResult[Dict[str, str]].fail(
                            f"Failed to configure {handler_name}: {config_result.error}"
                        )

                # Register with registry
                register_result = self.registry.register(handler_name, handler_instance)
                if register_result.success:
                    registration_results[handler_name] = register_result.value or handler_name

                    # Store handler information
                    self.handler_instances[handler_name] = handler_instance
                    self.handler_configs[handler_name] = handler_config
                    self.handler_metrics[handler_name] = {
                        "registered_at": datetime.datetime.now().isoformat(),
                        "type": handler_type,
                        "status": "active",
                        "requests_processed": 0,
                        "last_used": None
                    }

                    print(f"✅ Handler '{handler_name}' ({handler_type}) registered successfully")
                else:
                    return FlextResult[Dict[str, str]].fail(
                        f"Failed to register {handler_name}: {register_result.error}"
                    )

            except Exception as e:
                return FlextResult[Dict[str, str]].fail(
                    f"Error creating handler {handler_name}: {str(e)}"
                )

        return FlextResult[Dict[str, str]].ok(registration_results)

    def execute_handler_by_name(
        self,
        handler_name: str,
        request_data: dict
    ) -> FlextResult[dict]:
        """Execute handler by name with comprehensive tracking."""

        # Get handler from registry
        handler_result = self.registry.get_handler(handler_name)
        if handler_result.is_failure:
            return FlextResult[dict].fail(f"Handler '{handler_name}' not found: {handler_result.error}")

        handler = handler_result.value

        # Update usage metrics
        if handler_name in self.handler_metrics:
            self.handler_metrics[handler_name]["requests_processed"] += 1
            self.handler_metrics[handler_name]["last_used"] = datetime.datetime.now().isoformat()

        # Execute handler
        start_time = time.time()
        execution_result = handler.handle(request_data)
        duration_ms = (time.time() - start_time) * 1000

        # Update performance metrics
        if handler_name in self.handler_metrics:
            metrics = self.handler_metrics[handler_name]

            if "total_duration_ms" not in metrics:
                metrics["total_duration_ms"] = 0
                metrics["average_duration_ms"] = 0

            metrics["total_duration_ms"] += duration_ms
            metrics["average_duration_ms"] = metrics["total_duration_ms"] / metrics["requests_processed"]

            if execution_result.success:
                metrics["last_success"] = datetime.datetime.now().isoformat()
            else:
                metrics["last_failure"] = datetime.datetime.now().isoformat()
                metrics["last_error"] = execution_result.error

        return execution_result

    def get_handler_health_report(self) -> FlextResult[dict]:
        """Generate comprehensive handler health report."""

        try:
            # Get registry metrics
            registry_handlers = self.registry.list_handlers()

            health_report = {
                "report_generated_at": datetime.datetime.now().isoformat(),
                "total_registered_handlers": len(registry_handlers),
                "active_handlers": 0,
                "inactive_handlers": 0,
                "handlers_with_errors": 0,
                "total_requests_processed": 0,
                "average_response_time_ms": 0,
                "handler_details": {}
            }

            total_duration = 0
            handlers_with_duration = 0

            for handler_name, handler_info in registry_handlers.items():
                metrics = self.handler_metrics.get(handler_name, {})

                handler_detail = {
                    "type": metrics.get("type", "unknown"),
                    "status": metrics.get("status", "unknown"),
                    "registered_at": metrics.get("registered_at"),
                    "requests_processed": metrics.get("requests_processed", 0),
                    "last_used": metrics.get("last_used"),
                    "average_duration_ms": metrics.get("average_duration_ms", 0),
                    "last_success": metrics.get("last_success"),
                    "last_failure": metrics.get("last_failure"),
                    "last_error": metrics.get("last_error")
                }

                # Determine handler health status
                if handler_detail["last_failure"] and not handler_detail["last_success"]:
                    handler_detail["health_status"] = "unhealthy"
                    health_report["handlers_with_errors"] += 1
                elif handler_detail["requests_processed"] > 0:
                    handler_detail["health_status"] = "healthy"
                    health_report["active_handlers"] += 1
                else:
                    handler_detail["health_status"] = "idle"
                    health_report["inactive_handlers"] += 1

                health_report["handler_details"][handler_name] = handler_detail
                health_report["total_requests_processed"] += handler_detail["requests_processed"]

                if handler_detail["average_duration_ms"] > 0:
                    total_duration += handler_detail["average_duration_ms"]
                    handlers_with_duration += 1

            # Calculate overall average response time
            if handlers_with_duration > 0:
                health_report["average_response_time_ms"] = total_duration / handlers_with_duration

            return FlextResult[dict].ok(health_report)

        except Exception as e:
            return FlextResult[dict].fail(f"Failed to generate health report: {str(e)}")

    def cleanup_inactive_handlers(self, max_idle_hours: int = 24) -> FlextResult[List[str]]:
        """Clean up handlers that haven't been used within specified time."""

        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_idle_hours)
            handlers_to_cleanup = []

            for handler_name, metrics in self.handler_metrics.items():
                last_used = metrics.get("last_used")

                if not last_used:  # Never used
                    registered_at = datetime.datetime.fromisoformat(metrics.get("registered_at", ""))
                    if registered_at < cutoff_time:
                        handlers_to_cleanup.append(handler_name)
                else:
                    last_used_time = datetime.datetime.fromisoformat(last_used)
                    if last_used_time < cutoff_time:
                        handlers_to_cleanup.append(handler_name)

            # Remove inactive handlers
            cleaned_handlers = []
            for handler_name in handlers_to_cleanup:
                unregister_result = self.registry.unregister(handler_name)
                if unregister_result.success:
                    # Remove from local tracking
                    self.handler_instances.pop(handler_name, None)
                    self.handler_configs.pop(handler_name, None)
                    self.handler_metrics.pop(handler_name, None)

                    cleaned_handlers.append(handler_name)
                    print(f"🧹 Cleaned up inactive handler: {handler_name}")

            return FlextResult[List[str]].ok(cleaned_handlers)

        except Exception as e:
            return FlextResult[List[str]].fail(f"Cleanup failed: {str(e)}")

# Demonstrate Enterprise Handler Management
print("=== Enterprise Handler Management System ===")
handler_manager = EnterpriseHandlerManager()

# Define handler ecosystem configuration
def email_validator(data: dict) -> FlextResult[None]:
    """Email validation function."""
    email = data.get("email", "")
    if "@" not in email or "." not in email:
        return FlextResult[None].fail("Invalid email format")
    return FlextResult[None].ok(None)

def admin_authorizer(data: dict) -> bool:
    """Admin authorization function."""
    return data.get("user_role") == "admin"

handlers_ecosystem_config = [
    {
        "name": "user_validator",
        "type": "validating",
        "validator_function": email_validator,
        "config": {"validation_level": "strict", "log_level": "DEBUG"}
    },
    {
        "name": "admin_authorizer",
        "type": "authorizing",
        "auth_function": admin_authorizer,
        "config": {"environment": "production", "timeout": 5000}
    },
    {
        "name": "data_processor",
        "type": "basic",
        "config": {"environment": "production", "max_retries": 2}
    },
    {
        "name": "order_processor",
        "type": "custom",
        "handler_class": OrderProcessingHandler,
        "config": {"log_level": "INFO"}
    }
]

# Register handler ecosystem
registration_result = handler_manager.register_handler_ecosystem(handlers_ecosystem_config)
if registration_result.success:
    registered_handlers = registration_result.value
    print(f"✅ Successfully registered {len(registered_handlers)} handlers")
    for name, id in registered_handlers.items():
        print(f"   - {name}: {id}")
else:
    print(f"❌ Handler ecosystem registration failed: {registration_result.error}")

# Test handler execution through management system
test_requests = [
    # Test email validator
    {
        "handler": "user_validator",
        "data": {"email": "valid@example.com", "name": "John Doe"}
    },
    # Test admin authorizer
    {
        "handler": "admin_authorizer",
        "data": {"user_role": "admin", "operation": "delete_user"}
    },
    # Test basic processor
    {
        "handler": "data_processor",
        "data": {"type": "batch_process", "records": [1, 2, 3]}
    },
    # Test custom order processor
    {
        "handler": "order_processor",
        "data": {
            "customer_id": "cust_001",
            "items": [{"name": "Product A", "price": 100, "quantity": 2}]
        }
    }
]

print(f"\n🔧 Testing handlers through management system...")
for test in test_requests:
    handler_name = test["handler"]
    request_data = test["data"]

    print(f"\n--- Testing {handler_name} ---")
    result = handler_manager.execute_handler_by_name(handler_name, request_data)

    if result.success:
        print(f"✅ Handler {handler_name} executed successfully")
        response = result.value
        if isinstance(response, dict):
            for key, value in response.items():
                if key not in ["data", "items"]:  # Skip large data dumps
                    print(f"   {key}: {value}")
    else:
        print(f"❌ Handler {handler_name} failed: {result.error}")

# Generate comprehensive health report
print(f"\n📊 Generating Handler Health Report...")
health_result = handler_manager.get_handler_health_report()

if health_result.success:
    health_report = health_result.value

    print(f"Handler Ecosystem Health Report:")
    print(f"   Report generated: {health_report['report_generated_at']}")
    print(f"   Total handlers: {health_report['total_registered_handlers']}")
    print(f"   Active: {health_report['active_handlers']}")
    print(f"   Idle: {health_report['inactive_handlers']}")
    print(f"   With errors: {health_report['handlers_with_errors']}")
    print(f"   Total requests processed: {health_report['total_requests_processed']}")
    print(f"   Average response time: {health_report['average_response_time_ms']:.2f}ms")

    print(f"\nHandler Details:")
    for handler_name, details in health_report["handler_details"].items():
        print(f"   {handler_name} ({details['type']}):")
        print(f"     - Health: {details['health_status']}")
        print(f"     - Requests: {details['requests_processed']}")
        print(f"     - Avg time: {details['average_duration_ms']:.2f}ms")
        print(f"     - Last used: {details.get('last_used', 'Never')}")
        if details.get("last_error"):
            print(f"     - Last error: {details['last_error']}")

# Demonstrate handler cleanup
print(f"\n🧹 Testing handler cleanup (simulating 25-hour idle time)...")
cleanup_result = handler_manager.cleanup_inactive_handlers(max_idle_hours=0)  # Immediate cleanup for demo
if cleanup_result.success:
    cleaned = cleanup_result.value
    if cleaned:
        print(f"   Cleaned up handlers: {cleaned}")
    else:
        print("   No handlers needed cleanup (all recently active)")
else:
    print(f"   Cleanup failed: {cleanup_result.error}")
```

---

## ⚡ Advanced Implementation Patterns

### Complete Enterprise CQRS System with Event Sourcing

```python
from dataclasses import dataclass, field
from typing import Dict, List, object, Optional
from enum import Enum
import json
import uuid

class EventType(Enum):
    """Domain event types."""
    USER_CREATED = "UserCreated"
    USER_UPDATED = "UserUpdated"
    USER_DELETED = "UserDeleted"
    ORDER_CREATED = "OrderCreated"
    ORDER_SHIPPED = "OrderShipped"
    PAYMENT_PROCESSED = "PaymentProcessed"

@dataclass
class DomainEvent:
    """Base domain event with event sourcing metadata."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = field(default=EventType.USER_CREATED)
    aggregate_id: str = ""
    event_version: int = 1
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    event_data: Dict[str, object] = field(default_factory=dict)
    correlation_id: str = ""
    causation_id: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

class EventStore:
    """In-memory event store for event sourcing."""

    def __init__(self):
        self.events: List[DomainEvent] = []
        self.snapshots: Dict[str, Dict[str, object]] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {}

    def append_event(self, event: DomainEvent) -> FlextResult[str]:
        """Append event to the event store."""
        try:
            self.events.append(event)

            # Notify event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"⚠️ Event handler error: {e}")

            return FlextResult[str].ok(event.event_id)
        except Exception as e:
            return FlextResult[str].fail(f"Failed to append event: {e}")

    def get_events_for_aggregate(self, aggregate_id: str) -> List[DomainEvent]:
        """Get all events for a specific aggregate."""
        return [e for e in self.events if e.aggregate_id == aggregate_id]

    def get_events_by_type(self, event_type: EventType) -> List[DomainEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def subscribe_to_event(self, event_type: EventType, handler: Callable):
        """Subscribe handler to specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

class EnterpriseAggregateRoot:
    """Base class for aggregates with event sourcing."""

    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[DomainEvent] = []

    def apply_event(self, event: DomainEvent):
        """Apply event to aggregate state."""
        self._handle_event(event)
        self.version += 1

    def raise_event(self, event_type: EventType, event_data: Dict[str, object], correlation_id: str = ""):
        """Raise new domain event."""
        event = DomainEvent(
            event_type=event_type,
            aggregate_id=self.aggregate_id,
            event_version=self.version + 1,
            event_data=event_data,
            correlation_id=correlation_id
        )
        self.uncommitted_events.append(event)
        self.apply_event(event)

    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Get events that haven't been persisted yet."""
        return list(self.uncommitted_events)

    def mark_events_as_committed(self):
        """Mark events as committed to event store."""
        self.uncommitted_events.clear()

    def _handle_event(self, event: DomainEvent):
        """Handle event - to be implemented by subclasses."""
        pass

class User(EnterpriseAggregateRoot):
    """User aggregate with event sourcing."""

    def __init__(self, user_id: str):
        super().__init__(user_id)
        self.name = ""
        self.email = ""
        self.department = ""
        self.status = "active"
        self.created_at: Optional[datetime.datetime] = None
        self.updated_at: Optional[datetime.datetime] = None

    def create_user(self, name: str, email: str, department: str, correlation_id: str = ""):
        """Create new user - raises UserCreated event."""
        if self.created_at:  # Already created
            raise ValueError(f"User {self.aggregate_id} already exists")

        event_data = {
            "name": name,
            "email": email,
            "department": department,
            "status": "active",
            "created_at": datetime.datetime.now().isoformat()
        }

        self.raise_event(EventType.USER_CREATED, event_data, correlation_id)

    def update_user(self, updates: Dict[str, object], correlation_id: str = ""):
        """Update user - raises UserUpdated event."""
        if not self.created_at:
            raise ValueError(f"User {self.aggregate_id} does not exist")

        event_data = {
            "updates": updates,
            "updated_at": datetime.datetime.now().isoformat()
        }

        self.raise_event(EventType.USER_UPDATED, event_data, correlation_id)

    def delete_user(self, correlation_id: str = ""):
        """Delete user - raises UserDeleted event."""
        if not self.created_at:
            raise ValueError(f"User {self.aggregate_id} does not exist")

        event_data = {
            "deleted_at": datetime.datetime.now().isoformat(),
            "previous_status": self.status
        }

        self.raise_event(EventType.USER_DELETED, event_data, correlation_id)

    def _handle_event(self, event: DomainEvent):
        """Handle domain events to update aggregate state."""
        if event.event_type == EventType.USER_CREATED:
            data = event.event_data
            self.name = data["name"]
            self.email = data["email"]
            self.department = data["department"]
            self.status = data["status"]
            self.created_at = datetime.datetime.fromisoformat(data["created_at"])

        elif event.event_type == EventType.USER_UPDATED:
            data = event.event_data
            updates = data["updates"]

            for field, value in updates.items():
                if hasattr(self, field):
                    setattr(self, field, value)

            self.updated_at = datetime.datetime.fromisoformat(data["updated_at"])

        elif event.event_type == EventType.USER_DELETED:
            self.status = "deleted"

class EnterpriseCommandHandler:
    """Enhanced command handler with event sourcing integration."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.aggregates_cache: Dict[str, EnterpriseAggregateRoot] = {}

    def get_aggregate(self, aggregate_id: str, aggregate_class: type) -> EnterpriseAggregateRoot:
        """Load aggregate from event store or cache."""
        if aggregate_id in self.aggregates_cache:
            return self.aggregates_cache[aggregate_id]

        # Reconstruct aggregate from events
        aggregate = aggregate_class(aggregate_id)
        events = self.event_store.get_events_for_aggregate(aggregate_id)

        for event in sorted(events, key=lambda e: e.event_version):
            aggregate.apply_event(event)

        self.aggregates_cache[aggregate_id] = aggregate
        return aggregate

    def save_aggregate(self, aggregate: EnterpriseAggregateRoot) -> FlextResult[None]:
        """Save aggregate events to event store."""
        try:
            uncommitted_events = aggregate.get_uncommitted_events()

            for event in uncommitted_events:
                append_result = self.event_store.append_event(event)
                if append_result.is_failure:
                    return FlextResult[None].fail(f"Failed to save event: {append_result.error}")

            aggregate.mark_events_as_committed()
            self.aggregates_cache[aggregate.aggregate_id] = aggregate

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Error saving aggregate: {e}")

class EnterpriseUserCommandHandler(EnterpriseCommandHandler):
    """User-specific command handler with event sourcing."""

    def handle_create_user_command(self, command: CreateUserCommand) -> FlextResult[str]:
        """Handle user creation with event sourcing."""
        try:
            # Validate command
            validation = command.validate()
            if validation.is_failure:
                return FlextResult[str].fail(validation.error)

            # Generate user ID
            user_id = f"user_{uuid.uuid4().hex[:8]}"

            # Create user aggregate
            user = User(user_id)
            user.create_user(
                command.name,
                command.email,
                command.department,
                correlation_id=f"create_user_cmd_{uuid.uuid4().hex[:8]}"
            )

            # Save events
            save_result = self.save_aggregate(user)
            if save_result.is_failure:
                return FlextResult[str].fail(save_result.error)

            return FlextResult[str].ok(user_id)

        except Exception as e:
            return FlextResult[str].fail(f"Command handling error: {e}")

    def handle_update_user_command(self, command: UpdateUserCommand) -> FlextResult[bool]:
        """Handle user update with event sourcing."""
        try:
            # Validate command
            validation = command.validate()
            if validation.is_failure:
                return FlextResult[bool].fail(validation.error)

            # Load user aggregate
            user = self.get_aggregate(command.user_id, User)
            if not user.created_at:
                return FlextResult[bool].fail(f"User {command.user_id} not found")

            # Update user
            user.update_user(
                command.updates,
                correlation_id=f"update_user_cmd_{uuid.uuid4().hex[:8]}"
            )

            # Save events
            save_result = self.save_aggregate(user)
            if save_result.is_failure:
                return FlextResult[bool].fail(save_result.error)

            return FlextResult[bool].ok(True)

        except Exception as e:
            return FlextResult[bool].fail(f"Update command error: {e}")

# Event Handlers and Projections
class UserProjectionHandler:
    """Handler for creating read-model projections from events."""

    def __init__(self):
        self.user_projections: Dict[str, dict] = {}
        self.department_stats: Dict[str, dict] = {}

    def handle_user_created(self, event: DomainEvent) -> FlextResult[None]:
        """Handle UserCreated event for read-model projection."""
        try:
            user_data = event.event_data
            user_projection = {
                "user_id": event.aggregate_id,
                "name": user_data["name"],
                "email": user_data["email"],
                "department": user_data["department"],
                "status": user_data["status"],
                "created_at": user_data["created_at"],
                "updated_at": user_data["created_at"],
                "event_version": event.event_version
            }

            self.user_projections[event.aggregate_id] = user_projection

            # Update department statistics
            dept = user_data["department"]
            if dept not in self.department_stats:
                self.department_stats[dept] = {"user_count": 0, "active_users": 0}

            self.department_stats[dept]["user_count"] += 1
            self.department_stats[dept]["active_users"] += 1

            print(f"📊 User projection created: {user_data['name']} in {dept}")
            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Projection error: {e}")

    def handle_user_updated(self, event: DomainEvent) -> FlextResult[None]:
        """Handle UserUpdated event for read-model projection."""
        try:
            if event.aggregate_id not in self.user_projections:
                return FlextResult[None].fail(f"User projection {event.aggregate_id} not found")

            projection = self.user_projections[event.aggregate_id]
            updates = event.event_data["updates"]

            # Update projection
            for field, value in updates.items():
                if field in projection:
                    old_value = projection[field]
                    projection[field] = value

                    # Handle department changes
                    if field == "department":
                        # Remove from old department
                        if old_value in self.department_stats:
                            self.department_stats[old_value]["user_count"] -= 1

                        # Add to new department
                        if value not in self.department_stats:
                            self.department_stats[value] = {"user_count": 0, "active_users": 0}
                        self.department_stats[value]["user_count"] += 1

            projection["updated_at"] = event.event_data["updated_at"]
            projection["event_version"] = event.event_version

            print(f"📊 User projection updated: {projection['name']}")
            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Projection update error: {e}")

    def get_user_projection(self, user_id: str) -> FlextResult[dict]:
        """Get user read-model projection."""
        if user_id not in self.user_projections:
            return FlextResult[dict].fail(f"User {user_id} not found in projections")

        return FlextResult[dict].ok(dict(self.user_projections[user_id]))

    def get_department_statistics(self) -> FlextResult[Dict[str, dict]]:
        """Get department statistics from projections."""
        return FlextResult[Dict[str, dict]].ok(dict(self.department_stats))

# Complete Enterprise CQRS System Demo
print("=== Complete Enterprise CQRS with Event Sourcing ===")

# Initialize event store and components
event_store = EventStore()
command_handler = EnterpriseUserCommandHandler(event_store)
projection_handler = UserProjectionHandler()

# Subscribe projection handlers to events
event_store.subscribe_to_event(EventType.USER_CREATED, projection_handler.handle_user_created)
event_store.subscribe_to_event(EventType.USER_UPDATED, projection_handler.handle_user_updated)

# Setup CQRS buses
command_bus = FlextHandlers.CQRS.CommandBus()
query_bus = FlextHandlers.CQRS.QueryBus()
event_bus = FlextHandlers.CQRS.EventBus()

# Register command handlers
command_bus.register(CreateUserCommand, command_handler.handle_create_user_command)
command_bus.register(UpdateUserCommand, command_handler.handle_update_user_command)

# Register query handlers
query_bus.register("GetUserProjection", projection_handler.get_user_projection)
query_bus.register("GetDepartmentStats", projection_handler.get_department_statistics)

# Execute complete CQRS workflow
print("\n1. Creating users through command bus...")

users_to_create = [
    CreateUserCommand(name="Alice Johnson", email="alice@company.com", department="Engineering"),
    CreateUserCommand(name="Bob Smith", email="bob@company.com", department="Sales"),
    CreateUserCommand(name="Carol Davis", email="carol@company.com", department="Engineering"),
    CreateUserCommand(name="David Wilson", email="david@company.com", department="Marketing")
]

created_user_ids = []
for create_cmd in users_to_create:
    result = command_bus.send(create_cmd)
    if result.success:
        user_id = result.value
        created_user_ids.append(user_id)
        print(f"✅ Created user: {create_cmd.name} -> {user_id}")
    else:
        print(f"❌ Failed to create {create_cmd.name}: {result.error}")

print(f"\n2. Updating users through command bus...")

# Update some users
if created_user_ids:
    update_cmd = UpdateUserCommand(
        user_id=created_user_ids[0],
        updates={"department": "Senior Engineering", "title": "Senior Developer"}
    )
    update_result = command_bus.send(update_cmd)
    if update_result.success:
        print(f"✅ Updated user {created_user_ids[0]}")
    else:
        print(f"❌ Update failed: {update_result.error}")

print(f"\n3. Querying read models...")

# Query user projections
for user_id in created_user_ids[:2]:  # Query first 2 users
    projection_result = query_bus.execute("GetUserProjection", user_id)
    if projection_result.success:
        projection = projection_result.value
        print(f"✅ User projection {user_id}: {projection['name']} in {projection['department']}")
    else:
        print(f"❌ Query failed: {projection_result.error}")

# Query department statistics
stats_result = query_bus.execute("GetDepartmentStats", {})
if stats_result.success:
    dept_stats = stats_result.value
    print(f"\n📊 Department Statistics:")
    for dept, stats in dept_stats.items():
        print(f"   {dept}: {stats['user_count']} users")
else:
    print(f"❌ Department stats query failed: {stats_result.error}")

print(f"\n4. Event store analysis...")
print(f"   Total events in store: {len(event_store.events)}")
print(f"   Event types: {set(e.event_type.value for e in event_store.events)}")

# Show event sourcing capabilities
if created_user_ids:
    user_id = created_user_ids[0]
    user_events = event_store.get_events_for_aggregate(user_id)
    print(f"\n   Events for user {user_id}:")
    for event in user_events:
        print(f"     - {event.event_type.value} (v{event.event_version}) at {event.timestamp}")
```

---

## 📋 Implementation Checklist

### Phase 1: Foundation Setup

- [ ] **Handler Architecture Understanding**: Study 7-layer architecture and design pattern integration
- [ ] **FlextResult Integration**: Ensure all handler operations return FlextResult[T]
- [ ] **Type Safety Implementation**: Define generic type parameters for custom handlers
- [ ] **Configuration Integration**: Setup FlextTypes.Config integration for environment-aware behavior

### Phase 2: Core Handler Implementation

- [ ] **Basic Handler Extension**: Implement custom handlers extending AbstractHandler[TInput, TOutput]
- [ ] **Validation Patterns**: Create ValidatingHandler implementations with comprehensive business rules
- [ ] **Authorization Systems**: Implement AuthorizingHandler with role-based access control
- [ ] **Metrics Collection**: Enable comprehensive performance monitoring and metrics collection

### Phase 3: CQRS Implementation

- [ ] **Command Modeling**: Define domain commands with Pydantic validation
- [ ] **Command Handlers**: Implement command handlers with business logic and validation
- [ ] **Query Modeling**: Create query models with pagination and filtering support
- [ ] **Query Handlers**: Implement read-side query handlers with result transformation
- [ ] **Event Modeling**: Define domain events for event sourcing and notification
- [ ] **Event Handlers**: Create event handlers for projections and side effects

### Phase 4: Pattern Implementation

- [ ] **Handler Chains**: Build Chain of Responsibility patterns for processing pipelines
- [ ] **Pipeline Processing**: Implement linear processing pipelines with validation stages
- [ ] **Middleware Integration**: Create middleware for request/response transformation
- [ ] **Registry Management**: Setup centralized handler registry with discovery capabilities

### Phase 5: Advanced Features

- [ ] **Event Sourcing**: Implement event sourcing with aggregate reconstruction
- [ ] **Read Model Projections**: Create projections for query-side read models
- [ ] **Saga Patterns**: Implement distributed transaction patterns with compensation
- [ ] **Circuit Breakers**: Add resilience patterns for fault tolerance

### Phase 6: Enterprise Integration

- [ ] **Thread Safety Validation**: Ensure all operations use thread_safe_operation() context manager
- [ ] **Performance Optimization**: Optimize handler chains and CQRS buses for high throughput
- [ ] **Monitoring Integration**: Setup comprehensive observability and alerting
- [ ] **Documentation**: Create comprehensive handler documentation and examples

### Phase 7: Production Readiness

- [ ] **Load Testing**: Validate performance under realistic load conditions
- [ ] **Security Validation**: Verify security patterns and access control implementation
- [ ] **Error Handling**: Test comprehensive error scenarios and recovery patterns
- [ ] **Deployment**: Deploy with proper configuration for target environment
- [ ] **Team Training**: Train development team on FlextHandlers patterns and best practices

This implementation guide provides comprehensive coverage of FlextHandlers enterprise patterns, from basic handler creation through advanced CQRS and event sourcing implementations, ensuring teams can leverage the full power of the FlextHandlers ecosystem for sophisticated request processing and enterprise architecture patterns.
