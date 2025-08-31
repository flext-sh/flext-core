# FlextCore Implementation Guide

**Step-by-step guide for implementing FlextCore as the central orchestration hub in FLEXT ecosystem projects.**

---

## Overview

This guide provides comprehensive instructions for integrating FlextCore into new and existing FLEXT projects. FlextCore serves as the enterprise orchestration hub, providing unified access to dependency injection, domain modeling, validation, logging, and all architectural patterns through a thread-safe singleton interface.

---

## Quick Start Implementation

### Step 1: Basic Setup

```python
from flext_core.core import FlextCore

# Initialize the singleton instance
core = FlextCore.get_instance()

# Configure logging for your environment
core.configure_logging(log_level="INFO", _json_output=True)

# Verify system health
health = core.health_check()
if health.success:
    print("✅ FlextCore initialized successfully")
    print(f"System info: {core.get_system_info()}")
else:
    print(f"❌ Health check failed: {health.error}")
```

### Step 2: Service Registration

```python
# Register your core services
services = {
    "database": DatabaseService,
    "cache": CacheService,
    "user_repository": UserRepository,
    "notification_service": NotificationService
}

# Setup container with validation
container_result = core.setup_container_with_services(
    services,
    validator=core.validate_service_name
)

if container_result.success:
    print("✅ All services registered successfully")
else:
    print(f"❌ Service registration failed: {container_result.error}")
```

### Step 3: Environment Configuration

```python
# Load environment-specific configuration
config_result = core.create_environment_core_config("production")
if config_result.success:
    # Optimize for production workload
    optimized = core.optimize_core_performance({
        "performance_level": "high",
        "memory_limit_mb": 2048,
        "cpu_cores": 16
    })
    
    if optimized.success:
        # Apply configuration
        core.configure_core_system(optimized.value)
        print("✅ Production configuration applied")
```

---

## Detailed Implementation Patterns

### 1. Dependency Injection Implementation

#### Service Registration Patterns

```python
from flext_core.core import FlextCore

class ApplicationBootstrap:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def register_infrastructure_services(self):
        """Register infrastructure layer services."""
        # Database services
        self.core.register_service("primary_db", PostgreSQLService())
        self.core.register_service("cache_db", RedisService())
        
        # External API clients
        self.core.register_factory(
            "payment_gateway", 
            lambda: PaymentGateway(api_key=self.get_api_key())
        )
        
        # Message queues
        self.core.register_factory(
            "message_queue",
            lambda: RabbitMQService(host="localhost", port=5672)
        )
    
    def register_domain_services(self):
        """Register domain layer services."""
        services = {
            "user_service": UserService,
            "order_service": OrderService,
            "payment_service": PaymentService,
            "inventory_service": InventoryService
        }
        
        result = self.core.setup_container_with_services(
            services,
            validator=self.core.validate_service_name
        )
        
        if result.failure:
            raise RuntimeError(f"Domain service registration failed: {result.error}")
    
    def register_application_services(self):
        """Register application layer services."""
        # Command handlers
        self.core.register_service("create_user_handler", CreateUserHandler())
        self.core.register_service("process_order_handler", ProcessOrderHandler())
        
        # Query handlers
        self.core.register_service("user_query_handler", UserQueryHandler())
        self.core.register_service("order_query_handler", OrderQueryHandler())
    
    def bootstrap_application(self):
        """Complete application bootstrap sequence."""
        try:
            # Configure logging first
            self.core.configure_logging(log_level="INFO", _json_output=True)
            
            # Register services in dependency order
            self.register_infrastructure_services()
            self.register_domain_services()
            self.register_application_services()
            
            # Validate system health
            health = self.core.health_check()
            if health.failure:
                raise RuntimeError(f"System health check failed: {health.error}")
            
            print("✅ Application bootstrap completed successfully")
            
        except Exception as e:
            self.core.log_error("Bootstrap failed", error=str(e))
            raise
```

#### Service Retrieval Patterns

```python
class UserController:
    def __init__(self):
        self.core = FlextCore.get_instance()
        
        # Retrieve services with error handling
        self.user_service = self._get_required_service("user_service")
        self.notification_service = self._get_required_service("notification_service")
        self.logger = self.core.logger
    
    def _get_required_service(self, service_name: str):
        """Retrieve required service with error handling."""
        service_result = self.core.get_service(service_name)
        if service_result.failure:
            raise RuntimeError(f"Required service '{service_name}' not available: {service_result.error}")
        return service_result.value
    
    def create_user(self, user_data: dict) -> FlextResult[dict]:
        """Create user with comprehensive error handling."""
        return (
            self.core.validate_user_data(user_data)
            .flat_map(lambda data: self.user_service.create_user(data))
            .tap(lambda user: self.notification_service.send_welcome_email(user))
            .tap(lambda user: self.logger.info("User created", user_id=user.id))
            .map_error(lambda error: self.logger.error("User creation failed", error=str(error)))
        )
```

### 2. Railway-Oriented Programming Implementation

#### Domain Operations with Railway Patterns

```python
class OrderProcessingService:
    def __init__(self):
        self.core = FlextCore.get_instance()
        self.inventory_service = self._get_service("inventory_service")
        self.payment_service = self._get_service("payment_service")
        self.shipping_service = self._get_service("shipping_service")
    
    def _get_service(self, name: str):
        return self.core.get_service(name).unwrap()
    
    def process_order(self, order_data: dict) -> FlextResult[dict]:
        """Process order using railway-oriented programming."""
        correlation_id = self.core.generate_correlation_id()
        
        return (
            # Validation phase
            self.core.validate_api_request(order_data, required_fields=["customer_id", "items"])
            .flat_map(lambda data: self._validate_customer(data["customer_id"]))
            .flat_map(lambda customer: self._validate_items(order_data["items"]))
            
            # Business logic phase
            .flat_map(lambda items: self._check_inventory(items))
            .flat_map(lambda items: self._calculate_pricing(items))
            .flat_map(lambda pricing: self._process_payment(pricing))
            
            # Fulfillment phase
            .flat_map(lambda payment: self._reserve_inventory(order_data["items"]))
            .flat_map(lambda reservation: self._create_shipment(order_data))
            .flat_map(lambda shipment: self._finalize_order(order_data, shipment))
            
            # Logging and monitoring
            .tap(lambda order: self._log_success(order, correlation_id))
            .map_error(lambda error: self._log_error(error, correlation_id))
        )
    
    def _validate_customer(self, customer_id: str) -> FlextResult[dict]:
        """Validate customer existence and status."""
        customer_result = self.core.get_service("customer_service")
        if customer_result.failure:
            return FlextResult.fail("Customer service unavailable")
        
        customer_service = customer_result.value
        return customer_service.get_customer(customer_id)
    
    def _validate_items(self, items: list) -> FlextResult[list]:
        """Validate order items."""
        if not items:
            return FlextResult.fail("Order must contain at least one item")
        
        validated_items = []
        for item in items:
            item_result = (
                self.core.require_not_none(item.get("product_id"), "Product ID required")
                .flat_map(lambda _: self.core.require_positive(item.get("quantity"), "Quantity must be positive"))
                .map(lambda _: item)
            )
            if item_result.failure:
                return item_result
            validated_items.append(item_result.value)
        
        return FlextResult.ok(validated_items)
    
    def _log_success(self, order: dict, correlation_id: str):
        """Log successful order processing."""
        self.core.log_info(
            "Order processed successfully",
            order_id=order["id"],
            correlation_id=correlation_id,
            total_amount=order.get("total")
        )
    
    def _log_error(self, error: str, correlation_id: str):
        """Log order processing error."""
        self.core.log_error(
            "Order processing failed",
            error=error,
            correlation_id=correlation_id
        )
        return error
```

#### Result Composition Patterns

```python
class DataPipelineService:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def process_data_pipeline(self, input_data: dict) -> FlextResult[dict]:
        """Complex data pipeline with multiple validation and transformation steps."""
        
        # Sequential processing pipeline
        sequential_result = (
            self._extract_data(input_data)
            .flat_map(self._validate_schema)
            .flat_map(self._transform_data)
            .flat_map(self._enrich_data)
            .flat_map(self._load_data)
        )
        
        # Parallel validation pipeline
        validation_results = [
            self._validate_data_quality(input_data),
            self._validate_business_rules(input_data),
            self._validate_compliance(input_data)
        ]
        
        parallel_result = self.core.sequence(validation_results)
        
        # Combine sequential and parallel results
        return (
            parallel_result
            .flat_map(lambda _: sequential_result)
            .tap(lambda result: self._log_pipeline_success(result))
            .map_error(lambda error: self._log_pipeline_error(error))
        )
    
    def _extract_data(self, input_data: dict) -> FlextResult[dict]:
        """Extract data with error handling."""
        try:
            extracted = {
                "records": input_data.get("records", []),
                "metadata": input_data.get("metadata", {}),
                "timestamp": self.core.generate_correlation_id()
            }
            return FlextResult.ok(extracted)
        except Exception as e:
            return FlextResult.fail(f"Data extraction failed: {str(e)}")
```

### 3. Domain Modeling Implementation

#### Entity Creation and Management

```python
class DomainModelFactory:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def create_user_aggregate(self, user_data: dict) -> FlextResult[User]:
        """Create user aggregate root with validation."""
        return (
            # Validate input data
            self.core.validate_user_data(user_data, required_fields=["name", "email"])
            
            # Create value objects
            .flat_map(lambda data: self._create_email_value(data["email"]))
            .flat_map(lambda email: self._create_name_value(user_data["name"]))
            
            # Create entity
            .flat_map(lambda name: self.core.create_entity(
                User,
                id=self.core.generate_entity_id(),
                name=name,
                email=email,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(UTC)
            ))
            
            # Add domain events
            .tap(lambda user: self._add_user_created_event(user))
        )
    
    def _create_email_value(self, email: str) -> FlextResult[EmailAddress]:
        """Create email address value object."""
        return (
            self.core.validate_email(email)
            .flat_map(lambda _: self.core.create_value_object(
                EmailAddress,
                value=email,
                domain=email.split("@")[1] if "@" in email else None
            ))
        )
    
    def _create_name_value(self, name: str) -> FlextResult[PersonName]:
        """Create person name value object."""
        return (
            self.core.validate_string(name, min_length=2, max_length=100)
            .flat_map(lambda _: self.core.create_value_object(
                PersonName,
                full_name=name,
                first_name=name.split()[0] if name.split() else name,
                last_name=" ".join(name.split()[1:]) if len(name.split()) > 1 else ""
            ))
        )
    
    def _add_user_created_event(self, user: User):
        """Add domain event for user creation."""
        event_result = self.core.create_domain_event(
            "UserCreated",
            {
                "user_id": user.id,
                "email": user.email.value,
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": self.core.generate_correlation_id()
            }
        )
        
        if event_result.success:
            user.add_domain_event(event_result.value)
```

#### Dynamic Class Generation

```python
class DynamicModelBuilder:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def build_product_entity(self, schema: dict) -> type:
        """Dynamically build product entity based on schema."""
        
        # Define field validations based on schema
        field_validators = {}
        if "name" in schema:
            field_validators["name"] = lambda x: self.core.validate_string(x, min_length=1, max_length=200)
        if "price" in schema:
            field_validators["price"] = lambda x: self.core.validate_numeric(x, min_value=0)
        if "category" in schema:
            field_validators["category"] = lambda x: self.core.validate_string(x, min_length=1)
        
        # Define business rules
        def validate_product_business_rules(product):
            rules = []
            if hasattr(product, "price") and hasattr(product, "discount_price"):
                if product.discount_price >= product.price:
                    return FlextResult.fail("Discount price must be less than regular price")
            return FlextResult.ok(None)
        
        # Create dynamic entity class
        ProductEntity = self.core.create_entity_with_validators(
            "Product",
            fields=schema,
            validators=field_validators,
            business_rules=validate_product_business_rules
        )
        
        return ProductEntity
    
    def build_service_processor(self, processor_name: str, process_logic: callable) -> type:
        """Build dynamic service processor."""
        
        def enhanced_process_logic(request):
            # Add FlextCore integration to processing logic
            correlation_id = self.core.generate_correlation_id()
            
            result = (
                self.core.validate_api_request(request)
                .flat_map(lambda data: FlextResult.ok(process_logic(data)))
                .tap(lambda result: self.core.log_info(
                    f"{processor_name} completed",
                    correlation_id=correlation_id,
                    result_type=type(result).__name__
                ))
                .map_error(lambda error: self.core.log_error(
                    f"{processor_name} failed",
                    error=str(error),
                    correlation_id=correlation_id
                ))
            )
            
            return result
        
        return self.core.create_service_processor(
            processor_name,
            enhanced_process_logic,
            with_validation=True,
            with_logging=True
        )
```

### 4. Validation Implementation

#### Comprehensive Validation Chains

```python
class ValidationService:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def validate_order_data(self, order_data: dict) -> FlextResult[dict]:
        """Comprehensive order validation with multiple validation layers."""
        
        # Structure validation
        structure_result = self._validate_order_structure(order_data)
        
        # Business rule validation
        business_result = self._validate_business_rules(order_data)
        
        # Data integrity validation
        integrity_result = self._validate_data_integrity(order_data)
        
        # Combine all validation results
        validation_results = [structure_result, business_result, integrity_result]
        combined_result = self.core.sequence(validation_results)
        
        return combined_result.map(lambda _: order_data)
    
    def _validate_order_structure(self, data: dict) -> FlextResult[None]:
        """Validate order data structure."""
        required_fields = ["customer_id", "items", "shipping_address"]
        
        return (
            self.core.validate_api_request(data, required_fields=required_fields)
            .flat_map(lambda _: self._validate_items_structure(data.get("items", [])))
            .flat_map(lambda _: self._validate_address_structure(data.get("shipping_address", {})))
        )
    
    def _validate_business_rules(self, data: dict) -> FlextResult[None]:
        """Validate business rules for order."""
        validations = []
        
        # Minimum order amount
        total_amount = self._calculate_total_amount(data.get("items", []))
        if total_amount < 10.00:
            validations.append(FlextResult.fail("Order total must be at least $10.00"))
        else:
            validations.append(FlextResult.ok(None))
        
        # Customer validation
        customer_validation = self._validate_customer_eligibility(data.get("customer_id"))
        validations.append(customer_validation)
        
        # Item availability validation
        availability_validation = self._validate_item_availability(data.get("items", []))
        validations.append(availability_validation)
        
        return self.core.sequence(validations).map(lambda _: None)
    
    def _validate_customer_eligibility(self, customer_id: str) -> FlextResult[None]:
        """Validate customer eligibility to place orders."""
        return (
            self.core.require_not_none(customer_id, "Customer ID is required")
            .flat_map(lambda _: self._check_customer_status(customer_id))
            .flat_map(lambda _: self._check_customer_credit(customer_id))
        )
    
    def create_validation_chain(self, *validators) -> callable:
        """Create a reusable validation chain."""
        def validate_data(data):
            results = [validator(data) for validator in validators]
            return self.core.sequence(results).map(lambda _: data)
        
        return validate_data
```

### 5. Configuration Management Implementation

#### Environment-Specific Configuration

```python
class ConfigurationManager:
    def __init__(self):
        self.core = FlextCore.get_instance()
        self._config_cache = {}
    
    def setup_application_config(self, environment: str) -> FlextResult[dict]:
        """Setup complete application configuration for environment."""
        
        # Create base environment configuration
        base_config_result = self.core.create_environment_core_config(environment)
        if base_config_result.failure:
            return base_config_result
        
        # Load environment variables
        env_config_result = self.core.load_config_from_env(prefix="APP_")
        if env_config_result.failure:
            return env_config_result
        
        # Merge configurations
        merged_config_result = self.core.merge_configs(
            base_config_result.value,
            env_config_result.value,
            self._get_application_defaults()
        )
        
        if merged_config_result.failure:
            return merged_config_result
        
        # Validate final configuration
        validation_result = self.core.validate_config_with_types(
            merged_config_result.value,
            required_keys=["database_url", "api_key", "log_level"]
        )
        
        if validation_result.failure:
            return validation_result
        
        # Apply performance optimizations
        if environment == "production":
            optimization_result = self._apply_production_optimizations(
                merged_config_result.value
            )
            if optimization_result.failure:
                return optimization_result
            
            return optimization_result
        
        return merged_config_result
    
    def _get_application_defaults(self) -> dict:
        """Get application-specific default configuration."""
        return {
            "log_level": "INFO",
            "debug_mode": False,
            "max_connections": 100,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "cache_ttl": 3600,
            "batch_size": 100
        }
    
    def _apply_production_optimizations(self, config: dict) -> FlextResult[dict]:
        """Apply production-specific optimizations."""
        optimization_settings = {
            "performance_level": "high",
            "memory_limit_mb": 2048,
            "cpu_cores": 8,
            "enable_caching": True,
            "enable_metrics": True
        }
        
        optimized_result = self.core.optimize_core_performance(optimization_settings)
        if optimized_result.failure:
            return optimized_result
        
        return self.core.merge_configs(config, optimized_result.value)
    
    def get_cached_config(self, environment: str) -> FlextResult[dict]:
        """Get cached configuration for environment."""
        cache_key = f"config_{environment}"
        
        if cache_key in self._config_cache:
            return FlextResult.ok(self._config_cache[cache_key])
        
        config_result = self.setup_application_config(environment)
        if config_result.success:
            self._config_cache[cache_key] = config_result.value
        
        return config_result
```

### 6. Observability Implementation

#### Comprehensive Monitoring Setup

```python
class ObservabilityManager:
    def __init__(self):
        self.core = FlextCore.get_instance()
        self._setup_observability()
    
    def _setup_observability(self):
        """Setup comprehensive observability system."""
        # Configure structured logging
        self.core.configure_logging(log_level="INFO", _json_output=True)
        
        # Setup performance tracking
        self.logger = self.core.logger
        self.observability = self.core.observability
        
    def monitor_operation(self, operation_name: str):
        """Decorator for comprehensive operation monitoring."""
        def decorator(func):
            @self.core.track_performance(operation_name)
            def wrapper(*args, **kwargs):
                correlation_id = self.core.generate_correlation_id()
                
                # Log operation start
                self.logger.info(
                    f"Starting operation: {operation_name}",
                    correlation_id=correlation_id,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
                
                try:
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    # Log success
                    self.logger.info(
                        f"Operation completed: {operation_name}",
                        correlation_id=correlation_id,
                        result_type=type(result).__name__
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error with full context
                    self.logger.error(
                        f"Operation failed: {operation_name}",
                        correlation_id=correlation_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise
            
            return wrapper
        return decorator
    
    def setup_health_checks(self) -> FlextResult[None]:
        """Setup comprehensive health checks."""
        health_checks = [
            self._check_database_connectivity,
            self._check_cache_connectivity,
            self._check_external_apis,
            self._check_message_queues
        ]
        
        # Register health checks
        for check in health_checks:
            check_result = check()
            if check_result.failure:
                self.logger.error(
                    "Health check failed",
                    check_name=check.__name__,
                    error=check_result.error
                )
        
        # Perform system health check
        system_health = self.core.health_check()
        if system_health.failure:
            return FlextResult.fail(f"System health check failed: {system_health.error}")
        
        self.logger.info("All health checks passed")
        return FlextResult.ok(None)
    
    def _check_database_connectivity(self) -> FlextResult[None]:
        """Check database connectivity."""
        db_service_result = self.core.get_service("database")
        if db_service_result.failure:
            return FlextResult.fail("Database service not available")
        
        # Perform actual connectivity check
        try:
            db_service = db_service_result.value
            if hasattr(db_service, "ping"):
                db_service.ping()
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Database connectivity failed: {str(e)}")
```

---

## Advanced Implementation Patterns

### 1. Plugin Architecture Integration

```python
class PluginManager:
    def __init__(self):
        self.core = FlextCore.get_instance()
        self._plugins = {}
    
    def register_plugin(self, name: str, plugin_class: type) -> FlextResult[None]:
        """Register a plugin with the core system."""
        # Validate plugin
        validation_result = self._validate_plugin(plugin_class)
        if validation_result.failure:
            return validation_result
        
        # Create plugin instance
        try:
            plugin_instance = plugin_class(core=self.core)
            
            # Register plugin services
            if hasattr(plugin_instance, "get_services"):
                services = plugin_instance.get_services()
                for service_name, service_instance in services.items():
                    service_key = f"plugin_{name}_{service_name}"
                    self.core.register_service(service_key, service_instance)
            
            # Store plugin reference
            self._plugins[name] = plugin_instance
            
            self.core.log_info(f"Plugin registered: {name}")
            return FlextResult.ok(None)
            
        except Exception as e:
            return FlextResult.fail(f"Plugin registration failed: {str(e)}")
    
    def _validate_plugin(self, plugin_class: type) -> FlextResult[None]:
        """Validate plugin implementation."""
        required_methods = ["initialize", "get_name", "get_version"]
        
        for method in required_methods:
            if not hasattr(plugin_class, method):
                return FlextResult.fail(f"Plugin missing required method: {method}")
        
        return FlextResult.ok(None)
```

### 2. Cross-Service Communication

```python
class MessageBus:
    def __init__(self):
        self.core = FlextCore.get_instance()
        self._handlers = {}
    
    def publish_event(self, event_type: str, data: dict) -> FlextResult[None]:
        """Publish event to registered handlers."""
        # Create domain event
        event_result = self.core.create_domain_event(event_type, data)
        if event_result.failure:
            return event_result
        
        event = event_result.value
        
        # Find handlers for event type
        handlers = self._handlers.get(event_type, [])
        
        # Process handlers
        results = []
        for handler in handlers:
            try:
                result = handler(event)
                results.append(result)
            except Exception as e:
                results.append(FlextResult.fail(f"Handler failed: {str(e)}"))
        
        # Combine results
        combined_result = self.core.sequence(results)
        
        if combined_result.success:
            self.core.log_info(
                "Event published successfully",
                event_type=event_type,
                handlers_count=len(handlers)
            )
        else:
            self.core.log_error(
                "Event publishing failed",
                event_type=event_type,
                error=combined_result.error
            )
        
        return combined_result.map(lambda _: None)
```

---

## Testing Implementation

### Unit Testing with FlextCore

```python
import pytest
from flext_core.core import FlextCore

class TestFlextCoreIntegration:
    """Test FlextCore integration patterns."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Reset singleton for testing
        FlextCore._instance = None
        
        # Get fresh instance
        self.core = FlextCore.get_instance()
        
        # Configure for testing
        self.core.configure_logging(log_level="DEBUG", _json_output=False)
        
        yield
        
        # Cleanup
        self.core.reset_all_caches()
    
    def test_service_registration_and_retrieval(self):
        """Test service registration and retrieval."""
        # Mock service
        class MockService:
            def get_data(self):
                return "test_data"
        
        # Register service
        registration_result = self.core.register_service("test_service", MockService())
        assert registration_result.success
        
        # Retrieve service
        service_result = self.core.get_service("test_service")
        assert service_result.success
        assert service_result.value.get_data() == "test_data"
    
    def test_railway_oriented_programming(self):
        """Test railway-oriented programming patterns."""
        # Test successful chain
        result = (
            self.core.ok("test_data")
            .map(lambda x: x.upper())
            .map(lambda x: f"processed_{x}")
            .flat_map(lambda x: self.core.ok(f"final_{x}"))
        )
        
        assert result.success
        assert result.value == "final_processed_TEST_DATA"
        
        # Test failure chain
        failure_result = (
            self.core.fail("initial_error")
            .map(lambda x: x.upper())  # Should not execute
            .map_error(lambda e: f"handled_{e}")
        )
        
        assert failure_result.failure
        assert "handled_initial_error" in failure_result.error
    
    def test_domain_modeling(self):
        """Test domain modeling capabilities."""
        # Create entity
        entity_result = self.core.create_entity(
            dict,  # Using dict as simple entity for testing
            id="test_id",
            name="test_name"
        )
        
        assert entity_result.success
        entity = entity_result.value
        assert entity["id"] == "test_id"
        assert entity["name"] == "test_name"
```

---

## Performance Optimization

### Production Configuration

```python
class ProductionOptimizer:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def optimize_for_production(self) -> FlextResult[None]:
        """Apply production optimizations."""
        # Create production configuration
        prod_config_result = self.core.create_environment_core_config("production")
        if prod_config_result.failure:
            return prod_config_result
        
        # Apply performance optimizations
        perf_config = {
            "performance_level": "high",
            "memory_limit_mb": 4096,
            "cpu_cores": 16,
            "enable_caching": True,
            "cache_size": 1000,
            "enable_metrics": True,
            "metrics_buffer_size": 10000
        }
        
        optimized_result = self.core.optimize_core_performance(perf_config)
        if optimized_result.failure:
            return optimized_result
        
        # Merge and apply configuration
        final_config_result = self.core.merge_configs(
            prod_config_result.value,
            optimized_result.value
        )
        
        if final_config_result.success:
            self.core.configure_core_system(final_config_result.value)
            self.core.log_info("Production optimizations applied")
        
        return final_config_result.map(lambda _: None)
```

---

## Migration Strategy

### From Legacy Systems

```python
class LegacyMigrationHelper:
    def __init__(self):
        self.core = FlextCore.get_instance()
    
    def migrate_legacy_services(self, legacy_services: dict) -> FlextResult[None]:
        """Migrate legacy services to FlextCore."""
        migration_results = []
        
        for service_name, legacy_service in legacy_services.items():
            # Wrap legacy service
            wrapped_service = self._wrap_legacy_service(legacy_service)
            
            # Register with FlextCore
            registration_result = self.core.register_service(service_name, wrapped_service)
            migration_results.append(registration_result)
        
        # Validate all migrations succeeded
        combined_result = self.core.sequence(migration_results)
        
        if combined_result.success:
            self.core.log_info(f"Successfully migrated {len(legacy_services)} services")
        else:
            self.core.log_error("Service migration failed", error=combined_result.error)
        
        return combined_result.map(lambda _: None)
    
    def _wrap_legacy_service(self, legacy_service):
        """Wrap legacy service to provide FlextResult compatibility."""
        class LegacyServiceWrapper:
            def __init__(self, legacy):
                self.legacy = legacy
                self.core = FlextCore.get_instance()
            
            def __getattr__(self, name):
                attr = getattr(self.legacy, name)
                if callable(attr):
                    return self._wrap_method(attr)
                return attr
            
            def _wrap_method(self, method):
                def wrapped(*args, **kwargs):
                    try:
                        result = method(*args, **kwargs)
                        return FlextResult.ok(result)
                    except Exception as e:
                        self.core.log_error(f"Legacy method failed: {method.__name__}", error=str(e))
                        return FlextResult.fail(str(e))
                return wrapped
        
        return LegacyServiceWrapper(legacy_service)
```

---

## Best Practices Summary

### 1. Initialization
- Always use `FlextCore.get_instance()` for singleton access
- Configure logging early in application startup
- Perform health checks during initialization
- Use environment-specific configuration

### 2. Service Management
- Validate service names using `core.validate_service_name()`
- Use `setup_container_with_services()` for bulk registration
- Handle service retrieval failures gracefully
- Register services in dependency order

### 3. Railway Programming
- Chain operations using `.flat_map()` and `.map()`
- Use `.tap()` for side effects (logging, notifications)
- Handle errors with `.map_error()`
- Combine multiple results with `core.sequence()`

### 4. Error Handling
- Use FlextResult consistently across all operations
- Log errors with correlation IDs
- Provide meaningful error messages
- Implement fallback strategies

### 5. Configuration
- Use environment-specific configurations
- Validate configurations before applying
- Merge configurations safely
- Optimize for production environments

### 6. Monitoring
- Use structured logging with correlation IDs
- Implement comprehensive health checks
- Monitor performance with `@track_performance`
- Set up proper observability

---

This implementation guide provides comprehensive patterns for integrating FlextCore into FLEXT ecosystem projects, ensuring consistent architecture and best practices across all implementations.
