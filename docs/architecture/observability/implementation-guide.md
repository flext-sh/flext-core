# FlextObservability Implementation Guide

**Version**: 0.9.0  
**Module**: `flext_core.observability`  
**Target Audience**: Senior Developers, Platform Engineers, DevOps Engineers  

## Quick Start

This guide provides step-by-step implementation patterns for integrating FlextObservability across FLEXT ecosystem services, from basic setup to enterprise-grade monitoring orchestration.

**Prerequisite**: Ensure `flext-core` is installed and available in your environment.

---

## 🚀 Basic Implementation

### Step 1: Import and Initialize

```python
from flext_core import FlextObservability, FlextResult
import os
import time

# Initialize FlextObservability instance
obs = FlextObservability()

# Basic configuration
config = {
    "service_name": "my-service",
    "environment": os.getenv("ENV", "development"),
    "log_level": os.getenv("LOG_LEVEL", "INFO")
}

# Configure the observability system
config_result = obs.configure_observability_system(config)
if not config_result.success:
    raise Exception(f"Failed to configure observability: {config_result.error}")

print("✅ FlextObservability initialized successfully")
```

### Step 2: Create Basic Components

```python
# Create observability components
logger = obs.create_console_logger("my-service", "INFO")
tracer = obs.create_tracer("my-service")
metrics = obs.create_metrics_collector("my-service")
health = obs.create_health_monitor([])

# Test basic functionality
logger.info("Service starting", service_name="my-service", version="1.0.0")
metrics.increment_counter("service_starts", 1)

print("✅ Basic components created successfully")
```

### Step 3: Implement Basic Monitoring

```python
def monitored_operation(data: dict) -> FlextResult[dict]:
    """Example function with basic observability integration."""
    
    with tracer.trace_operation("data_processing") as span:
        start_time = time.time()
        
        try:
            # Set span context
            span.set_tag("data_size", len(data))
            span.set_tag("operation_type", "processing")
            
            # Simulate processing
            result = {"processed": True, "record_count": len(data)}
            
            # Success metrics and logging
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_histogram("processing_duration_ms", duration_ms)
            metrics.increment_counter("processing_success", 1)
            
            logger.info("Processing completed",
                records_processed=len(data),
                duration_ms=duration_ms
            )
            
            return FlextResult.ok(result)
            
        except Exception as e:
            # Error observability
            metrics.increment_counter("processing_errors", 1)
            logger.exception("Processing failed", error_type=type(e).__name__)
            span.set_tag("error", True)
            
            return FlextResult.fail(f"Processing failed: {e}")

# Test the monitored operation
test_data = {"item1": "value1", "item2": "value2"}
result = monitored_operation(test_data)

if result.success:
    print(f"✅ Operation successful: {result.value}")
else:
    print(f"❌ Operation failed: {result.error}")
```

---

## 🏗️ Enterprise Implementation

### Step 1: Service Base Class

```python
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import threading
from datetime import datetime

class ObservableService(ABC):
    """Base class for FLEXT services with comprehensive observability."""
    
    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize observability
        self._setup_observability()
        
    def _setup_observability(self):
        """Initialize all observability components."""
        
        self.obs = FlextObservability()
        
        # Configure system
        obs_config = {
            "service_name": self.service_name,
            "environment": self.config.get("environment", "development"),
            "log_level": self.config.get("log_level", "INFO"),
            "metrics_enabled": self.config.get("metrics_enabled", True),
            "tracing_enabled": self.config.get("tracing_enabled", True),
            "health_enabled": self.config.get("health_enabled", True)
        }
        
        config_result = self.obs.configure_observability_system(obs_config)
        if not config_result.success:
            raise Exception(f"Observability setup failed: {config_result.error}")
        
        # Create components
        self.logger = self.obs.create_console_logger(self.service_name, obs_config["log_level"])
        self.metrics = self.obs.create_metrics_collector(self.service_name)
        self.tracer = self.obs.create_tracer(self.service_name)
        self.health = self.obs.create_health_monitor([])
        self.alerts = self.obs.create_alert_manager(self.config.get("alerts", {}))
        
        # Setup standard health checks
        self._register_standard_health_checks()
        
        self.logger.info("Service observability initialized",
            service=self.service_name,
            components=["logger", "metrics", "tracer", "health", "alerts"]
        )
        
        self._initialized = True
    
    def _register_standard_health_checks(self):
        """Register standard health checks for the service."""
        
        def service_health_check() -> FlextResult[dict]:
            """Basic service health check."""
            if not self._initialized:
                return FlextResult.fail("Service not initialized")
            
            return FlextResult.ok({
                "status": "healthy",
                "service": self.service_name,
                "initialized": self._initialized,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        def memory_health_check() -> FlextResult[dict]:
            """Memory usage health check."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                
                if memory.percent > 90:
                    return FlextResult.fail(f"High memory usage: {memory.percent}%")
                
                return FlextResult.ok({
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2)
                })
            except ImportError:
                return FlextResult.ok({"status": "psutil_not_available"})
        
        # Register health checks
        self.health.register_health_check("service", service_health_check)
        self.health.register_health_check("memory", memory_health_check)
    
    def execute_with_observability(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with comprehensive observability."""
        
        with self.tracer.trace_operation(operation_name) as span:
            start_time = time.time()
            
            # Set base span tags
            span.set_tag("service", self.service_name)
            span.set_tag("operation", operation_name)
            span.set_tag("args_count", len(args))
            span.set_tag("kwargs_count", len(kwargs))
            
            try:
                # Execute operation
                result = operation_func(*args, **kwargs)
                
                # Success observability
                duration_ms = (time.time() - start_time) * 1000
                
                # Metrics
                self.metrics.record_histogram(f"{operation_name}_duration_ms", duration_ms)
                self.metrics.increment_counter(f"{operation_name}_success_total", 1)
                
                # Span tags
                span.set_tag("success", True)
                span.set_tag("result_type", type(result).__name__ if result else "None")
                
                # Structured logging
                self.logger.info(f"Operation {operation_name} completed",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Error observability
                duration_ms = (time.time() - start_time) * 1000
                error_type = type(e).__name__
                
                # Metrics
                self.metrics.increment_counter(f"{operation_name}_error_total", 1,
                    error_type=error_type
                )
                
                # Span tags
                span.set_tag("error", True)
                span.set_tag("error_type", error_type)
                span.set_tag("error_message", str(e))
                
                # Structured logging
                self.logger.exception(f"Operation {operation_name} failed",
                    operation=operation_name,
                    error_type=error_type,
                    duration_ms=duration_ms
                )
                
                # Alert on errors
                self.alerts.send_alert("ERROR",
                    f"Operation {operation_name} failed in {self.service_name}",
                    service=self.service_name,
                    operation=operation_name,
                    error_type=error_type,
                    error_message=str(e)
                )
                
                raise
    
    @abstractmethod
    def start(self) -> FlextResult[None]:
        """Start the service."""
        pass
    
    @abstractmethod
    def stop(self) -> FlextResult[None]:
        """Stop the service."""
        pass
    
    def get_health_status(self) -> FlextResult[dict]:
        """Get comprehensive service health status."""
        return self.health.check_health()
```

### Step 2: Service Implementation Example

```python
class UserService(ObservableService):
    """Example user service with comprehensive observability."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.db_connection = None
        
        # Initialize with observability
        super().__init__("user-service", {
            "environment": "production",
            "log_level": "INFO",
            "alerts": {
                "channels": {
                    "slack": {"webhook_url": os.getenv("SLACK_WEBHOOK")}
                }
            }
        })
        
        # Register service-specific health checks
        self._register_service_health_checks()
    
    def _register_service_health_checks(self):
        """Register user service specific health checks."""
        
        def database_health_check() -> FlextResult[dict]:
            """Database connectivity health check."""
            try:
                if not self.db_connection:
                    return FlextResult.fail("Database connection not established")
                
                # Test connection (placeholder)
                # self.db_connection.execute("SELECT 1")
                
                return FlextResult.ok({
                    "database": "healthy",
                    "url": self.database_url
                })
            except Exception as e:
                return FlextResult.fail(f"Database unhealthy: {e}")
        
        self.health.register_health_check("database", database_health_check)
    
    def start(self) -> FlextResult[None]:
        """Start user service with observability."""
        
        return FlextResult.from_callable(
            lambda: self.execute_with_observability(
                "service_start",
                self._start_implementation
            )
        )
    
    def _start_implementation(self):
        """Actual service start implementation."""
        # Connect to database (placeholder)
        self.db_connection = f"connected_to_{self.database_url}"
        
        self.logger.info("User service started",
            database_url=self.database_url,
            service_status="running"
        )
        
        # Service startup metric
        self.metrics.increment_counter("service_starts_total", 1)
    
    def stop(self) -> FlextResult[None]:
        """Stop user service with observability."""
        
        return FlextResult.from_callable(
            lambda: self.execute_with_observability(
                "service_stop",
                self._stop_implementation
            )
        )
    
    def _stop_implementation(self):
        """Actual service stop implementation."""
        if self.db_connection:
            # Close database connection (placeholder)
            self.db_connection = None
        
        self.logger.info("User service stopped")
    
    def create_user(self, user_data: dict) -> FlextResult[dict]:
        """Create user with comprehensive observability."""
        
        def _create_user_impl():
            # Input validation and logging
            self.logger.info("Creating user", user_email=user_data.get("email"))
            
            # Business logic (placeholder)
            user_id = f"user_{int(time.time())}"
            user = {
                "id": user_id,
                "email": user_data["email"],
                "name": user_data["name"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Success logging with business context
            self.logger.info("User created successfully",
                user_id=user_id,
                email=user_data["email"]
            )
            
            # Business metrics
            self.metrics.increment_counter("users_created_total", 1)
            
            return user
        
        return FlextResult.from_callable(
            lambda: self.execute_with_observability(
                "create_user",
                _create_user_impl
            )
        )
    
    def get_user(self, user_id: str) -> FlextResult[dict]:
        """Get user with observability."""
        
        def _get_user_impl():
            # Simulated database lookup
            if user_id.startswith("user_"):
                user = {
                    "id": user_id,
                    "email": f"{user_id}@example.com",
                    "name": f"User {user_id}"
                }
                
                self.logger.info("User retrieved", user_id=user_id)
                self.metrics.increment_counter("users_retrieved_total", 1)
                
                return user
            else:
                raise ValueError(f"User not found: {user_id}")
        
        return FlextResult.from_callable(
            lambda: self.execute_with_observability(
                "get_user",
                _get_user_impl
            )
        )
```

### Step 3: Service Usage and Testing

```python
def main():
    """Main function demonstrating comprehensive observability usage."""
    
    # Initialize service
    user_service = UserService("postgresql://localhost:5432/users")
    
    # Start service
    start_result = user_service.start()
    if not start_result.success:
        print(f"❌ Failed to start service: {start_result.error}")
        return
    
    try:
        # Test service operations
        
        # Create user
        user_data = {"email": "john@example.com", "name": "John Doe"}
        create_result = user_service.create_user(user_data)
        
        if create_result.success:
            user = create_result.value
            print(f"✅ User created: {user['id']}")
            
            # Get user
            get_result = user_service.get_user(user["id"])
            if get_result.success:
                print(f"✅ User retrieved: {get_result.value['email']}")
            else:
                print(f"❌ Failed to get user: {get_result.error}")
        else:
            print(f"❌ Failed to create user: {create_result.error}")
        
        # Check service health
        health_result = user_service.get_health_status()
        if health_result.success:
            health_data = health_result.value
            print(f"✅ Service health: {health_data['status']}")
            
            # Print health details
            for check_name, check_result in health_data.get("checks", {}).items():
                status = "✅" if check_result.get("healthy", False) else "❌"
                print(f"  {status} {check_name}: {check_result.get('status', 'unknown')}")
        else:
            print(f"❌ Health check failed: {health_result.error}")
    
    finally:
        # Stop service
        stop_result = user_service.stop()
        if stop_result.success:
            print("✅ Service stopped successfully")
        else:
            print(f"❌ Failed to stop service: {stop_result.error}")

if __name__ == "__main__":
    main()
```

---

## 🔧 Advanced Implementation Patterns

### 1. Decorator-Based Monitoring

```python
from functools import wraps
from typing import TypeVar, Callable, Any

F = TypeVar('F', bound=Callable[..., Any])

def observable_operation(operation_name: str, service_name: str = "default"):
    """Decorator for automatic operation observability."""
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create observability instance
            obs = FlextObservability()
            logger = obs.create_console_logger(service_name, "INFO")
            tracer = obs.create_tracer(service_name)
            metrics = obs.create_metrics_collector(service_name)
            
            with tracer.trace_operation(operation_name) as span:
                start_time = time.time()
                
                # Set span context
                span.set_tag("function", func.__name__)
                span.set_tag("module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Success observability
                    duration_ms = (time.time() - start_time) * 1000
                    metrics.record_histogram(f"{operation_name}_duration_ms", duration_ms)
                    metrics.increment_counter(f"{operation_name}_success_total", 1)
                    
                    logger.info(f"{operation_name} completed",
                        function=func.__name__,
                        duration_ms=duration_ms
                    )
                    
                    return result
                    
                except Exception as e:
                    # Error observability
                    metrics.increment_counter(f"{operation_name}_error_total", 1,
                        error_type=type(e).__name__
                    )
                    
                    span.set_tag("error", True)
                    logger.exception(f"{operation_name} failed",
                        function=func.__name__,
                        error_type=type(e).__name__
                    )
                    
                    raise
        
        return wrapper
    return decorator

# Usage example
@observable_operation("user_authentication", "auth-service")
def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user with automatic observability."""
    
    # Authentication logic (placeholder)
    if username and password:
        return {
            "user_id": f"user_{username}",
            "authenticated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise ValueError("Invalid credentials")

# Test the decorated function
auth_result = authenticate_user("john_doe", "secret123")
print(f"✅ Authentication result: {auth_result}")
```

### 2. Context Manager for Operations

```python
from contextlib import contextmanager

@contextmanager
def observable_context(operation_name: str, service_name: str = "default", **initial_context):
    """Context manager for comprehensive operation observability."""
    
    obs = FlextObservability()
    logger = obs.create_console_logger(service_name, "INFO")
    tracer = obs.create_tracer(service_name)
    metrics = obs.create_metrics_collector(service_name)
    
    with tracer.trace_operation(operation_name) as span:
        start_time = time.time()
        
        # Set initial context
        for key, value in initial_context.items():
            span.set_tag(key, value)
        
        # Create operation context
        context = {
            "logger": logger,
            "span": span,
            "metrics": metrics,
            "operation_name": operation_name,
            "start_time": start_time
        }
        
        try:
            logger.info(f"Starting {operation_name}", **initial_context)
            yield context
            
            # Success observability
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_histogram(f"{operation_name}_duration_ms", duration_ms)
            metrics.increment_counter(f"{operation_name}_success_total", 1)
            
            logger.info(f"{operation_name} completed successfully",
                duration_ms=duration_ms,
                **initial_context
            )
            
        except Exception as e:
            # Error observability
            duration_ms = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            
            metrics.increment_counter(f"{operation_name}_error_total", 1,
                error_type=error_type
            )
            
            span.set_tag("error", True)
            span.set_tag("error_type", error_type)
            
            logger.exception(f"{operation_name} failed",
                error_type=error_type,
                duration_ms=duration_ms,
                **initial_context
            )
            
            raise

# Usage example
def process_order(order_id: str, customer_id: str):
    """Process order with context-managed observability."""
    
    with observable_context("order_processing", "order-service",
                           order_id=order_id, customer_id=customer_id) as ctx:
        
        logger = ctx["logger"]
        span = ctx["span"]
        metrics = ctx["metrics"]
        
        # Step 1: Validate order
        logger.info("Validating order", order_id=order_id)
        span.log_event("validation_started")
        
        # Validation logic (placeholder)
        time.sleep(0.1)  # Simulate processing
        
        span.set_tag("validation_result", "success")
        span.log_event("validation_completed")
        
        # Step 2: Process payment
        logger.info("Processing payment", order_id=order_id)
        span.log_event("payment_started")
        
        # Payment processing (placeholder)
        time.sleep(0.2)  # Simulate payment processing
        
        span.set_tag("payment_result", "success")
        span.set_tag("payment_amount", 99.99)
        span.log_event("payment_completed")
        
        # Step 3: Update inventory
        logger.info("Updating inventory", order_id=order_id)
        span.log_event("inventory_update_started")
        
        # Inventory update (placeholder)
        time.sleep(0.05)  # Simulate inventory update
        
        span.log_event("inventory_update_completed")
        
        # Business metrics
        metrics.increment_counter("orders_processed_total", 1,
            customer_tier="premium"
        )
        metrics.set_gauge("inventory_updates_pending", 0)
        
        logger.info("Order processing completed successfully",
            order_id=order_id,
            customer_id=customer_id
        )

# Test the context-managed operation
process_order("order_123", "customer_456")
print("✅ Order processed with comprehensive observability")
```

### 3. Middleware Pattern for Web Applications

```python
from typing import Protocol

class HTTPRequest(Protocol):
    method: str
    path: str
    headers: dict
    query_params: dict

class HTTPResponse(Protocol):  
    status_code: int
    headers: dict
    body: str

class ObservabilityMiddleware:
    """HTTP middleware for automatic request observability."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.obs = FlextObservability()
        self.logger = self.obs.create_console_logger(service_name, "INFO")
        self.tracer = self.obs.create_tracer(service_name)
        self.metrics = self.obs.create_metrics_collector(service_name)
    
    def process_request(self, request: HTTPRequest, next_handler: Callable) -> HTTPResponse:
        """Process HTTP request with comprehensive observability."""
        
        operation_name = f"{request.method} {request.path}"
        
        with self.tracer.trace_operation(operation_name) as span:
            start_time = time.time()
            
            # Set request context
            span.set_tag("http.method", request.method)
            span.set_tag("http.url", request.path)
            span.set_tag("http.user_agent", request.headers.get("User-Agent", "unknown"))
            
            # Request metrics
            self.metrics.increment_counter("http_requests_total", 1,
                method=request.method,
                endpoint=request.path
            )
            
            try:
                # Process request
                response = next_handler(request)
                
                # Success observability
                duration_ms = (time.time() - start_time) * 1000
                
                # Response metrics
                self.metrics.record_histogram("http_request_duration_ms", duration_ms,
                    method=request.method,
                    status_code=str(response.status_code)
                )
                
                # Span tags
                span.set_tag("http.status_code", response.status_code)
                span.set_tag("response_size", len(response.body))
                
                # Structured logging
                self.logger.info("HTTP request processed",
                    method=request.method,
                    path=request.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    response_size=len(response.body)
                )
                
                return response
                
            except Exception as e:
                # Error observability
                duration_ms = (time.time() - start_time) * 1000
                error_type = type(e).__name__
                
                # Error metrics
                self.metrics.increment_counter("http_requests_error_total", 1,
                    method=request.method,
                    error_type=error_type
                )
                
                # Span error tags
                span.set_tag("error", True)
                span.set_tag("error_type", error_type)
                
                # Error logging
                self.logger.exception("HTTP request failed",
                    method=request.method,
                    path=request.path,
                    error_type=error_type,
                    duration_ms=duration_ms
                )
                
                raise

# Usage example with mock HTTP framework
class MockRequest:
    def __init__(self, method: str, path: str, headers: dict = None):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.query_params = {}

class MockResponse:
    def __init__(self, status_code: int, body: str = ""):
        self.status_code = status_code
        self.headers = {}
        self.body = body

def sample_handler(request: MockRequest) -> MockResponse:
    """Sample HTTP handler."""
    if request.path == "/users":
        return MockResponse(200, '{"users": []}')
    elif request.path == "/error":
        raise ValueError("Simulated error")
    else:
        return MockResponse(404, "Not Found")

# Test middleware
middleware = ObservabilityMiddleware("web-service")

# Successful request
request1 = MockRequest("GET", "/users", {"User-Agent": "test-client/1.0"})
response1 = middleware.process_request(request1, sample_handler)
print(f"✅ Request processed: {response1.status_code}")

# Error request
try:
    request2 = MockRequest("POST", "/error")
    response2 = middleware.process_request(request2, sample_handler)
except ValueError:
    print("✅ Error request handled with observability")
```

---

## 🎯 Best Practices and Recommendations

### 1. Configuration Management

```python
import os
from typing import Dict, Any

def get_observability_config(service_name: str) -> Dict[str, Any]:
    """Get environment-specific observability configuration."""
    
    env = os.getenv("ENVIRONMENT", "development")
    
    base_config = {
        "service_name": service_name,
        "environment": env,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "metrics_enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true",
        "tracing_enabled": os.getenv("TRACING_ENABLED", "true").lower() == "true",
        "health_enabled": os.getenv("HEALTH_ENABLED", "true").lower() == "true",
    }
    
    # Environment-specific configurations
    if env == "production":
        base_config.update({
            "log_level": "WARN",
            "tracing_sample_rate": 0.01,  # 1% sampling in prod
            "metrics_flush_interval": 30,
            "alerts": {
                "enabled": True,
                "channels": {
                    "slack": {"webhook_url": os.getenv("SLACK_WEBHOOK")},
                    "pagerduty": {"api_key": os.getenv("PAGERDUTY_API_KEY")}
                }
            }
        })
    elif env == "staging":
        base_config.update({
            "log_level": "INFO",
            "tracing_sample_rate": 0.1,  # 10% sampling in staging
            "alerts": {
                "enabled": True,
                "channels": {
                    "slack": {"webhook_url": os.getenv("SLACK_WEBHOOK")}
                }
            }
        })
    else:  # development
        base_config.update({
            "log_level": "DEBUG",
            "tracing_sample_rate": 1.0,  # 100% sampling in dev
            "alerts": {"enabled": False}
        })
    
    return base_config
```

### 2. Performance Optimization

```python
class OptimizedObservabilityService(ObservableService):
    """Performance-optimized observability service."""
    
    def __init__(self, service_name: str):
        super().__init__(service_name, get_observability_config(service_name))
        
        # Performance optimizations
        self._metrics_buffer = []
        self._buffer_size = 100
        self._last_flush = time.time()
        self._flush_interval = 30  # seconds
    
    def record_metric_optimized(self, name: str, value: float, **tags):
        """Record metric with buffering for performance."""
        
        metric_data = {
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": time.time()
        }
        
        self._metrics_buffer.append(metric_data)
        
        # Flush buffer if needed
        if (len(self._metrics_buffer) >= self._buffer_size or 
            time.time() - self._last_flush >= self._flush_interval):
            self._flush_metrics_buffer()
    
    def _flush_metrics_buffer(self):
        """Flush buffered metrics to observability system."""
        
        if not self._metrics_buffer:
            return
        
        try:
            # Process buffered metrics
            for metric_data in self._metrics_buffer:
                self.metrics.record_histogram(
                    metric_data["name"],
                    metric_data["value"],
                    **metric_data["tags"]
                )
            
            self.logger.debug("Metrics buffer flushed",
                metrics_count=len(self._metrics_buffer)
            )
            
        except Exception as e:
            self.logger.exception("Failed to flush metrics buffer", error=str(e))
        
        finally:
            self._metrics_buffer.clear()
            self._last_flush = time.time()
```

### 3. Error Handling and Resilience

```python
class ResilientObservabilityService(ObservableService):
    """Observability service with resilience patterns."""
    
    def __init__(self, service_name: str):
        super().__init__(service_name, get_observability_config(service_name))
        self._observability_healthy = True
        self._last_health_check = time.time()
        self._health_check_interval = 60  # seconds
    
    def execute_with_resilient_observability(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with resilient observability."""
        
        # Check observability health
        self._check_observability_health()
        
        if self._observability_healthy:
            try:
                return self.execute_with_observability(
                    operation_name, operation_func, *args, **kwargs
                )
            except Exception as obs_error:
                # If observability fails, don't let it crash the operation
                self.logger.exception("Observability system error",
                    observability_error=str(obs_error)
                )
                self._observability_healthy = False
                
                # Execute operation without observability
                return operation_func(*args, **kwargs)
        else:
            # Execute operation without observability
            return operation_func(*args, **kwargs)
    
    def _check_observability_health(self):
        """Check if observability system is healthy."""
        
        if time.time() - self._last_health_check < self._health_check_interval:
            return
        
        try:
            # Test basic observability functions
            test_logger = self.obs.create_console_logger("health_check", "INFO")
            test_metrics = self.obs.create_metrics_collector("health_check")
            
            test_logger.info("Observability health check")
            test_metrics.increment_counter("health_checks", 1)
            
            self._observability_healthy = True
            
        except Exception as e:
            self.logger.exception("Observability health check failed", error=str(e))
            self._observability_healthy = False
        
        finally:
            self._last_health_check = time.time()
```

---

This comprehensive implementation guide demonstrates how to effectively integrate FlextObservability across different service patterns, from basic monitoring to enterprise-grade observability orchestration with performance optimization and resilience considerations.
