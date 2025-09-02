# FlextExceptions Implementation Guide

**Step-by-step guide for implementing FlextExceptions as the comprehensive error handling foundation in FLEXT ecosystem projects.**

---

## Overview

This guide provides comprehensive instructions for integrating FlextExceptions into new and existing FLEXT projects. FlextExceptions serves as the enterprise exception system, providing structured error handling, automatic metrics collection, distributed tracing support, and comprehensive observability across all FLEXT components.

---

## Quick Start Implementation

### Step 1: Basic Exception Usage

```python
from flext_core import FlextExceptions

# Simple exception raising with automatic type selection
try:
    user_data = {"name": "", "email": "invalid-email"}
    
    # Automatically creates ValidationError
    if not user_data["name"]:
        raise FlextExceptions(
            "Name is required",
            field="name",
            value=user_data["name"],
            context={"form_step": "personal_info"}
        )
    
    # Automatically creates ValidationError with email context  
    if "@" not in user_data["email"]:
        raise FlextExceptions(
            "Invalid email format",
            field="email",
            value=user_data["email"],
            validation_details={"expected_format": "email"},
            context={"form_step": "contact_info"}
        )
        
except FlextExceptions.BaseError as e:
    print(f"Error Code: {e.error_code}")
    print(f"Message: {e.message}")
    print(f"Correlation ID: {e.correlation_id}")
    print(f"Context: {e.context}")
```

### Step 2: Specialized Exception Types

```python
# Use specific exception types for different error categories
def demonstrate_specialized_exceptions():
    
    # Configuration errors
    try:
        raise FlextExceptions.ConfigurationError(
            "Database URL not configured",
            config_key="DATABASE_URL",
            config_file="/app/.env",
            context={"environment": "production"}
        )
    except FlextExceptions.ConfigurationError as e:
        print(f"Config Error: {e.config_key} in {e.config_file}")
    
    # Network connection errors
    try:
        raise FlextExceptions.ConnectionError(
            "Failed to connect to external API",
            service="user_service",
            endpoint="https://api.example.com/users",
            context={"timeout": 30, "retry_count": 3}
        )
    except FlextExceptions.ConnectionError as e:
        print(f"Connection Error: {e.service} at {e.endpoint}")
    
    # Authentication errors
    try:
        raise FlextExceptions.AuthenticationError(
            "Invalid JWT token",
            auth_method="jwt_bearer",
            context={"token_expired": True, "user_id": "12345"}
        )
    except FlextExceptions.AuthenticationError as e:
        print(f"Auth Error: {e.auth_method}")
```

### Step 3: Metrics and Monitoring Setup

```python
# Enable automatic metrics collection and monitoring
def setup_exception_monitoring():
    
    # Generate some exceptions for demonstration
    for i in range(5):
        try:
            raise FlextExceptions.ValidationError(f"Test validation error {i}")
        except FlextExceptions.ValidationError:
            pass
    
    for i in range(3):
        try:
            raise FlextExceptions.ConnectionError(f"Test connection error {i}")
        except FlextExceptions.ConnectionError:
            pass
    
    # Get metrics
    metrics = FlextExceptions.get_metrics()
    print("Exception Metrics:")
    for exception_type, count in metrics.items():
        print(f"  {exception_type}: {count}")
    
    # Clear metrics for next monitoring period
    FlextExceptions.clear_metrics()
    print("Metrics cleared for next period")
```

---

## Detailed Implementation Patterns

### 1. Application-Level Exception Handling

#### Centralized Exception Handler

```python
from flext_core import FlextExceptions
import logging

class ApplicationExceptionHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure structured logging for exceptions."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def handle_exception(self, exception: FlextExceptions.BaseError) -> dict:
        """Central exception handling with structured logging and metrics."""
        
        # Log exception with full context
        self.logger.error(
            f"Exception occurred: {exception}",
            extra={
                "error_code": exception.error_code,
                "correlation_id": exception.correlation_id,
                "context": exception.context,
                "exception_type": type(exception).__name__,
                "timestamp": exception.timestamp
            }
        )
        
        # Determine response based on exception type
        if isinstance(exception, FlextExceptions.ValidationError):
            return self._handle_validation_error(exception)
        elif isinstance(exception, FlextExceptions.ConfigurationError):
            return self._handle_configuration_error(exception)
        elif isinstance(exception, FlextExceptions.ConnectionError):
            return self._handle_connection_error(exception)
        elif isinstance(exception, FlextExceptions.AuthenticationError):
            return self._handle_authentication_error(exception)
        elif isinstance(exception, FlextExceptions.CriticalError):
            return self._handle_critical_error(exception)
        else:
            return self._handle_generic_error(exception)
    
    def _handle_validation_error(self, exception: FlextExceptions.ValidationError) -> dict:
        """Handle validation errors with field-specific messaging."""
        return {
            "error_type": "validation_error",
            "message": f"Validation failed: {exception.message}",
            "field": getattr(exception, 'field', None),
            "error_code": exception.error_code,
            "correlation_id": exception.correlation_id,
            "recoverable": True
        }
    
    def _handle_configuration_error(self, exception: FlextExceptions.ConfigurationError) -> dict:
        """Handle configuration errors with system administrator guidance."""
        return {
            "error_type": "configuration_error", 
            "message": "System configuration error - please contact administrator",
            "config_key": getattr(exception, 'config_key', None),
            "error_code": exception.error_code,
            "correlation_id": exception.correlation_id,
            "recoverable": False
        }
    
    def _handle_connection_error(self, exception: FlextExceptions.ConnectionError) -> dict:
        """Handle connection errors with retry suggestions."""
        return {
            "error_type": "connection_error",
            "message": "Service temporarily unavailable - please try again",
            "service": getattr(exception, 'service', None),
            "error_code": exception.error_code,
            "correlation_id": exception.correlation_id,
            "recoverable": True,
            "retry_after": 30
        }
    
    def _handle_authentication_error(self, exception: FlextExceptions.AuthenticationError) -> dict:
        """Handle authentication errors with security considerations."""
        return {
            "error_type": "authentication_error",
            "message": "Authentication required",
            "auth_method": getattr(exception, 'auth_method', None),
            "error_code": exception.error_code,
            "correlation_id": exception.correlation_id,
            "recoverable": True,
            "action_required": "login"
        }
    
    def _handle_critical_error(self, exception: FlextExceptions.CriticalError) -> dict:
        """Handle critical errors with immediate escalation."""
        # Send immediate alert
        self._send_critical_alert(exception)
        
        return {
            "error_type": "critical_error",
            "message": "Critical system error - incident reported",
            "error_code": exception.error_code,
            "correlation_id": exception.correlation_id,
            "recoverable": False,
            "incident_id": self._create_incident(exception)
        }
    
    def _send_critical_alert(self, exception: FlextExceptions.CriticalError):
        """Send immediate alert for critical errors."""
        # Implementation would integrate with alerting system
        print(f"CRITICAL ALERT: {exception.message}")
        print(f"Correlation ID: {exception.correlation_id}")
        print(f"Context: {exception.context}")
    
    def _create_incident(self, exception: FlextExceptions.CriticalError) -> str:
        """Create incident ticket for critical errors."""
        # Implementation would integrate with incident management system
        incident_id = f"INC-{exception.correlation_id}"
        print(f"Created incident: {incident_id}")
        return incident_id
```

#### Decorator-Based Exception Handling

```python
from functools import wraps

def flext_exception_handler(default_response=None):
    """Decorator for automatic FlextException handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ApplicationExceptionHandler()
            
            try:
                return func(*args, **kwargs)
            except FlextExceptions.BaseError as e:
                return handler.handle_exception(e)
            except Exception as e:
                # Convert non-FLEXT exceptions to FlextExceptions
                flext_exception = FlextExceptions.Error(
                    f"Unexpected error: {str(e)}",
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "original_exception": type(e).__name__
                    }
                )
                return handler.handle_exception(flext_exception)
        
        return wrapper
    return decorator

# Usage example
@flext_exception_handler()
def risky_user_operation(user_id: str, operation_data: dict):
    """Example function with automatic exception handling."""
    
    # Validation
    if not user_id:
        raise FlextExceptions.ValidationError(
            "User ID is required",
            field="user_id",
            value=user_id
        )
    
    # Business logic that might fail
    if not operation_data.get("email"):
        raise FlextExceptions.ValidationError(
            "Email is required for this operation", 
            field="email",
            value=operation_data.get("email"),
            context={"operation": "user_update", "user_id": user_id}
        )
    
    return {"status": "success", "user_id": user_id}
```

### 2. Service Layer Exception Patterns

#### Service Exception Wrapper

```python
class ServiceExceptionWrapper:
    """Wrapper for consistent service-level exception handling."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"service.{service_name}")
    
    def wrap_operation(self, operation_name: str):
        """Decorator for wrapping service operations with exception handling."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                correlation_id = kwargs.pop('correlation_id', None) or f"svc_{int(time.time() * 1000)}"
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful operation
                    self.logger.info(
                        f"Operation {operation_name} completed successfully",
                        extra={
                            "service": self.service_name,
                            "operation": operation_name,
                            "correlation_id": correlation_id
                        }
                    )
                    
                    return result
                
                except FlextExceptions.BaseError as e:
                    # Re-raise with service context added
                    e.context.update({
                        "service": self.service_name,
                        "operation": operation_name
                    })
                    raise
                
                except Exception as e:
                    # Convert to FlextException with service context
                    raise FlextExceptions.ProcessingError(
                        f"Service operation failed: {str(e)}",
                        operation=operation_name,
                        context={
                            "service": self.service_name,
                            "original_exception": type(e).__name__,
                            "function": func.__name__
                        },
                        correlation_id=correlation_id
                    )
            
            return wrapper
        return decorator

# Usage in service classes
class UserService:
    def __init__(self):
        self.wrapper = ServiceExceptionWrapper("user_service")
    
    @ServiceExceptionWrapper("user_service").wrap_operation("create_user")
    def create_user(self, user_data: dict) -> dict:
        """Create new user with automatic exception wrapping."""
        
        # Input validation
        if not user_data.get("email"):
            raise FlextExceptions.ValidationError(
                "Email is required for user creation",
                field="email",
                value=user_data.get("email"),
                context={"operation": "create_user"}
            )
        
        if not user_data.get("name"):
            raise FlextExceptions.ValidationError(
                "Name is required for user creation",
                field="name", 
                value=user_data.get("name"),
                context={"operation": "create_user"}
            )
        
        # Simulate database operation that might fail
        if user_data["email"] == "duplicate@example.com":
            raise FlextExceptions.AlreadyExistsError(
                "User with this email already exists",
                resource_id=user_data["email"],
                resource_type="User",
                context={"operation": "create_user"}
            )
        
        return {
            "user_id": f"user_{int(time.time())}",
            "email": user_data["email"],
            "name": user_data["name"],
            "status": "created"
        }
    
    @ServiceExceptionWrapper("user_service").wrap_operation("get_user")
    def get_user(self, user_id: str) -> dict:
        """Retrieve user with automatic exception wrapping."""
        
        if not user_id:
            raise FlextExceptions.ValidationError(
                "User ID is required",
                field="user_id",
                value=user_id
            )
        
        # Simulate user lookup
        if user_id == "missing":
            raise FlextExceptions.NotFoundError(
                "User not found",
                resource_id=user_id,
                resource_type="User"
            )
        
        return {
            "user_id": user_id,
            "email": f"user{user_id}@example.com",
            "name": f"User {user_id}",
            "status": "active"
        }
```

### 3. Database Integration Patterns

#### Database Exception Translation

```python
import sqlite3
import psycopg2
from typing import object, Callable, TypeVar

T = TypeVar('T')

class DatabaseExceptionTranslator:
    """Translate database-specific exceptions to FlextExceptions."""
    
    def __init__(self, database_type: str = "postgresql"):
        self.database_type = database_type
    
    def translate_and_execute(self, func: Callable[[], T]) -> T:
        """Execute database function and translate exceptions."""
        try:
            return func()
        except sqlite3.IntegrityError as e:
            raise self._handle_integrity_error(str(e))
        except sqlite3.OperationalError as e:
            raise self._handle_operational_error(str(e))
        except psycopg2.IntegrityError as e:
            raise self._handle_integrity_error(str(e))
        except psycopg2.OperationalError as e:
            raise self._handle_operational_error(str(e))
        except psycopg2.DatabaseError as e:
            raise self._handle_database_error(str(e))
        except Exception as e:
            raise FlextExceptions.ProcessingError(
                f"Database operation failed: {str(e)}",
                operation="database_operation",
                context={
                    "database_type": self.database_type,
                    "original_exception": type(e).__name__
                }
            )
    
    def _handle_integrity_error(self, error_msg: str) -> FlextExceptions.BaseError:
        """Handle database integrity constraint violations."""
        if "unique" in error_msg.lower():
            return FlextExceptions.AlreadyExistsError(
                "Record with this key already exists",
                resource_type="database_record",
                context={"database_error": error_msg}
            )
        elif "foreign key" in error_msg.lower():
            return FlextExceptions.ValidationError(
                "Referenced record does not exist",
                field="foreign_key",
                validation_details={"constraint_type": "foreign_key"},
                context={"database_error": error_msg}
            )
        else:
            return FlextExceptions.ValidationError(
                "Data integrity constraint violation",
                validation_details={"constraint_type": "integrity"},
                context={"database_error": error_msg}
            )
    
    def _handle_operational_error(self, error_msg: str) -> FlextExceptions.BaseError:
        """Handle database operational errors."""
        if "connection" in error_msg.lower():
            return FlextExceptions.ConnectionError(
                "Database connection failed",
                service="database",
                context={"database_error": error_msg}
            )
        elif "timeout" in error_msg.lower():
            return FlextExceptions.TimeoutError(
                "Database operation timed out",
                context={"database_error": error_msg}
            )
        else:
            return FlextExceptions.ProcessingError(
                "Database operational error",
                operation="database_operation", 
                context={"database_error": error_msg}
            )
    
    def _handle_database_error(self, error_msg: str) -> FlextExceptions.BaseError:
        """Handle general database errors."""
        return FlextExceptions.ProcessingError(
            "Database error occurred",
            operation="database_operation",
            context={"database_error": error_msg}
        )

# Usage with database operations
class DatabaseService:
    def __init__(self, connection):
        self.connection = connection
        self.translator = DatabaseExceptionTranslator()
    
    def create_user(self, user_data: dict) -> dict:
        """Create user with automatic exception translation."""
        def _create_user():
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s)",
                (user_data["email"], user_data["name"])
            )
            user_id = cursor.lastrowid
            self.connection.commit()
            return {"user_id": user_id, **user_data}
        
        return self.translator.translate_and_execute(_create_user)
    
    def get_user(self, user_id: int) -> dict:
        """Get user with automatic exception translation."""
        def _get_user():
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            
            if not result:
                raise FlextExceptions.NotFoundError(
                    "User not found",
                    resource_id=str(user_id),
                    resource_type="User"
                )
            
            return {
                "user_id": result[0],
                "email": result[1],
                "name": result[2]
            }
        
        return self.translator.translate_and_execute(_get_user)
```

### 4. API Integration Patterns

#### HTTP Client Exception Handling

```python
import requests
from typing import Dict, object

class HTTPClientWithFlextExceptions:
    """HTTP client with automatic FlextException translation."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def get(self, endpoint: str, **kwargs) -> Dict[str, object]:
        """GET request with FlextException handling."""
        return self._make_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> Dict[str, object]:
        """POST request with FlextException handling."""
        return self._make_request('POST', endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> Dict[str, object]:
        """PUT request with FlextException handling."""
        return self._make_request('PUT', endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Dict[str, object]:
        """DELETE request with FlextException handling."""
        return self._make_request('DELETE', endpoint, **kwargs)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, object]:
        """Make HTTP request with comprehensive exception handling."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        correlation_id = kwargs.pop('correlation_id', f"http_{int(time.time() * 1000)}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            
            # Handle HTTP error status codes
            if response.status_code >= 400:
                return self._handle_http_error(response, method, url, correlation_id)
            
            return response.json()
        
        except requests.exceptions.Timeout:
            raise FlextExceptions.TimeoutError(
                f"HTTP request timed out after {self.timeout} seconds",
                timeout_seconds=float(self.timeout),
                context={
                    "method": method,
                    "url": url,
                    "correlation_id": correlation_id
                }
            )
        
        except requests.exceptions.ConnectionError as e:
            raise FlextExceptions.ConnectionError(
                f"Failed to connect to {url}",
                service="http_api",
                endpoint=url,
                context={
                    "method": method,
                    "correlation_id": correlation_id,
                    "original_error": str(e)
                }
            )
        
        except requests.exceptions.RequestException as e:
            raise FlextExceptions.ProcessingError(
                f"HTTP request failed: {str(e)}",
                operation="http_request",
                context={
                    "method": method,
                    "url": url,
                    "correlation_id": correlation_id,
                    "original_error": str(e)
                }
            )
        
        except ValueError as e:
            # JSON decode error
            raise FlextExceptions.ProcessingError(
                "Invalid JSON response from API",
                operation="json_decode", 
                context={
                    "method": method,
                    "url": url,
                    "correlation_id": correlation_id,
                    "original_error": str(e)
                }
            )
    
    def _handle_http_error(self, response: requests.Response, method: str, url: str, correlation_id: str):
        """Handle HTTP error status codes."""
        status_code = response.status_code
        
        try:
            error_body = response.json()
        except ValueError:
            error_body = {"message": response.text}
        
        context = {
            "method": method,
            "url": url,
            "status_code": status_code,
            "response_body": error_body,
            "correlation_id": correlation_id
        }
        
        if status_code == 400:
            raise FlextExceptions.ValidationError(
                f"Bad request: {error_body.get('message', 'Invalid request')}",
                field="request_data",
                validation_details=error_body,
                context=context
            )
        
        elif status_code == 401:
            raise FlextExceptions.AuthenticationError(
                "Authentication failed",
                auth_method="api_key",
                context=context
            )
        
        elif status_code == 403:
            raise FlextExceptions.PermissionError(
                "Access denied",
                required_permission="api_access",
                context=context
            )
        
        elif status_code == 404:
            raise FlextExceptions.NotFoundError(
                "Resource not found",
                resource_type="api_resource",
                context=context
            )
        
        elif status_code == 409:
            raise FlextExceptions.AlreadyExistsError(
                "Resource already exists",
                resource_type="api_resource",
                context=context
            )
        
        elif status_code >= 500:
            raise FlextExceptions.ProcessingError(
                f"Server error: {error_body.get('message', 'Internal server error')}",
                operation="api_request",
                context=context
            )
        
        else:
            raise FlextExceptions.ProcessingError(
                f"HTTP error {status_code}: {error_body.get('message', 'Unknown error')}",
                operation="api_request",
                context=context
            )

# Usage example
class UserAPIClient:
    def __init__(self):
        self.client = HTTPClientWithFlextExceptions("https://api.example.com")
    
    def create_user(self, user_data: dict) -> dict:
        """Create user via API with automatic exception handling."""
        return self.client.post(
            "/users",
            json=user_data,
            correlation_id=f"create_user_{int(time.time())}"
        )
    
    def get_user(self, user_id: str) -> dict:
        """Get user via API with automatic exception handling."""
        return self.client.get(
            f"/users/{user_id}",
            correlation_id=f"get_user_{user_id}"
        )
```

### 5. Configuration-Based Exception Handling

#### Environment-Specific Exception Configuration

```python
import os

class FlextExceptionConfigManager:
    """Manage FlextException configuration based on environment."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self._configure_for_environment()
    
    def _configure_for_environment(self):
        """Configure exception handling based on environment."""
        
        if self.environment == "production":
            config = {
                "environment": "production",
                "log_level": "ERROR",
                "validation_level": "STRICT",
                "enable_metrics": True,
                "enable_stack_traces": False,
                "max_error_details": 500,
                "error_correlation_enabled": True
            }
        
        elif self.environment == "staging":
            config = {
                "environment": "staging",
                "log_level": "INFO",
                "validation_level": "NORMAL",
                "enable_metrics": True,
                "enable_stack_traces": True,
                "max_error_details": 1000,
                "error_correlation_enabled": True
            }
        
        elif self.environment == "development":
            config = {
                "environment": "development",
                "log_level": "DEBUG",
                "validation_level": "LOOSE",
                "enable_metrics": True,
                "enable_stack_traces": True,
                "max_error_details": 2000,
                "error_correlation_enabled": True
            }
        
        else:  # test environment
            config = {
                "environment": "test",
                "log_level": "WARNING",
                "validation_level": "NORMAL",
                "enable_metrics": False,
                "enable_stack_traces": False,
                "max_error_details": 1000,
                "error_correlation_enabled": False
            }
        
        # Apply configuration
        result = FlextExceptions.configure_error_handling(config)
        if result.failure:
            print(f"Failed to configure exception handling: {result.error}")
        else:
            print(f"Exception handling configured for {self.environment} environment")
    
    def get_current_config(self) -> dict:
        """Get current exception handling configuration."""
        result = FlextExceptions.get_error_handling_config()
        if result.success:
            return result.value
        else:
            return {"error": result.error}
    
    def create_environment_config(self, target_env: str) -> dict:
        """Create configuration for specific environment."""
        result = FlextExceptions.create_environment_specific_config(target_env)
        if result.success:
            return result.value
        else:
            return {"error": result.error}

# Usage in application startup
def initialize_application():
    """Initialize application with environment-specific exception handling."""
    
    # Configure exception handling
    config_manager = FlextExceptionConfigManager()
    current_config = config_manager.get_current_config()
    
    print("Exception Configuration:")
    for key, value in current_config.items():
        print(f"  {key}: {value}")
    
    # Setup exception monitoring
    setup_exception_monitoring()
    
    print("Application initialized with FlextException handling")

def setup_exception_monitoring():
    """Setup periodic exception monitoring."""
    
    def monitor_exceptions():
        """Monitor exception patterns and send alerts."""
        metrics = FlextExceptions.get_metrics()
        
        # Check for critical errors
        if metrics.get("CriticalError", 0) > 0:
            send_alert("Critical errors detected", severity="high")
        
        # Check for high validation error rates
        if metrics.get("ValidationError", 0) > 50:
            send_alert("High validation error rate", severity="medium")
        
        # Check for connection issues
        if metrics.get("ConnectionError", 0) > 10:
            send_alert("Connection issues detected", severity="medium")
        
        # Log metrics summary
        print(f"Exception metrics: {metrics}")
        
        # Reset metrics for next period
        FlextExceptions.clear_metrics()
    
    # In a real application, this would be scheduled periodically
    monitor_exceptions()

def send_alert(message: str, severity: str = "medium"):
    """Send alert to monitoring system."""
    print(f"ALERT [{severity.upper()}]: {message}")
```

---

## Advanced Implementation Patterns

### 1. Exception Enrichment and Context Chaining

```python
class ExceptionEnricher:
    """Enrich exceptions with additional context as they propagate."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.context_stack = []
    
    def add_context(self, **context):
        """Add context to be included in any exceptions raised."""
        self.context_stack.append(context)
    
    def enrich_exception(self, exception: FlextExceptions.BaseError) -> FlextExceptions.BaseError:
        """Enrich exception with accumulated context."""
        
        # Add service context
        exception.context.update({
            "service": self.service_name,
            "context_stack": self.context_stack.copy()
        })
        
        return exception
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, FlextExceptions.BaseError):
            # Enrich the exception with accumulated context
            self.enrich_exception(exc_val)
        
        # Don't suppress the exception
        return False

# Usage with context enrichment
def complex_business_operation(user_id: str, order_data: dict):
    """Complex operation with context enrichment."""
    
    with ExceptionEnricher("order_service") as enricher:
        enricher.add_context(user_id=user_id, operation="process_order")
        
        # Validate user
        enricher.add_context(step="user_validation")
        user = validate_user(user_id)
        
        # Validate order
        enricher.add_context(step="order_validation", user_role=user.get("role"))
        validate_order_data(order_data)
        
        # Process payment
        enricher.add_context(step="payment_processing", amount=order_data.get("total"))
        payment_result = process_payment(order_data["payment"])
        
        # Create order
        enricher.add_context(step="order_creation", payment_id=payment_result["id"])
        order = create_order(user_id, order_data, payment_result)
        
        return order

def validate_user(user_id: str) -> dict:
    """Validate user existence and status."""
    if not user_id:
        raise FlextExceptions.ValidationError(
            "User ID is required",
            field="user_id",
            value=user_id
        )
    
    # Simulate user lookup that might fail
    if user_id == "blocked":
        raise FlextExceptions.PermissionError(
            "User account is blocked",
            required_permission="active_account",
            context={"user_id": user_id}
        )
    
    return {"id": user_id, "role": "customer", "status": "active"}
```

### 2. Exception Recovery and Retry Patterns

```python
from functools import wraps
import time
import random

class ExceptionRecoveryManager:
    """Manage exception recovery and retry logic."""
    
    def __init__(self):
        self.recovery_strategies = {
            FlextExceptions.ConnectionError: self._retry_with_backoff,
            FlextExceptions.TimeoutError: self._retry_with_backoff,
            FlextExceptions.ProcessingError: self._retry_once,
            FlextExceptions.ValidationError: self._no_retry,
            FlextExceptions.AuthenticationError: self._no_retry,
            FlextExceptions.CriticalError: self._no_retry
        }
    
    def with_recovery(self, max_attempts: int = 3, backoff_multiplier: float = 2.0):
        """Decorator for automatic exception recovery."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    
                    except FlextExceptions.BaseError as e:
                        last_exception = e
                        
                        # Get recovery strategy for this exception type
                        strategy = self._get_recovery_strategy(e)
                        
                        if not strategy(e, attempt, max_attempts):
                            # Strategy says don't retry
                            break
                        
                        # Wait before retry with exponential backoff
                        if attempt < max_attempts - 1:
                            wait_time = (backoff_multiplier ** attempt) + random.uniform(0, 1)
                            time.sleep(wait_time)
                
                # All retry attempts failed
                if last_exception:
                    # Enrich with retry information
                    last_exception.context.update({
                        "retry_attempts": max_attempts,
                        "final_attempt": True,
                        "recovery_attempted": True
                    })
                    raise last_exception
                
            return wrapper
        return decorator
    
    def _get_recovery_strategy(self, exception: FlextExceptions.BaseError):
        """Get recovery strategy for exception type."""
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(exception, exc_type):
                return strategy
        
        # Default to no retry for unknown exception types
        return self._no_retry
    
    def _retry_with_backoff(self, exception: FlextExceptions.BaseError, 
                           attempt: int, max_attempts: int) -> bool:
        """Retry with exponential backoff for transient errors."""
        return attempt < max_attempts - 1
    
    def _retry_once(self, exception: FlextExceptions.BaseError,
                   attempt: int, max_attempts: int) -> bool:
        """Retry only once for processing errors."""
        return attempt == 0
    
    def _no_retry(self, exception: FlextExceptions.BaseError,
                 attempt: int, max_attempts: int) -> bool:
        """No retry for validation and authentication errors."""
        return False

# Usage with automatic recovery
recovery_manager = ExceptionRecoveryManager()

class ReliableService:
    """Service with automatic exception recovery."""
    
    @recovery_manager.with_recovery(max_attempts=3, backoff_multiplier=2.0)
    def call_external_api(self, endpoint: str) -> dict:
        """Call external API with automatic retry on transient failures."""
        
        # Simulate API call that might fail
        import random
        
        if random.random() < 0.7:  # 70% chance of failure
            if random.random() < 0.5:
                raise FlextExceptions.ConnectionError(
                    "Failed to connect to API",
                    service="external_api",
                    endpoint=endpoint
                )
            else:
                raise FlextExceptions.TimeoutError(
                    "API call timed out",
                    timeout_seconds=30.0,
                    context={"endpoint": endpoint}
                )
        
        return {"status": "success", "data": "api_response_data"}
    
    @recovery_manager.with_recovery(max_attempts=2)
    def process_data(self, data: dict) -> dict:
        """Process data with limited retry on processing errors."""
        
        # Simulate processing that might fail
        if not data:
            raise FlextExceptions.ValidationError(
                "Data cannot be empty",
                field="data",
                value=data
            )
        
        if random.random() < 0.3:  # 30% chance of processing error
            raise FlextExceptions.ProcessingError(
                "Data processing failed",
                operation="data_processing",
                context={"data_keys": list(data.keys()) if data else []}
            )
        
        return {"status": "processed", "result": "processed_data"}
```

### 3. Exception Aggregation and Batch Processing

```python
from typing import List, Tuple, Union

class ExceptionAggregator:
    """Aggregate multiple exceptions for batch processing scenarios."""
    
    def __init__(self):
        self.exceptions: List[Tuple[str, FlextExceptions.BaseError]] = []
    
    def add_exception(self, identifier: str, exception: FlextExceptions.BaseError):
        """Add exception with identifier for later processing."""
        self.exceptions.append((identifier, exception))
    
    def has_exceptions(self) -> bool:
        """Check if any exceptions were collected."""
        return len(self.exceptions) > 0
    
    def get_exception_summary(self) -> dict:
        """Get summary of collected exceptions."""
        summary = {}
        for identifier, exception in self.exceptions:
            exc_type = type(exception).__name__
            if exc_type not in summary:
                summary[exc_type] = []
            summary[exc_type].append({
                "identifier": identifier,
                "message": exception.message,
                "error_code": exception.error_code,
                "correlation_id": exception.correlation_id
            })
        
        return summary
    
    def create_aggregated_exception(self) -> FlextExceptions.ProcessingError:
        """Create single exception representing all collected exceptions."""
        if not self.exceptions:
            raise ValueError("No exceptions to aggregate")
        
        summary = self.get_exception_summary()
        total_errors = len(self.exceptions)
        error_types = list(summary.keys())
        
        return FlextExceptions.ProcessingError(
            f"Batch processing failed with {total_errors} errors of types: {', '.join(error_types)}",
            operation="batch_processing",
            context={
                "total_errors": total_errors,
                "error_summary": summary,
                "failed_identifiers": [id for id, _ in self.exceptions]
            }
        )
    
    def clear(self):
        """Clear all collected exceptions."""
        self.exceptions.clear()

# Usage in batch processing
def process_user_batch(users_data: List[dict]) -> dict:
    """Process multiple users with exception aggregation."""
    
    aggregator = ExceptionAggregator()
    successful_users = []
    
    for i, user_data in enumerate(users_data):
        user_id = f"user_{i}"
        
        try:
            # Process individual user
            result = process_single_user(user_data)
            successful_users.append(result)
            
        except FlextExceptions.BaseError as e:
            # Collect exception instead of failing immediately
            aggregator.add_exception(user_id, e)
    
    # Check if any processing failed
    if aggregator.has_exceptions():
        # Get summary of all failures
        summary = aggregator.get_exception_summary()
        
        # Decide whether to raise aggregated exception or return partial results
        if len(successful_users) == 0:
            # All failed - raise aggregated exception
            raise aggregator.create_aggregated_exception()
        
        else:
            # Partial success - return results with error summary
            return {
                "status": "partial_success",
                "successful_count": len(successful_users),
                "failed_count": len(aggregator.exceptions),
                "successful_users": successful_users,
                "error_summary": summary
            }
    
    # All succeeded
    return {
        "status": "success",
        "successful_count": len(successful_users),
        "failed_count": 0,
        "successful_users": successful_users
    }

def process_single_user(user_data: dict) -> dict:
    """Process single user - may raise FlextExceptions."""
    
    # Validation
    if not user_data.get("email"):
        raise FlextExceptions.ValidationError(
            "Email is required",
            field="email",
            value=user_data.get("email")
        )
    
    if "@" not in user_data.get("email", ""):
        raise FlextExceptions.ValidationError(
            "Invalid email format",
            field="email",
            value=user_data.get("email"),
            validation_details={"expected_format": "email"}
        )
    
    # Simulate processing
    return {
        "user_id": f"user_{hash(user_data['email'])}",
        "email": user_data["email"],
        "status": "processed"
    }
```

---

## Testing FlextExceptions

### 1. Unit Testing Exception Behavior

```python
import pytest
from flext_core import FlextExceptions

class TestFlextExceptionsImplementation:
    """Comprehensive tests for FlextException implementation."""
    
    def test_automatic_type_selection(self):
        """Test automatic exception type selection based on parameters."""
        
        # ValidationError for field parameter
        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            raise FlextExceptions(
                "Field validation failed",
                field="username",
                value=""
            )
        
        assert exc_info.value.field == "username"
        assert exc_info.value.value == ""
        assert "FLEXT_" in exc_info.value.error_code
        
        # ConfigurationError for config_key parameter
        with pytest.raises(FlextExceptions.ConfigurationError) as exc_info:
            raise FlextExceptions(
                "Config missing",
                config_key="DATABASE_URL",
                config_file=".env"
            )
        
        assert exc_info.value.config_key == "DATABASE_URL"
        assert exc_info.value.config_file == ".env"
        
        # OperationError for operation parameter
        with pytest.raises(FlextExceptions.OperationError) as exc_info:
            raise FlextExceptions(
                "Operation failed",
                operation="create_user",
                context={"retry_count": 3}
            )
        
        assert exc_info.value.operation == "create_user"
        assert exc_info.value.context["retry_count"] == 3
    
    def test_exception_context_preservation(self):
        """Test that exception context is preserved and enriched."""
        
        original_context = {"user_id": "12345", "operation_step": 1}
        
        try:
            raise FlextExceptions.ValidationError(
                "Validation failed",
                field="email",
                value="invalid",
                context=original_context
            )
        except FlextExceptions.ValidationError as e:
            # Context should be preserved
            assert e.context["user_id"] == "12345"
            assert e.context["operation_step"] == 1
            
            # Field context should be added
            assert e.context["field"] == "email"
            assert e.context["value"] == "invalid"
            
            # Correlation ID should be generated
            assert e.correlation_id is not None
            assert e.correlation_id.startswith("flext_")
    
    def test_metrics_collection(self):
        """Test automatic metrics collection."""
        
        # Clear metrics
        FlextExceptions.clear_metrics()
        
        # Generate exceptions
        for _ in range(3):
            try:
                raise FlextExceptions.ValidationError("Test validation error")
            except FlextExceptions.ValidationError:
                pass
        
        for _ in range(2):
            try:
                raise FlextExceptions.ConnectionError("Test connection error")
            except FlextExceptions.ConnectionError:
                pass
        
        # Check metrics
        metrics = FlextExceptions.get_metrics()
        assert metrics["_ValidationError"] == 3
        assert metrics["_ConnectionError"] == 2
    
    def test_error_code_consistency(self):
        """Test that error codes are consistent with FlextConstants."""
        
        validation_error = FlextExceptions.ValidationError("Test")
        assert validation_error.error_code == FlextExceptions.ErrorCodes.VALIDATION_ERROR
        
        config_error = FlextExceptions.ConfigurationError("Test")
        assert config_error.error_code == FlextExceptions.ErrorCodes.CONFIGURATION_ERROR
        
        connection_error = FlextExceptions.ConnectionError("Test")
        assert connection_error.error_code == FlextExceptions.ErrorCodes.CONNECTION_ERROR
    
    def test_exception_inheritance(self):
        """Test proper exception inheritance for Python exception handling."""
        
        # ValidationError should inherit from ValueError
        validation_error = FlextExceptions.ValidationError("Test")
        assert isinstance(validation_error, ValueError)
        assert isinstance(validation_error, FlextExceptions.BaseError)
        
        # ConnectionError should inherit from ConnectionError
        connection_error = FlextExceptions.ConnectionError("Test")
        assert isinstance(connection_error, ConnectionError)
        assert isinstance(connection_error, FlextExceptions.BaseError)
        
        # AuthenticationError should inherit from PermissionError
        auth_error = FlextExceptions.AuthenticationError("Test")
        assert isinstance(auth_error, PermissionError)
        assert isinstance(auth_error, FlextExceptions.BaseError)

### 2. Integration Testing

class TestFlextExceptionsIntegration:
    """Test FlextExceptions integration with other systems."""
    
    def test_exception_handler_integration(self):
        """Test integration with application exception handler."""
        
        handler = ApplicationExceptionHandler()
        
        # Test validation error handling
        validation_error = FlextExceptions.ValidationError(
            "Email is required",
            field="email",
            value=""
        )
        
        result = handler.handle_exception(validation_error)
        
        assert result["error_type"] == "validation_error"
        assert result["field"] == "email"
        assert result["recoverable"] is True
        assert result["correlation_id"] == validation_error.correlation_id
    
    def test_service_wrapper_integration(self):
        """Test integration with service exception wrapper."""
        
        user_service = UserService()
        
        # Test successful operation
        result = user_service.create_user({
            "name": "John Doe",
            "email": "john@example.com"
        })
        
        assert result["status"] == "created"
        assert "user_id" in result
        
        # Test validation error
        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            user_service.create_user({"name": "", "email": "john@example.com"})
        
        # Should have service context added
        assert exc_info.value.context["service"] == "user_service"
        assert exc_info.value.context["operation"] == "create_user"
    
    def test_http_client_integration(self):
        """Test integration with HTTP client exception translation."""
        
        # Mock the requests library behavior
        import unittest.mock
        
        client = HTTPClientWithFlextExceptions("https://api.example.com")
        
        # Test timeout handling
        with unittest.mock.patch.object(client.session, 'request') as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout()
            
            with pytest.raises(FlextExceptions.TimeoutError) as exc_info:
                client.get("/users/123")
            
            assert exc_info.value.timeout_seconds == 30.0
            assert "correlation_id" in exc_info.value.context
```

---

## Performance Considerations

### 1. Exception Performance Optimization

```python
import time
from typing import Optional

class OptimizedFlextException:
    """Performance-optimized FlextException usage patterns."""
    
    def __init__(self):
        self._context_cache = {}
        self._correlation_id_cache = {}
    
    def create_cached_exception(self, 
                              exception_class: type,
                              message: str,
                              cache_key: Optional[str] = None,
                              **kwargs) -> FlextExceptions.BaseError:
        """Create exception with cached context for performance."""
        
        if cache_key and cache_key in self._context_cache:
            # Use cached context to avoid repeated computation
            cached_context = self._context_cache[cache_key]
            kwargs.setdefault("context", {}).update(cached_context)
        
        return exception_class(message, **kwargs)
    
    def cache_common_context(self, cache_key: str, context: dict):
        """Cache common context for reuse."""
        self._context_cache[cache_key] = context.copy()
    
    def lazy_context_exception(self, 
                              exception_class: type,
                              message: str,
                              context_func: callable = None,
                              **kwargs) -> FlextExceptions.BaseError:
        """Create exception with lazy context evaluation."""
        
        if context_func:
            # Only evaluate context if exception is actually raised
            class LazyContextException(exception_class):
                def __init__(self, *args, **kwargs):
                    # Evaluate context lazily
                    if context_func:
                        lazy_context = context_func()
                        kwargs.setdefault("context", {}).update(lazy_context)
                    super().__init__(*args, **kwargs)
            
            return LazyContextException(message, **kwargs)
        
        return exception_class(message, **kwargs)

# Usage for performance-critical code
def performance_critical_validation(data_batch: list) -> list:
    """Validation with performance optimizations."""
    
    optimizer = OptimizedFlextException()
    
    # Cache common context
    optimizer.cache_common_context("validation_context", {
        "batch_size": len(data_batch),
        "validation_timestamp": time.time(),
        "validator_version": "1.0"
    })
    
    results = []
    
    for i, item in enumerate(data_batch):
        try:
            # Perform validation
            if not item.get("id"):
                raise optimizer.create_cached_exception(
                    FlextExceptions.ValidationError,
                    "ID is required",
                    cache_key="validation_context",
                    field="id",
                    value=item.get("id"),
                    context={"batch_index": i}
                )
            
            results.append({"status": "valid", "item": item})
            
        except FlextExceptions.ValidationError as e:
            results.append({
                "status": "invalid", 
                "error": e.message,
                "correlation_id": e.correlation_id
            })
    
    return results
```

### 2. Memory-Efficient Exception Handling

```python
import weakref
from typing import WeakSet

class MemoryEfficientExceptionTracker:
    """Memory-efficient exception tracking for long-running applications."""
    
    def __init__(self, max_exceptions: int = 1000):
        self.max_exceptions = max_exceptions
        self._exceptions: WeakSet[FlextExceptions.BaseError] = WeakSet()
        self._metrics_buffer = {}
        self._buffer_size = 0
    
    def track_exception(self, exception: FlextExceptions.BaseError):
        """Track exception with memory-efficient storage."""
        
        # Add to weak reference set (automatically cleaned up when exception is garbage collected)
        self._exceptions.add(exception)
        
        # Update metrics buffer
        exc_type = type(exception).__name__
        self._metrics_buffer[exc_type] = self._metrics_buffer.get(exc_type, 0) + 1
        self._buffer_size += 1
        
        # Flush buffer if it gets too large
        if self._buffer_size > self.max_exceptions:
            self._flush_metrics_buffer()
    
    def _flush_metrics_buffer(self):
        """Flush metrics buffer to persistent storage."""
        
        # In a real implementation, this would write to a database or log file
        print(f"Flushing metrics buffer: {self._metrics_buffer}")
        
        # Clear buffer
        self._metrics_buffer.clear()
        self._buffer_size = 0
    
    def get_active_exceptions_count(self) -> int:
        """Get count of exceptions still in memory."""
        return len(self._exceptions)
    
    def cleanup(self):
        """Force cleanup of weak references."""
        # Weak references are automatically cleaned up, but we can force it
        self._exceptions.clear()
        self._flush_metrics_buffer()

# Usage in long-running applications
class LongRunningService:
    """Service optimized for long-running applications."""
    
    def __init__(self):
        self.exception_tracker = MemoryEfficientExceptionTracker()
    
    def process_request(self, request_data: dict) -> dict:
        """Process request with memory-efficient exception tracking."""
        
        try:
            # Process request
            return self._do_process_request(request_data)
        
        except FlextExceptions.BaseError as e:
            # Track exception efficiently
            self.exception_tracker.track_exception(e)
            raise
    
    def _do_process_request(self, request_data: dict) -> dict:
        """Actual request processing logic."""
        
        if not request_data:
            raise FlextExceptions.ValidationError(
                "Request data cannot be empty",
                field="request_data",
                value=request_data
            )
        
        return {"status": "processed", "data": request_data}
    
    def get_health_status(self) -> dict:
        """Get service health status including exception metrics."""
        
        return {
            "status": "healthy",
            "active_exceptions": self.exception_tracker.get_active_exceptions_count(),
            "metrics_buffer_size": self.exception_tracker._buffer_size
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.exception_tracker.cleanup()
```

---

## Best Practices Summary

### 1. Exception Creation
- Use specific exception types when the error category is known
- Include rich context information for debugging
- Preserve correlation IDs across service boundaries
- Use automatic type selection for generic error handling

### 2. Exception Handling
- Implement centralized exception handling for consistent responses
- Use environment-specific configuration for error details
- Aggregate exceptions in batch processing scenarios
- Implement recovery strategies for transient failures

### 3. Monitoring and Observability
- Enable automatic metrics collection in production
- Set up alerting for critical error patterns
- Use correlation IDs for distributed tracing
- Export metrics to monitoring systems periodically

### 4. Performance Optimization
- Cache common exception context
- Use lazy context evaluation for expensive operations
- Implement memory-efficient tracking for long-running applications
- Consider exception aggregation for batch operations

### 5. Testing
- Test automatic exception type selection
- Verify context preservation and enrichment
- Test metrics collection and aggregation
- Validate integration with other systems

---

This implementation guide provides comprehensive patterns for integrating FlextExceptions into FLEXT ecosystem projects, ensuring consistent error handling, comprehensive observability, and optimal performance across all applications.
