# FlextLogger Implementation Guide

**Step-by-step guide for implementing FlextLogger as the comprehensive structured logging foundation in FLEXT ecosystem projects.**

---

## Overview

This guide provides comprehensive instructions for integrating FlextLogger into new and existing FLEXT projects. FlextLogger serves as the enterprise structured logging system, providing JSON output, automatic correlation ID generation, request context tracking, operation performance metrics, and sensitive data sanitization across all FLEXT components.

---

## Quick Start Implementation

### Step 1: Basic FlextLogger Usage

```python
from flext_core import FlextLogger

# Initialize logger (singleton pattern - reuses instances by name)
logger = FlextLogger(__name__)

# Basic structured logging with context
logger.debug("Starting application initialization", component="startup", version="2.0.0")
logger.info("User request received", user_id=123, action="login", ip_address="192.168.1.100")
logger.warning("Rate limit approaching", current_requests=95, limit=100, user_id=456)
logger.error("Database connection failed", error="Connection timeout", retries=3, max_retries=5)
logger.critical("System overload detected", cpu_usage=98, memory_usage=97, load_avg=4.5)

# Exception logging with automatic stack trace capture
try:
    risky_database_operation()
except Exception:
    logger.exception("Database operation failed", operation="user_lookup", table="users")
```

### Step 2: Error Logging with Rich Context

```python
# Error logging with Exception objects
try:
    user_service.create_user(user_data)
except ValidationError as e:
    logger.error("User validation failed", 
        error=e,                    # Exception object with stack trace
        user_id=user_data.get("id"),
        validation_fields=["email", "password"],
        form_step="registration"
    )

# Error logging with string messages
logger.error("Custom validation error", 
    error="Email format is invalid",     # String error message
    field="email",
    value="not-an-email",
    user_id=123,
    validation_rule="email_regex"
)

# Critical errors with system context
logger.critical("Memory exhaustion detected",
    error="Out of memory",
    memory_used_gb=7.8,
    memory_available_gb=8.0,
    process_count=127,
    requires_restart=True
)
```

### Step 3: Correlation ID and Request Tracking

```python
# Set global correlation ID for distributed tracing
FlextLogger.set_global_correlation_id("req_abc123456")

# All subsequent log entries include this correlation ID
logger.info("Processing user request")  # correlation_id: "req_abc123456"
logger.info("Database query executed")  # correlation_id: "req_abc123456"

# Instance-level correlation ID (overrides global)
logger.set_correlation_id("specific_operation_789")
logger.info("Specific operation completed")  # correlation_id: "specific_operation_789"

# Request context tracking (thread-local storage)
logger.set_request_context(
    request_id="req_789",
    user_id="user_123", 
    session_id="sess_456",
    client_ip="192.168.1.100",
    user_agent="Mozilla/5.0..."
)

# All logs in this thread include request context automatically
logger.info("Request processing started")  # Includes all request context
logger.error("Validation failed")          # Includes all request context

# Clear request context when request processing completes
logger.clear_request_context()
```

---

## Detailed Implementation Patterns

### 1. Service-Level Integration

#### HTTP API Service Integration

```python
from flext_core import FlextLogger
from uuid import uuid4
import time

class APIService:
    def __init__(self):
        self.logger = FlextLogger(__name__)
        
        # Set service-level context that appears in all logs
        self.logger.set_context(
            service="user-api",
            version="2.1.0",
            component="api_handler"
        )
    
    def handle_request(self, request):
        """Handle HTTP request with comprehensive logging."""
        
        # Generate unique correlation ID for this request
        correlation_id = f"req_{uuid4().hex[:16]}"
        FlextLogger.set_global_correlation_id(correlation_id)
        
        # Set request-specific context (thread-local)
        self.logger.set_request_context(
            method=request.method,
            path=request.path,
            content_type=request.headers.get("Content-Type"),
            user_agent=request.headers.get("User-Agent"),
            client_ip=request.remote_addr,
            request_size=len(request.get_data()) if hasattr(request, 'get_data') else 0
        )
        
        # Start operation tracking for performance monitoring
        operation_id = self.logger.start_operation(
            "api_request",
            endpoint=request.path,
            method=request.method
        )
        
        try:
            # Log request start
            self.logger.info("API request started",
                endpoint=f"{request.method} {request.path}",
                content_length=request.content_length or 0
            )
            
            # Process request
            result = self.process_request(request)
            
            # Log successful completion
            self.logger.complete_operation(
                operation_id,
                success=True,
                status_code=200,
                response_size=len(str(result)) if result else 0
            )
            
            self.logger.info("API request completed successfully",
                status_code=200,
                processing_result="success"
            )
            
            return result
            
        except ValidationError as e:
            # Handle validation errors
            self.logger.complete_operation(
                operation_id,
                success=False,
                status_code=400,
                error_type="ValidationError"
            )
            
            self.logger.error("API request validation failed",
                error=e,
                status_code=400,
                validation_errors=getattr(e, 'errors', [])
            )
            raise
            
        except AuthenticationError as e:
            # Handle authentication errors  
            self.logger.complete_operation(
                operation_id,
                success=False,
                status_code=401,
                error_type="AuthenticationError"
            )
            
            self.logger.error("API request authentication failed",
                error=e,
                status_code=401,
                auth_method=getattr(e, 'auth_method', 'unknown')
            )
            raise
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.complete_operation(
                operation_id,
                success=False,
                status_code=500,
                error_type=type(e).__name__
            )
            
            self.logger.error("API request processing failed",
                error=e,
                status_code=500,
                unexpected_error=True
            )
            raise
            
        finally:
            # Always clear request context when done
            self.logger.clear_request_context()
    
    def process_request(self, request):
        """Process the actual request logic."""
        # Simulate request processing
        time.sleep(0.1)  # Simulate work
        return {"status": "success", "data": "processed"}
```

#### Database Service Integration

```python
class DatabaseService:
    def __init__(self, connection_string):
        self.logger = FlextLogger(__name__)
        self.connection_string = connection_string
        
        # Set database-specific context
        self.logger.set_context(
            component="database",
            driver="oracle",
            database_type="production"
        )
    
    def execute_query(self, sql, parameters=None, query_name=None):
        """Execute database query with comprehensive logging."""
        
        # Create query-specific logger with bound context
        query_logger = self.logger.bind(
            query_name=query_name or "unnamed_query",
            query_type=self._classify_query(sql),
            parameter_count=len(parameters) if parameters else 0,
            sql_length=len(sql)
        )
        
        # Use context manager for automatic duration tracking
        with query_logger.track_duration("database_query") as tracker:
            try:
                # Log query start
                query_logger.info("Database query started",
                    sql_preview=sql[:200] + "..." if len(sql) > 200 else sql,
                    has_parameters=bool(parameters)
                )
                
                # Execute query (simulate with actual database call)
                start_time = time.time()
                result = self._execute_sql(sql, parameters)
                execution_time = time.time() - start_time
                
                # Add execution context to tracker
                tracker.add_context(
                    rows_affected=getattr(result, 'rowcount', 0),
                    execution_time_ms=round(execution_time * 1000, 3),
                    result_size=len(result) if hasattr(result, '__len__') else 0,
                    cache_hit=self._was_cache_hit(sql),
                    execution_plan=self._get_execution_plan_type(sql)
                )
                
                query_logger.info("Database query completed successfully",
                    rows_returned=len(result) if hasattr(result, '__len__') else 0,
                    performance_rating=self._rate_query_performance(execution_time)
                )
                
                return result
                
            except Exception as e:
                query_logger.error("Database query failed",
                    error=e,
                    sql_preview=sql[:100] + "..." if len(sql) > 100 else sql,
                    error_category=self._categorize_db_error(e)
                )
                raise
    
    def _classify_query(self, sql):
        """Classify SQL query type."""
        sql_upper = sql.upper().strip()
        if sql_upper.startswith("SELECT"):
            return "select"
        elif sql_upper.startswith(("INSERT", "UPDATE", "DELETE")):
            return "modify"
        elif sql_upper.startswith(("CREATE", "ALTER", "DROP")):
            return "ddl"
        else:
            return "other"
    
    def _execute_sql(self, sql, parameters):
        """Simulate SQL execution."""
        time.sleep(0.05)  # Simulate database work
        return [{"id": 1, "name": "result"}]  # Mock result
    
    def _was_cache_hit(self, sql):
        """Simulate cache hit detection."""
        return "FROM users" in sql.upper()  # Mock cache logic
    
    def _get_execution_plan_type(self, sql):
        """Simulate execution plan analysis."""
        if "INDEX" in sql.upper():
            return "index_scan"
        elif "JOIN" in sql.upper():
            return "nested_loop"
        else:
            return "table_scan"
    
    def _rate_query_performance(self, execution_time):
        """Rate query performance."""
        if execution_time < 0.01:
            return "excellent"
        elif execution_time < 0.1:
            return "good"
        elif execution_time < 1.0:
            return "acceptable"
        else:
            return "poor"
    
    def _categorize_db_error(self, error):
        """Categorize database error."""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return "timeout"
        elif "connection" in error_str:
            return "connection"
        elif "constraint" in error_str:
            return "constraint_violation"
        else:
            return "unknown"
```

### 2. Context Management and Logger Binding

#### Context Binding for Specific Operations

```python
class UserService:
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.logger.set_context(
            service="user_service",
            version="1.5.0"
        )
    
    def create_user(self, user_data):
        """Create user with operation-specific logging context."""
        
        # Create user-specific logger with bound context
        user_logger = self.logger.bind(
            operation="create_user",
            user_email=user_data.get("email"),
            registration_type=user_data.get("type", "standard"),
            source=user_data.get("source", "web")
        )
        
        # Alternative context creation method
        # user_logger = self.logger.with_context(
        #     operation="create_user",
        #     user_email=user_data.get("email")
        # )
        
        op_id = user_logger.start_operation("user_creation")
        
        try:
            # Validate user data
            user_logger.info("User validation started")
            self._validate_user_data(user_data, user_logger)
            
            # Create user record
            user_logger.info("User record creation started")
            user_id = self._create_user_record(user_data, user_logger)
            
            # Send welcome email
            user_logger.info("Welcome email sending started")
            self._send_welcome_email(user_data, user_logger)
            
            user_logger.complete_operation(op_id, success=True, user_id=user_id)
            user_logger.info("User creation completed successfully", user_id=user_id)
            
            return {"user_id": user_id, "status": "created"}
            
        except Exception as e:
            user_logger.complete_operation(op_id, success=False, error_type=type(e).__name__)
            user_logger.error("User creation failed", error=e)
            raise
    
    def _validate_user_data(self, user_data, logger):
        """Validate user data with detailed logging."""
        validation_logger = logger.bind(validation_step="user_data")
        
        # Email validation
        if not user_data.get("email"):
            validation_logger.error("Email validation failed", 
                error="Email is required",
                field="email"
            )
            raise ValidationError("Email is required")
        
        if "@" not in user_data["email"]:
            validation_logger.error("Email format validation failed",
                error="Invalid email format", 
                field="email",
                value=user_data["email"]
            )
            raise ValidationError("Invalid email format")
        
        validation_logger.info("Email validation passed")
        
        # Password validation
        if len(user_data.get("password", "")) < 8:
            validation_logger.error("Password validation failed",
                error="Password too short",
                field="password",
                min_length=8,
                actual_length=len(user_data.get("password", ""))
            )
            raise ValidationError("Password too short")
        
        validation_logger.info("Password validation passed")
        validation_logger.info("User data validation completed successfully")
    
    def _create_user_record(self, user_data, logger):
        """Create user record with database logging."""
        db_logger = logger.bind(database_operation="insert", table="users")
        
        try:
            # Simulate user creation
            user_id = f"user_{int(time.time())}"
            time.sleep(0.1)  # Simulate database work
            
            db_logger.info("User record created successfully", 
                user_id=user_id,
                database_duration_ms=100
            )
            
            return user_id
            
        except Exception as e:
            db_logger.error("User record creation failed", error=e)
            raise
    
    def _send_welcome_email(self, user_data, logger):
        """Send welcome email with email service logging."""
        email_logger = logger.bind(
            email_service="sendgrid",
            email_type="welcome",
            recipient=user_data["email"]
        )
        
        try:
            # Simulate email sending
            time.sleep(0.05)
            
            email_logger.info("Welcome email sent successfully",
                email_id=f"email_{int(time.time())}",
                delivery_status="sent"
            )
            
        except Exception as e:
            email_logger.error("Welcome email sending failed", 
                error=e,
                email_service_status="failed"
            )
            # Don't raise - email failure shouldn't break user creation
```

#### Permanent vs Request Context Management

```python
class ApplicationService:
    def __init__(self):
        self.logger = FlextLogger(__name__)
        
        # Set permanent context that appears in all logs from this instance
        self.logger.set_context({
            "service": "application_service",
            "version": "2.0.0",
            "environment": "production",
            "instance_id": "app-01"
        })
    
    def handle_user_request(self, user_id, request_data):
        """Handle request with both permanent and request-specific context."""
        
        # Set thread-local request context
        self.logger.set_request_context(
            user_id=user_id,
            request_type=request_data.get("type"),
            request_timestamp=time.time(),
            session_id=request_data.get("session_id")
        )
        
        try:
            # This log includes both permanent context and request context
            self.logger.info("Request processing started",
                operation="handle_user_request",
                request_size=len(str(request_data))
            )
            
            # Call other methods - they inherit the request context
            result = self._process_request_data(request_data)
            
            self.logger.info("Request processing completed",
                result_status="success",
                processing_duration_ms=50
            )
            
            return result
            
        finally:
            # Always clear request context when done
            self.logger.clear_request_context()
    
    def _process_request_data(self, request_data):
        """Process request data - inherits request context automatically."""
        
        # This log automatically includes the request context set above
        self.logger.info("Data processing started",
            data_type=type(request_data).__name__,
            processing_step="validation"
        )
        
        # Simulate processing
        time.sleep(0.05)
        
        self.logger.info("Data processing completed",
            processing_step="completed",
            records_processed=1
        )
        
        return {"status": "processed", "data": request_data}
```

### 3. Performance Tracking and Operation Monitoring

#### Advanced Operation Tracking

```python
class DataProcessor:
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.logger.set_context(component="data_processor")
    
    def process_large_dataset(self, dataset):
        """Process large dataset with detailed performance tracking."""
        
        # Create processor-specific logger
        processor_logger = self.logger.bind(
            dataset_type=type(dataset).__name__,
            dataset_size=len(dataset),
            processing_mode="batch"
        )
        
        # Start main operation tracking
        main_op_id = processor_logger.start_operation(
            "dataset_processing",
            total_records=len(dataset),
            expected_duration_seconds=len(dataset) * 0.01  # Estimate
        )
        
        results = []
        
        try:
            # Phase 1: Data validation
            with processor_logger.track_duration("data_validation") as validation_tracker:
                processor_logger.info("Data validation phase started")
                
                valid_records = self._validate_dataset(dataset, processor_logger)
                
                validation_tracker.add_context(
                    records_validated=len(dataset),
                    valid_records=len(valid_records),
                    invalid_records=len(dataset) - len(valid_records),
                    validation_rate=len(valid_records) / len(dataset)
                )
                
                processor_logger.info("Data validation phase completed",
                    validation_success_rate=f"{len(valid_records)/len(dataset)*100:.1f}%"
                )
            
            # Phase 2: Data transformation  
            with processor_logger.track_duration("data_transformation") as transform_tracker:
                processor_logger.info("Data transformation phase started")
                
                transformed_records = []
                for i, record in enumerate(valid_records):
                    transformed = self._transform_record(record, processor_logger)
                    transformed_records.append(transformed)
                    
                    # Log progress every 1000 records
                    if (i + 1) % 1000 == 0:
                        processor_logger.info("Transformation progress", 
                            processed_count=i + 1,
                            total_count=len(valid_records),
                            progress_percent=f"{(i+1)/len(valid_records)*100:.1f}%"
                        )
                
                transform_tracker.add_context(
                    records_transformed=len(transformed_records),
                    transformation_rate=len(transformed_records) / len(valid_records),
                    avg_transform_time_ms=transform_tracker.duration_ms / len(transformed_records)
                )
                
                processor_logger.info("Data transformation phase completed")
            
            # Phase 3: Data persistence
            with processor_logger.track_duration("data_persistence") as persistence_tracker:
                processor_logger.info("Data persistence phase started")
                
                saved_count = self._save_records(transformed_records, processor_logger)
                
                persistence_tracker.add_context(
                    records_saved=saved_count,
                    persistence_success_rate=saved_count / len(transformed_records),
                    records_per_second=saved_count / (persistence_tracker.duration_ms / 1000)
                )
                
                processor_logger.info("Data persistence phase completed")
            
            # Complete main operation successfully
            processor_logger.complete_operation(
                main_op_id,
                success=True,
                total_processed=len(dataset),
                final_result_count=saved_count,
                overall_success_rate=saved_count / len(dataset)
            )
            
            processor_logger.info("Dataset processing completed successfully",
                input_records=len(dataset),
                output_records=saved_count,
                processing_efficiency=f"{saved_count/len(dataset)*100:.1f}%"
            )
            
            return {"processed": saved_count, "success_rate": saved_count / len(dataset)}
            
        except Exception as e:
            processor_logger.complete_operation(
                main_op_id,
                success=False,
                error_type=type(e).__name__,
                processing_stage=self._determine_current_stage(e)
            )
            
            processor_logger.error("Dataset processing failed", error=e)
            raise
    
    def _validate_dataset(self, dataset, logger):
        """Validate dataset records."""
        valid_records = []
        
        for i, record in enumerate(dataset):
            try:
                if self._is_valid_record(record):
                    valid_records.append(record)
                else:
                    logger.warning("Record validation failed",
                        record_index=i,
                        record_id=record.get("id", "unknown"),
                        validation_error="Invalid record format"
                    )
            except Exception as e:
                logger.error("Record validation error",
                    record_index=i,
                    error=e
                )
        
        return valid_records
    
    def _transform_record(self, record, logger):
        """Transform individual record."""
        try:
            # Simulate transformation
            transformed = {
                "id": record.get("id"),
                "processed_data": f"transformed_{record.get('data', '')}",
                "timestamp": time.time()
            }
            return transformed
            
        except Exception as e:
            logger.error("Record transformation failed",
                record_id=record.get("id", "unknown"),
                error=e
            )
            raise
    
    def _save_records(self, records, logger):
        """Save records with batch processing."""
        saved_count = 0
        batch_size = 100
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                # Simulate batch save
                time.sleep(0.01 * len(batch))  # Simulate database work
                saved_count += len(batch)
                
                logger.info("Batch saved successfully",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    total_saved=saved_count
                )
                
            except Exception as e:
                logger.error("Batch save failed",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    error=e
                )
                raise
        
        return saved_count
    
    def _is_valid_record(self, record):
        """Check if record is valid."""
        return isinstance(record, dict) and "id" in record
    
    def _determine_current_stage(self, error):
        """Determine processing stage based on error."""
        error_str = str(error).lower()
        if "validation" in error_str:
            return "validation"
        elif "transform" in error_str:
            return "transformation"
        elif "save" in error_str or "persist" in error_str:
            return "persistence"
        else:
            return "unknown"
```

#### Microservice Communication with Correlation Tracking

```python
import requests

class ServiceClient:
    def __init__(self, service_name, base_url):
        self.service_name = service_name
        self.base_url = base_url
        self.logger = FlextLogger(__name__)
        self.logger.set_context(
            component="service_client",
            target_service=service_name
        )
    
    def call_service(self, endpoint, data, method="POST"):
        """Make service call with correlation tracking and comprehensive logging."""
        
        # Create call-specific logger
        call_logger = self.logger.bind(
            service_call=True,
            target_endpoint=endpoint,
            http_method=method,
            request_size=len(str(data)) if data else 0
        )
        
        # Get correlation ID for propagation
        correlation_id = call_logger.get_correlation_id()
        
        # Prepare headers with correlation ID
        headers = {
            "Content-Type": "application/json",
            "X-Correlation-ID": correlation_id,
            "X-Calling-Service": "current-service",
            "User-Agent": "FlextServiceClient/1.0"
        }
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Track the entire service call duration
        with call_logger.track_duration("service_call") as tracker:
            try:
                call_logger.info("Service call started",
                    target_url=url,
                    correlation_id=correlation_id,
                    has_payload=bool(data)
                )
                
                # Make the HTTP request
                start_time = time.time()
                if method.upper() == "POST":
                    response = requests.post(url, json=data, headers=headers, timeout=30)
                elif method.upper() == "GET":
                    response = requests.get(url, headers=headers, timeout=30)
                elif method.upper() == "PUT":
                    response = requests.put(url, json=data, headers=headers, timeout=30)
                else:
                    response = requests.request(method, url, json=data, headers=headers, timeout=30)
                
                request_duration = time.time() - start_time
                
                # Add response context to tracker
                tracker.add_context(
                    status_code=response.status_code,
                    response_size=len(response.content),
                    request_duration_ms=round(request_duration * 1000, 3),
                    response_headers=dict(response.headers),
                    success=200 <= response.status_code < 300
                )
                
                # Log response details
                if 200 <= response.status_code < 300:
                    call_logger.info("Service call completed successfully",
                        status_code=response.status_code,
                        response_size=len(response.content),
                        request_duration_ms=round(request_duration * 1000, 3)
                    )
                    
                    return response.json() if response.content else None
                    
                else:
                    call_logger.error("Service call returned error status",
                        status_code=response.status_code,
                        response_body=response.text[:500],  # Limit response body size
                        error_category="http_error"
                    )
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout as e:
                call_logger.error("Service call timed out",
                    error=e,
                    timeout_seconds=30,
                    error_category="timeout"
                )
                raise
                
            except requests.exceptions.ConnectionError as e:
                call_logger.error("Service call connection failed",
                    error=e,
                    target_url=url,
                    error_category="connection"
                )
                raise
                
            except requests.exceptions.RequestException as e:
                call_logger.error("Service call request failed", 
                    error=e,
                    error_category="request_error"
                )
                raise
                
            except Exception as e:
                call_logger.error("Service call unexpected error",
                    error=e,
                    error_category="unexpected"
                )
                raise

class OrderService:
    """Example service using ServiceClient with correlation tracking."""
    
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.user_client = ServiceClient("user-service", "https://api.users.com")
        self.payment_client = ServiceClient("payment-service", "https://api.payments.com")
        self.inventory_client = ServiceClient("inventory-service", "https://api.inventory.com")
    
    def process_order(self, order_data):
        """Process order with multiple service calls maintaining correlation."""
        
        # Set correlation ID for the entire order processing
        order_correlation_id = f"order_{order_data['order_id']}"
        FlextLogger.set_global_correlation_id(order_correlation_id)
        
        order_logger = self.logger.bind(
            order_id=order_data["order_id"],
            customer_id=order_data["customer_id"],
            order_total=order_data["total"]
        )
        
        op_id = order_logger.start_operation("order_processing")
        
        try:
            # Step 1: Validate user
            order_logger.info("User validation started")
            user_data = self.user_client.call_service(
                f"users/{order_data['customer_id']}", 
                None, 
                method="GET"
            )
            order_logger.info("User validation completed", user_valid=bool(user_data))
            
            # Step 2: Check inventory
            order_logger.info("Inventory check started")
            inventory_data = self.inventory_client.call_service(
                "inventory/check",
                {"items": order_data["items"]},
                method="POST"
            )
            order_logger.info("Inventory check completed", items_available=inventory_data["available"])
            
            # Step 3: Process payment
            order_logger.info("Payment processing started")
            payment_result = self.payment_client.call_service(
                "payments/charge",
                {
                    "customer_id": order_data["customer_id"],
                    "amount": order_data["total"],
                    "order_id": order_data["order_id"]
                },
                method="POST"
            )
            order_logger.info("Payment processing completed", 
                payment_id=payment_result["payment_id"],
                payment_status=payment_result["status"]
            )
            
            # Complete order processing
            order_logger.complete_operation(op_id, success=True, 
                user_id=user_data["id"],
                payment_id=payment_result["payment_id"],
                items_count=len(order_data["items"])
            )
            
            order_logger.info("Order processing completed successfully")
            
            return {
                "order_id": order_data["order_id"],
                "status": "completed",
                "payment_id": payment_result["payment_id"]
            }
            
        except Exception as e:
            order_logger.complete_operation(op_id, success=False, error_type=type(e).__name__)
            order_logger.error("Order processing failed", error=e)
            raise
```

---

## Configuration and Environment Management

### 1. System-Wide Configuration

```python
from flext_core import FlextLogger

def configure_application_logging(environment):
    """Configure logging for different environments."""
    
    if environment == "development":
        # Development configuration
        FlextLogger.configure(
            log_level="DEBUG",
            json_output=False,          # Colored console output
            include_source=True,        # Include file/line info
            structured_output=True      # Enable all processors
        )
        
        # Additional development config
        dev_config = {
            "environment": "development",
            "log_level": "DEBUG",
            "enable_console_output": True,
            "enable_json_logging": False,
            "enable_correlation_tracking": True,
            "enable_performance_logging": True,
            "enable_sensitive_data_sanitization": True,
            "max_log_message_size": 20000,  # Large for debugging
            "async_logging_enabled": False   # Sync for immediate feedback
        }
        
    elif environment == "production":
        # Production configuration
        FlextLogger.configure(
            log_level="WARNING",
            json_output=True,           # JSON for log aggregation
            include_source=False,       # Reduce log size
            structured_output=True
        )
        
        # Additional production config
        prod_config = {
            "environment": "production",
            "log_level": "WARNING",
            "enable_console_output": False,
            "enable_json_logging": True,
            "enable_correlation_tracking": True,
            "enable_performance_logging": True,
            "enable_sensitive_data_sanitization": True,
            "max_log_message_size": 5000,   # Limit size
            "async_logging_enabled": True   # Async for performance
        }
        
    elif environment == "test":
        # Test configuration  
        FlextLogger.configure(
            log_level="ERROR",
            json_output=False,
            include_source=False,
            structured_output=False     # Minimal processing
        )
        
        # Additional test config
        test_config = {
            "environment": "test",
            "log_level": "ERROR",
            "enable_console_output": False,
            "enable_json_logging": False,
            "enable_correlation_tracking": False,
            "enable_performance_logging": False,
            "enable_sensitive_data_sanitization": True,
            "max_log_message_size": 1000,
            "async_logging_enabled": False
        }
    
    else:
        # Default/staging configuration
        prod_config = {
            "environment": environment,
            "log_level": "INFO",
            "enable_console_output": True,
            "enable_json_logging": True,
            "enable_correlation_tracking": True,
            "enable_performance_logging": True,
            "enable_sensitive_data_sanitization": True,
            "max_log_message_size": 10000,
            "async_logging_enabled": False
        }
    
    # Apply the configuration
    result = FlextLogger.configure_logging_system(locals()[f"{environment}_config"])
    if result.success:
        logger = FlextLogger("application")
        logger.info(f"Logging configured for {environment} environment",
            config_applied=True,
            log_level=result.value["log_level"],
            json_output=result.value["enable_json_logging"]
        )
    else:
        print(f"Failed to configure logging: {result.error}")
```

### 2. Application Startup Integration

```python
import os
from flext_core import FlextLogger

class Application:
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.configure_logging()
        self.logger = FlextLogger(__name__)
        
        # Set application-wide context
        self.logger.set_context(
            application="my-service",
            version=os.getenv("APP_VERSION", "1.0.0"),
            environment=self.environment,
            instance_id=os.getenv("INSTANCE_ID", "local-dev")
        )
    
    def configure_logging(self):
        """Configure logging based on environment."""
        
        # Auto-create environment-specific configuration
        config_result = FlextLogger.create_environment_logging_config(self.environment)
        
        if config_result.success:
            config = config_result.value
            
            # Apply the configuration
            apply_result = FlextLogger.configure_logging_system(config)
            
            if apply_result.success:
                # Also apply system-wide structlog configuration
                FlextLogger.configure(
                    log_level=config["log_level"],
                    json_output=config["enable_json_logging"],
                    include_source=self.environment != "production",
                    structured_output=True
                )
            else:
                print(f"Failed to apply logging config: {apply_result.error}")
        else:
            print(f"Failed to create logging config: {config_result.error}")
    
    def startup(self):
        """Application startup with comprehensive logging."""
        
        startup_logger = self.logger.bind(phase="startup")
        
        startup_op_id = startup_logger.start_operation("application_startup")
        
        try:
            startup_logger.info("Application startup initiated",
                environment=self.environment,
                python_version=os.sys.version,
                working_directory=os.getcwd()
            )
            
            # Initialize components
            self._initialize_database(startup_logger)
            self._initialize_cache(startup_logger)
            self._initialize_external_services(startup_logger)
            self._start_background_tasks(startup_logger)
            
            startup_logger.complete_operation(startup_op_id, success=True)
            startup_logger.info("Application startup completed successfully",
                startup_time_ms=1500  # Would be calculated
            )
            
        except Exception as e:
            startup_logger.complete_operation(startup_op_id, success=False, error_type=type(e).__name__)
            startup_logger.critical("Application startup failed", error=e)
            raise
    
    def _initialize_database(self, logger):
        """Initialize database with logging."""
        db_logger = logger.bind(component="database")
        
        try:
            db_logger.info("Database initialization started")
            # Simulate database initialization
            time.sleep(0.1)
            db_logger.info("Database initialization completed")
            
        except Exception as e:
            db_logger.error("Database initialization failed", error=e)
            raise
    
    def _initialize_cache(self, logger):
        """Initialize cache with logging."""
        cache_logger = logger.bind(component="cache")
        
        try:
            cache_logger.info("Cache initialization started")
            # Simulate cache initialization
            time.sleep(0.05)
            cache_logger.info("Cache initialization completed")
            
        except Exception as e:
            cache_logger.error("Cache initialization failed", error=e)
            raise
    
    def _initialize_external_services(self, logger):
        """Initialize external services with logging."""
        services_logger = logger.bind(component="external_services")
        
        try:
            services_logger.info("External services initialization started")
            # Simulate service initialization
            time.sleep(0.2)
            services_logger.info("External services initialization completed")
            
        except Exception as e:
            services_logger.error("External services initialization failed", error=e)
            raise
    
    def _start_background_tasks(self, logger):
        """Start background tasks with logging."""
        tasks_logger = logger.bind(component="background_tasks")
        
        try:
            tasks_logger.info("Background tasks startup initiated")
            # Simulate background task startup
            time.sleep(0.1)
            tasks_logger.info("Background tasks startup completed")
            
        except Exception as e:
            tasks_logger.error("Background tasks startup failed", error=e)
            raise
    
    def shutdown(self):
        """Application shutdown with logging."""
        shutdown_logger = self.logger.bind(phase="shutdown")
        
        shutdown_op_id = shutdown_logger.start_operation("application_shutdown")
        
        try:
            shutdown_logger.info("Application shutdown initiated")
            
            # Cleanup components in reverse order
            self._stop_background_tasks(shutdown_logger)
            self._cleanup_external_services(shutdown_logger)
            self._cleanup_cache(shutdown_logger)
            self._cleanup_database(shutdown_logger)
            
            shutdown_logger.complete_operation(shutdown_op_id, success=True)
            shutdown_logger.info("Application shutdown completed successfully")
            
        except Exception as e:
            shutdown_logger.complete_operation(shutdown_op_id, success=False, error_type=type(e).__name__)
            shutdown_logger.critical("Application shutdown failed", error=e)
            raise
    
    def _stop_background_tasks(self, logger):
        """Stop background tasks."""
        logger.bind(component="background_tasks").info("Background tasks stopped")
    
    def _cleanup_external_services(self, logger):
        """Cleanup external services."""
        logger.bind(component="external_services").info("External services cleaned up")
    
    def _cleanup_cache(self, logger):
        """Cleanup cache."""
        logger.bind(component="cache").info("Cache cleaned up")
    
    def _cleanup_database(self, logger):
        """Cleanup database."""
        logger.bind(component="database").info("Database cleaned up")

# Application usage
if __name__ == "__main__":
    app = Application()
    try:
        app.startup()
        # Run application...
    finally:
        app.shutdown()
```

---

## Performance Optimization Patterns

### 1. High-Throughput Application Configuration

```python
def configure_high_performance_logging():
    """Configure logging for high-throughput applications."""
    
    # High-performance configuration
    high_perf_config = {
        "performance_level": "high",
        "async_logging_enabled": True,      # Non-blocking logging
        "buffer_size": 10000,               # Large buffer
        "flush_interval_ms": 500,           # Fast flush
        "max_concurrent_operations": 1000,  # High concurrency
        "enable_log_compression": True,     # Reduce I/O
        "batch_log_processing": True,       # Process in batches
        "disable_trace_logging": True,      # Skip trace level
        "log_level_caching": True,          # Cache level checks
    }
    
    result = FlextLogger.optimize_logging_performance(high_perf_config)
    
    if result.success:
        config = result.value
        
        # Apply optimized settings
        FlextLogger.configure(
            log_level="INFO",               # Reduced verbosity
            json_output=True,              # Structured output
            include_source=False,          # Reduce overhead  
            structured_output=True
        )
        
        logger = FlextLogger("performance")
        logger.info("High-performance logging configured",
            async_enabled=config["async_logging_enabled"],
            buffer_size=config["buffer_size"],
            batch_processing=config["batch_log_processing"]
        )
    else:
        print(f"Performance optimization failed: {result.error}")

class HighThroughputProcessor:
    """Example processor optimized for high throughput logging."""
    
    def __init__(self):
        self.logger = FlextLogger(__name__)
        
        # Set context once to avoid repeated work
        self.logger.set_context(
            processor_type="high_throughput",
            version="2.0.0",
            optimization_level="maximum"
        )
        
        # Pre-create bound loggers for common operations
        self.request_logger = self.logger.bind(operation_type="request_processing")
        self.batch_logger = self.logger.bind(operation_type="batch_processing")
        self.error_logger = self.logger.bind(operation_type="error_handling")
    
    def process_requests_batch(self, requests):
        """Process batch of requests with optimized logging."""
        
        # Log batch start (minimal context)
        self.batch_logger.info("Batch processing started", count=len(requests))
        
        processed_count = 0
        error_count = 0
        
        for request in requests:
            try:
                self._process_single_request(request)
                processed_count += 1
                
                # Only log every 1000 requests to reduce overhead
                if processed_count % 1000 == 0:
                    self.batch_logger.info("Progress update",
                        processed=processed_count,
                        total=len(requests)
                    )
                    
            except Exception as e:
                error_count += 1
                
                # Log errors but don't include expensive context
                self.error_logger.error("Request processing failed",
                    error_type=type(e).__name__,
                    request_id=request.get("id", "unknown")
                )
        
        # Log batch completion
        self.batch_logger.info("Batch processing completed",
            total_requests=len(requests),
            processed_successfully=processed_count,
            errors=error_count,
            success_rate=f"{processed_count/len(requests)*100:.1f}%"
        )
    
    def _process_single_request(self, request):
        """Process single request - minimal logging for performance."""
        
        # Only log for debug level or errors
        if self.logger._level <= logging.DEBUG:
            self.request_logger.debug("Processing request", 
                request_id=request.get("id")
            )
        
        # Simulate processing
        if request.get("invalid"):
            raise ValueError("Invalid request")
```

### 2. Memory-Efficient Context Management

```python
class MemoryEfficientService:
    """Service optimized for memory-efficient logging."""
    
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.logger.set_context(service="memory_efficient")
        
        # Cache for reusable context objects
        self._context_cache = {}
        self._bound_logger_cache = {}
    
    def process_with_cached_context(self, operation_type, data):
        """Process with cached context to reduce memory allocation."""
        
        # Get or create cached bound logger
        if operation_type not in self._bound_logger_cache:
            self._bound_logger_cache[operation_type] = self.logger.bind(
                operation_type=operation_type
            )
        
        operation_logger = self._bound_logger_cache[operation_type]
        
        # Use lightweight logging for high-frequency operations
        operation_logger.info("Operation started", data_size=len(str(data)))
        
        try:
            result = self._process_data(data, operation_logger)
            operation_logger.info("Operation completed", result_size=len(str(result)))
            return result
            
        except Exception as e:
            operation_logger.error("Operation failed", error_type=type(e).__name__)
            raise
    
    def _process_data(self, data, logger):
        """Process data with minimal logging overhead."""
        
        # Only create expensive context for errors
        try:
            # Simulate processing
            return {"processed": True, "data": data}
            
        except Exception as e:
            # Only now create expensive debugging context
            debug_context = {
                "data_type": type(data).__name__,
                "data_keys": list(data.keys()) if isinstance(data, dict) else [],
                "processing_stage": "data_transformation"
            }
            
            logger.error("Data processing failed", error=e, **debug_context)
            raise
```

---

## Security and Data Sanitization

### 1. Custom Sensitive Data Management

```python
class SecureService:
    """Service with enhanced sensitive data handling."""
    
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.logger.set_context(service="secure_service")
        
        # Add custom sensitive keys for this domain
        self.logger.add_sensitive_key("social_security_number")
        self.logger.add_sensitive_key("credit_card_number")
        self.logger.add_sensitive_key("bank_account_number")
        self.logger.add_sensitive_key("tax_id")
        self.logger.add_sensitive_key("employee_id")
        self.logger.add_sensitive_key("customer_secret")
    
    def process_user_registration(self, registration_data):
        """Process user registration with secure logging."""
        
        # Create sanitized version for logging
        safe_registration_data = self.logger.sanitize_sensitive_data(registration_data)
        
        registration_logger = self.logger.bind(
            operation="user_registration",
            registration_type=registration_data.get("type", "standard")
        )
        
        registration_logger.info("User registration started",
            registration_data=safe_registration_data,  # Automatically sanitized
            data_fields=list(registration_data.keys())
        )
        
        try:
            # Validate sensitive data (don't log the actual values)
            if not self._validate_ssn(registration_data.get("social_security_number")):
                registration_logger.error("SSN validation failed",
                    field="social_security_number",
                    validation_rule="ssn_format",
                    # Note: actual SSN value is not logged
                )
                raise ValidationError("Invalid SSN format")
            
            # Process credit card (if provided)
            if registration_data.get("credit_card_number"):
                card_result = self._process_credit_card(
                    registration_data["credit_card_number"],
                    registration_logger
                )
                registration_logger.info("Credit card processing completed",
                    card_type=card_result.get("card_type"),
                    card_valid=card_result.get("valid"),
                    # Note: card number is not logged
                )
            
            # Create user account
            user_id = self._create_user_account(registration_data, registration_logger)
            
            registration_logger.info("User registration completed successfully",
                user_id=user_id,
                account_type=registration_data.get("type"),
                verification_required=True
            )
            
            return {"user_id": user_id, "status": "created"}
            
        except Exception as e:
            registration_logger.error("User registration failed",
                error=e,
                # Safe context without sensitive data
                registration_step=self._determine_failed_step(e),
                data_provided=list(registration_data.keys())
            )
            raise
    
    def _validate_ssn(self, ssn):
        """Validate SSN format (never log actual value)."""
        return ssn and len(ssn) == 11 and ssn[3] == '-' and ssn[6] == '-'
    
    def _process_credit_card(self, card_number, logger):
        """Process credit card with secure logging."""
        
        # Log processing without the actual card number
        logger.info("Credit card processing started",
            card_length=len(card_number) if card_number else 0,
            card_type=self._detect_card_type(card_number)
        )
        
        try:
            # Simulate card processing
            result = {
                "valid": True,
                "card_type": self._detect_card_type(card_number),
                "last_four": card_number[-4:] if card_number else "****"
            }
            
            logger.info("Credit card processing completed",
                card_valid=result["valid"],
                card_type=result["card_type"],
                last_four_digits=result["last_four"]  # Only log last 4 digits
            )
            
            return result
            
        except Exception as e:
            logger.error("Credit card processing failed",
                error=e,
                card_type=self._detect_card_type(card_number),
                # Never log the actual card number
            )
            raise
    
    def _detect_card_type(self, card_number):
        """Detect credit card type from number."""
        if not card_number:
            return "unknown"
        
        if card_number.startswith("4"):
            return "visa"
        elif card_number.startswith("5"):
            return "mastercard"
        elif card_number.startswith("3"):
            return "amex"
        else:
            return "other"
    
    def _create_user_account(self, registration_data, logger):
        """Create user account with secure logging."""
        
        # Extract non-sensitive data for logging
        safe_data = {
            "email": registration_data.get("email"),
            "first_name": registration_data.get("first_name"),
            "last_name": registration_data.get("last_name"),
            "account_type": registration_data.get("type", "standard")
        }
        
        logger.info("User account creation started", user_data=safe_data)
        
        try:
            # Simulate account creation
            user_id = f"user_{int(time.time())}"
            
            logger.info("User account created successfully",
                user_id=user_id,
                email=registration_data.get("email"),
                account_type=safe_data["account_type"]
            )
            
            return user_id
            
        except Exception as e:
            logger.error("User account creation failed", error=e)
            raise
    
    def _determine_failed_step(self, error):
        """Determine which step failed based on error."""
        error_str = str(error).lower()
        if "ssn" in error_str:
            return "ssn_validation"
        elif "credit" in error_str or "card" in error_str:
            return "credit_card_processing"
        elif "account" in error_str:
            return "account_creation"
        else:
            return "unknown"

# Usage example with automatic sanitization
secure_service = SecureService()

user_data = {
    "email": "user@example.com",
    "password": "secret_password",           # Will be [REDACTED]
    "social_security_number": "123-45-6789", # Will be [REDACTED]
    "credit_card_number": "4111111111111111", # Will be [REDACTED]
    "first_name": "John",
    "last_name": "Doe"
}

# All sensitive data is automatically sanitized in logs
result = secure_service.process_user_registration(user_data)
```

### 2. Audit Trail Integration

```python
class AuditableService:
    """Service with audit trail logging."""
    
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self.logger.set_context(
            service="auditable_service",
            audit_enabled=True
        )
        
        # Create dedicated audit logger
        self.audit_logger = self.logger.bind(
            log_type="audit",
            compliance_required=True
        )
    
    def perform_sensitive_operation(self, user_id, operation_type, operation_data):
        """Perform operation with full audit trail."""
        
        # Create operation-specific audit logger
        operation_audit_logger = self.audit_logger.bind(
            user_id=user_id,
            operation_type=operation_type,
            operation_timestamp=time.time()
        )
        
        # Log audit trail start
        operation_audit_logger.info("AUDIT: Sensitive operation initiated",
            user_id=user_id,
            operation_type=operation_type,
            data_classification="sensitive",
            requires_approval=True,
            audit_trail_id=f"audit_{uuid4().hex[:16]}"
        )
        
        op_id = operation_audit_logger.start_operation("sensitive_operation")
        
        try:
            # Pre-operation audit
            operation_audit_logger.info("AUDIT: Pre-operation validation",
                operation_authorized=self._check_authorization(user_id, operation_type),
                security_context=self._get_security_context(user_id),
                operation_risk_level=self._assess_risk_level(operation_type)
            )
            
            # Perform operation
            result = self._execute_sensitive_operation(operation_type, operation_data, operation_audit_logger)
            
            # Post-operation audit
            operation_audit_logger.complete_operation(op_id, success=True,
                result_classification="sensitive",
                data_modified=True,
                compliance_status="compliant"
            )
            
            operation_audit_logger.info("AUDIT: Sensitive operation completed",
                operation_result="success",
                data_integrity_verified=True,
                compliance_check_passed=True,
                audit_trail_complete=True
            )
            
            return result
            
        except Exception as e:
            operation_audit_logger.complete_operation(op_id, success=False,
                error_type=type(e).__name__,
                security_incident=True,
                requires_investigation=True
            )
            
            operation_audit_logger.error("AUDIT: Sensitive operation failed",
                error=e,
                operation_result="failure",
                security_impact="potential_breach",
                immediate_action_required=True,
                incident_id=f"incident_{uuid4().hex[:16]}"
            )
            raise
    
    def _check_authorization(self, user_id, operation_type):
        """Check if user is authorized for operation."""
        # Simulate authorization check
        return True
    
    def _get_security_context(self, user_id):
        """Get security context for user."""
        return {
            "user_role": "admin",
            "session_valid": True,
            "mfa_verified": True
        }
    
    def _assess_risk_level(self, operation_type):
        """Assess risk level of operation."""
        high_risk_operations = ["delete_user", "modify_permissions", "export_data"]
        return "high" if operation_type in high_risk_operations else "medium"
    
    def _execute_sensitive_operation(self, operation_type, operation_data, audit_logger):
        """Execute the actual sensitive operation."""
        
        audit_logger.info("AUDIT: Operation execution started",
            operation_stage="execution",
            data_access_logged=True
        )
        
        # Simulate operation
        time.sleep(0.1)
        
        audit_logger.info("AUDIT: Operation execution completed",
            operation_stage="completed",
            data_changes_applied=True
        )
        
        return {"status": "completed", "operation_id": f"op_{int(time.time())}"}
```

---

## Testing and Validation

### 1. Unit Testing FlextLogger Integration

```python
import pytest
from unittest.mock import patch, MagicMock
from flext_core import FlextLogger
import json

class TestFlextLoggerIntegration:
    """Test FlextLogger integration patterns."""
    
    def setup_method(self):
        """Setup for each test."""
        FlextLogger.clear_metrics()
        FlextLogger.set_global_correlation_id(None)
    
    def test_basic_logging_with_context(self):
        """Test basic logging includes context."""
        logger = FlextLogger("test_service")
        logger.set_context(service="test", version="1.0")
        
        # Capture log output
        with patch('structlog.get_logger') as mock_structlog:
            mock_bound_logger = MagicMock()
            mock_structlog.return_value = mock_bound_logger
            
            logger.info("Test message", user_id=123)
            
            # Verify structured log was called with correct data
            mock_bound_logger.info.assert_called_once()
            args, kwargs = mock_bound_logger.info.call_args
            
            assert "Test message" in args[0]
            assert "correlation_id" in kwargs
            assert "service" in kwargs
    
    def test_correlation_id_propagation(self):
        """Test correlation ID propagation."""
        test_correlation_id = "test_corr_123"
        
        # Set global correlation ID
        FlextLogger.set_global_correlation_id(test_correlation_id)
        
        logger1 = FlextLogger("service1")
        logger2 = FlextLogger("service2")
        
        # Both loggers should have the same correlation ID
        assert logger1.get_correlation_id() == test_correlation_id
        assert logger2.get_correlation_id() == test_correlation_id
        
        # Instance-level override
        logger1.set_correlation_id("specific_123")
        assert logger1.get_correlation_id() == "specific_123"
        assert logger2.get_correlation_id() == test_correlation_id  # Unchanged
    
    def test_operation_tracking(self):
        """Test operation performance tracking."""
        logger = FlextLogger("test_service")
        
        with patch('structlog.get_logger') as mock_structlog:
            mock_bound_logger = MagicMock()
            mock_structlog.return_value = mock_bound_logger
            
            # Start operation
            op_id = logger.start_operation("test_operation", user_id=123)
            assert op_id.startswith("op_")
            
            # Complete operation
            logger.complete_operation(op_id, success=True, result_count=5)
            
            # Verify both start and complete calls
            assert mock_bound_logger.info.call_count == 2
    
    def test_context_binding(self):
        """Test logger context binding."""
        logger = FlextLogger("test_service")
        
        # Create bound logger
        bound_logger = logger.bind(user_id=123, operation="test")
        
        # Bound logger should be different instance
        assert bound_logger is not logger
        
        # Both should have the same name but different context
        assert bound_logger._name == logger._name
    
    def test_request_context_thread_local(self):
        """Test thread-local request context."""
        logger = FlextLogger("test_service")
        
        # Set request context
        logger.set_request_context(request_id="req_123", user_id=456)
        
        # Context should be available
        assert hasattr(logger._local, 'request_context')
        assert logger._local.request_context["request_id"] == "req_123"
        
        # Clear context
        logger.clear_request_context()
        
        # Context should be cleared
        if hasattr(logger._local, 'request_context'):
            assert len(logger._local.request_context) == 0
    
    def test_sensitive_data_sanitization(self):
        """Test automatic sensitive data sanitization."""
        logger = FlextLogger("test_service")
        
        sensitive_data = {
            "username": "john_doe",
            "password": "secret123",
            "email": "john@example.com",
            "api_key": "sk_live_123456",
            "token": "jwt_token_abc"
        }
        
        sanitized = logger.sanitize_sensitive_data(sensitive_data)
        
        # Non-sensitive data preserved
        assert sanitized["username"] == "john_doe"
        assert sanitized["email"] == "john@example.com"
        
        # Sensitive data redacted
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
    
    def test_nested_sensitive_data_sanitization(self):
        """Test sanitization of nested dictionaries."""
        logger = FlextLogger("test_service")
        
        nested_data = {
            "user": {
                "name": "John Doe",
                "credentials": {
                    "password": "secret",
                    "api_key": "key_123"
                }
            },
            "session": {
                "id": "session_456",
                "token": "bearer_token"
            }
        }
        
        sanitized = nested_data.copy()
        sanitized = logger.sanitize_sensitive_data(sanitized)
        
        # Check nested sanitization
        assert sanitized["user"]["name"] == "John Doe"  # Preserved
        assert sanitized["user"]["credentials"]["password"] == "[REDACTED]"
        assert sanitized["user"]["credentials"]["api_key"] == "[REDACTED]"
        assert sanitized["session"]["id"] == "[REDACTED]"  # session_id is sensitive
        assert sanitized["session"]["token"] == "[REDACTED]"
    
    def test_error_logging_with_exception(self):
        """Test error logging with Exception objects."""
        logger = FlextLogger("test_service")
        
        with patch('structlog.get_logger') as mock_structlog:
            mock_bound_logger = MagicMock()
            mock_structlog.return_value = mock_bound_logger
            
            # Test with Exception object
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.error("Operation failed", error=e, user_id=123)
            
            # Verify error details were captured
            mock_bound_logger.error.assert_called_once()
            args, kwargs = mock_bound_logger.error.call_args
            
            assert "error" in kwargs
            assert "user_id" in kwargs
    
    def test_error_logging_with_string(self):
        """Test error logging with string messages."""
        logger = FlextLogger("test_service")
        
        with patch('structlog.get_logger') as mock_structlog:
            mock_bound_logger = MagicMock()
            mock_structlog.return_value = mock_bound_logger
            
            # Test with string error
            logger.error("Validation failed", error="Email is invalid", field="email")
            
            mock_bound_logger.error.assert_called_once()
            args, kwargs = mock_bound_logger.error.call_args
            
            assert "error" in kwargs
            assert "field" in kwargs

class TestFlextLoggerConfiguration:
    """Test FlextLogger configuration patterns."""
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        
        # Test production config
        prod_config = FlextLogger.create_environment_logging_config("production")
        assert prod_config.success
        
        config = prod_config.value
        assert config["log_level"] == "WARNING"
        assert config["enable_json_logging"] is True
        assert config["enable_console_output"] is False
        
        # Test development config
        dev_config = FlextLogger.create_environment_logging_config("development")
        assert dev_config.success
        
        config = dev_config.value
        assert config["log_level"] == "DEBUG"
        assert config["enable_json_logging"] is False
        assert config["enable_console_output"] is True
    
    def test_logging_system_configuration(self):
        """Test logging system configuration."""
        
        config = {
            "environment": "test",
            "log_level": "INFO",
            "enable_correlation_tracking": True,
            "enable_performance_logging": False
        }
        
        result = FlextLogger.configure_logging_system(config)
        assert result.success
        
        validated_config = result.value
        assert validated_config["environment"] == "test"
        assert validated_config["log_level"] == "INFO"
        assert validated_config["enable_correlation_tracking"] is True
    
    def test_performance_optimization_config(self):
        """Test performance optimization configuration."""
        
        config = {
            "performance_level": "high",
            "async_logging_enabled": True,
            "buffer_size": 5000
        }
        
        result = FlextLogger.optimize_logging_performance(config)
        assert result.success
        
        optimized_config = result.value
        assert optimized_config["async_logging_enabled"] is True
        assert optimized_config["buffer_size"] == 5000
        assert optimized_config["batch_log_processing"] is True  # Auto-enabled for high perf

class TestFlextLoggerIntegrationPatterns:
    """Test real-world integration patterns."""
    
    def test_api_service_integration(self):
        """Test API service integration pattern."""
        
        class MockAPIService:
            def __init__(self):
                self.logger = FlextLogger(__name__)
                self.logger.set_context(service="api", version="1.0")
            
            def handle_request(self, request_data):
                self.logger.set_request_context(**request_data.get("context", {}))
                
                op_id = self.logger.start_operation("api_request")
                
                try:
                    self.logger.info("Processing request")
                    result = {"status": "success"}
                    self.logger.complete_operation(op_id, success=True)
                    return result
                finally:
                    self.logger.clear_request_context()
        
        service = MockAPIService()
        request = {
            "data": {"user_id": 123},
            "context": {"request_id": "req_123", "method": "POST"}
        }
        
        result = service.handle_request(request)
        assert result["status"] == "success"
    
    def test_database_service_integration(self):
        """Test database service integration pattern."""
        
        class MockDatabaseService:
            def __init__(self):
                self.logger = FlextLogger(__name__)
                self.logger.set_context(component="database")
            
            def execute_query(self, sql):
                query_logger = self.logger.bind(query_type="select", sql_length=len(sql))
                
                with query_logger.track_duration("query_execution") as tracker:
                    # Simulate query execution
                    result = [{"id": 1, "name": "test"}]
                    
                    tracker.add_context(
                        rows_returned=len(result),
                        execution_plan="index_scan"
                    )
                    
                    return result
        
        service = MockDatabaseService()
        result = service.execute_query("SELECT * FROM users")
        assert len(result) == 1
```

---

## Best Practices Summary

### 1. Logger Creation and Management
- Use singleton pattern: `FlextLogger(__name__)` reuses instances by name
- Create bound loggers for operation-specific context: `logger.bind(user_id="123")`
- Set permanent context once: `logger.set_context(service="api", version="1.0")`
- Use request context for thread-local data: `logger.set_request_context(request_id="req_123")`

### 2. Correlation ID Management
- Set global correlation ID at request boundary: `FlextLogger.set_global_correlation_id("req_123")`
- Propagate correlation IDs in HTTP headers: `"X-Correlation-ID": correlation_id`
- Use instance-level correlation for specific operations: `logger.set_correlation_id("op_456")`
- Clear request context when request completes: `logger.clear_request_context()`

### 3. Performance Optimization
- Configure async logging for high-throughput: `"async_logging_enabled": True`
- Use bound loggers for repeated context: `user_logger = logger.bind(user_id="123")`
- Cache expensive context creation: `if logger._level <= logging.INFO: create_context()`
- Limit log message size in production: `"max_log_message_size": 5000`

### 4. Security Best Practices
- Sensitive data is automatically redacted: `password`, `token`, `api_key`
- Add domain-specific sensitive keys: `logger.add_sensitive_key("ssn")`
- Use sanitized context for logging: `logger.sanitize_sensitive_data(data)`
- Never log credentials or personal information directly

### 5. Error Handling
- Use Exception objects for full context: `logger.error("Failed", error=e)`
- Include structured context: `logger.error("Failed", error=e, user_id=123, step="validation")`
- Use appropriate log levels: `error` for failures, `critical` for system issues
- Implement audit trails for sensitive operations

### 6. Environment Configuration
- Use environment-specific configurations: `create_environment_logging_config(env)`
- Production: JSON output, WARNING level, no console, async enabled
- Development: Console output, DEBUG level, colored output, sync logging
- Test: Minimal output, ERROR level, no correlation, sync logging

---

This implementation guide provides comprehensive patterns for integrating FlextLogger into FLEXT ecosystem projects, ensuring consistent, observable, and secure logging across all applications and services.
