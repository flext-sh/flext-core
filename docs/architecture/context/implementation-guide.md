# FlextContext Implementation Guide

**Version**: 0.9.0
**Target Audience**: FLEXT Developers, System Architects, DevOps Engineers
**Implementation Time**: 1-3 weeks per service
**Complexity**: Beginner to Advanced

## 📖 Overview

This guide provides comprehensive, step-by-step instructions for implementing `FlextContext` distributed tracing and context management in FLEXT services. The context system enables enterprise-grade observability, cross-service correlation, and comprehensive performance monitoring.

### Prerequisites

- Python 3.13+ with contextvars support
- Understanding of distributed systems concepts
- Familiarity with microservices architecture
- Knowledge of observability patterns

### Implementation Benefits

- 📊 **100% request traceability** across service boundaries
- 🔗 **Automatic correlation ID propagation** with parent-child relationships
- ⏱️ **Built-in performance monitoring** with operation timing
- 🔧 **Thread-safe context management** using Python contextvars
- 🌐 **Cross-service communication** with context serialization

---

## 🚀 Quick Start

### Basic Context Usage

```python
from flext_core.context import FlextContext

# Basic correlation tracking
with FlextContext.Correlation.new_correlation() as correlation_id:
    with FlextContext.Service.service_context("user-service", "v1.0.0"):
        with FlextContext.Performance.timed_operation("user_lookup"):
            # All context automatically managed
            user = lookup_user(user_id)

            # Context automatically includes:
            # - Correlation ID: correlation_id
            # - Service: user-service v1.0.0
            # - Performance: timing and metadata

            return user
```

### Context Propagation

```python
# Export context for HTTP calls
headers = FlextContext.Serialization.get_correlation_context()
response = httpx.get("/api/users", headers=headers)

# Import context from incoming request
FlextContext.Serialization.set_from_context(request.headers)
```

---

## 📚 Step-by-Step Implementation

### Step 1: Basic Context Setup

#### 1.1 Service Identification Setup

Every service should establish its identity:

```python
from flext_core.context import FlextContext

class UserService:
    """User service with context identification."""

    def __init__(self, version: str = "1.0.0"):
        self.service_name = "user-service"
        self.service_version = version

    def startup(self):
        """Initialize service context on startup."""
        # Set global service context
        FlextContext.Service.set_service_name(self.service_name)
        FlextContext.Service.set_service_version(self.service_version)

        logger.info(f"Service {self.service_name} v{self.service_version} started")

    def handle_request(self, request_data: dict):
        """Handle request with service context."""
        with FlextContext.Service.service_context(self.service_name, self.service_version):
            # Service context automatically available
            service = FlextContext.Service.get_service_name()
            version = FlextContext.Service.get_service_version()

            logger.info(f"Processing request in {service} v{version}")
            return self.process_request(request_data)
```

#### 1.2 Correlation ID Management

Implement distributed tracing with correlation IDs:

```python
class DistributedTracingHandler:
    """Handler with distributed tracing support."""

    def handle_incoming_request(self, request):
        """Handle incoming request with correlation tracking."""

        # Set context from incoming headers
        FlextContext.Serialization.set_from_context(request.headers)

        # Create or inherit correlation
        with FlextContext.Correlation.inherit_correlation() as correlation_id:
            logger.info(f"Processing request with correlation: {correlation_id}")

            # Process request with correlation context
            result = self.process_request_logic(request)

            # Correlation automatically available to downstream calls
            return result

    def call_downstream_service(self, service_url: str, data: dict):
        """Call downstream service with correlation propagation."""

        # Get correlation context for headers
        correlation_headers = FlextContext.Serialization.get_correlation_context()

        # Add any custom headers
        headers = {
            **correlation_headers,
            "Content-Type": "application/json",
            "User-Agent": "user-service/1.0.0"
        }

        # Call downstream with context
        response = httpx.post(service_url, json=data, headers=headers)

        return response

    def create_child_operation(self):
        """Create child operation with nested correlation."""

        # Nested correlation maintains parent relationship
        with FlextContext.Correlation.new_correlation() as child_correlation:
            parent_correlation = FlextContext.Correlation.get_parent_correlation_id()

            logger.info(f"Child operation: {child_correlation}, Parent: {parent_correlation}")

            # Perform child operation
            return self.perform_child_operation()
```

### Step 2: Request Context Management

#### 2.1 Request-Level Context Setup

Implement comprehensive request context tracking:

```python
class RequestContextHandler:
    """Request handler with comprehensive context management."""

    async def handle_api_request(self, request):
        """Handle API request with full context setup."""

        # Extract user information
        user_id = await self.extract_user_id(request)
        request_id = request.headers.get("X-Request-Id") or self.generate_request_id()

        # Set up comprehensive request context
        with FlextContext.Request.request_context(
            user_id=user_id,
            operation_name=f"{request.method}_{request.url.path}",
            request_id=request_id,
            metadata={
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "content_type": request.headers.get("content-type", ""),
                "content_length": request.headers.get("content-length", "0")
            }
        ):
            # All request context available throughout processing
            current_user = FlextContext.Request.get_user_id()
            operation = FlextContext.Request.get_operation_name()

            logger.info(f"Processing {operation} for user {current_user}")

            # Process request with full context
            return await self.process_api_request(request)

    async def extract_user_id(self, request):
        """Extract user ID from request (JWT, session, etc.)."""
        # Implementation depends on authentication method
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return await self.decode_jwt_user_id(token)

        # Check session cookies
        session_id = request.cookies.get("session_id")
        if session_id:
            return await self.get_user_from_session(session_id)

        return None

    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        from flext_core.utilities import FlextUtilities
        return FlextUtilities.Generators.generate_request_id()
```

#### 2.2 Advanced Request Context Patterns

```python
class AdvancedRequestContextHandler:
    """Advanced request context patterns for complex scenarios."""

    async def handle_batch_request(self, batch_request):
        """Handle batch request with individual item context."""

        # Main batch context
        with FlextContext.Request.request_context(
            operation_name="batch_processing",
            request_id=batch_request.get("batch_id"),
            metadata={
                "batch_size": len(batch_request.get("items", [])),
                "batch_type": batch_request.get("type", "unknown")
            }
        ):
            results = []

            # Process each item with its own context
            for i, item in enumerate(batch_request.get("items", [])):
                with FlextContext.Request.request_context(
                    operation_name=f"batch_item_{i}",
                    metadata={
                        "item_index": i,
                        "item_id": item.get("id"),
                        "item_type": item.get("type")
                    }
                ):
                    # Each item has its own context while maintaining batch context
                    item_result = await self.process_batch_item(item)
                    results.append(item_result)

            return {"batch_results": results}

    async def handle_workflow_request(self, workflow_request):
        """Handle multi-step workflow with step context tracking."""

        workflow_id = workflow_request.get("workflow_id")

        with FlextContext.Request.request_context(
            operation_name="workflow_execution",
            request_id=workflow_id,
            metadata={
                "workflow_type": workflow_request.get("type"),
                "step_count": len(workflow_request.get("steps", []))
            }
        ):
            workflow_results = []

            # Execute each workflow step with context
            for step_index, step in enumerate(workflow_request.get("steps", [])):
                step_result = await self.execute_workflow_step(
                    step, step_index, workflow_id
                )
                workflow_results.append(step_result)

                # Break on step failure
                if not step_result.get("success", False):
                    break

            return {"workflow_id": workflow_id, "step_results": workflow_results}

    async def execute_workflow_step(self, step: dict, step_index: int, workflow_id: str):
        """Execute individual workflow step with context."""

        with FlextContext.Request.request_context(
            operation_name=f"workflow_step_{step.get('name', step_index)}",
            metadata={
                "step_index": step_index,
                "step_name": step.get("name"),
                "step_type": step.get("type"),
                "workflow_id": workflow_id
            }
        ):
            with FlextContext.Performance.timed_operation(f"step_{step_index}") as perf:
                try:
                    # Execute step logic
                    result = await self.perform_step_operation(step)

                    # Add step success metadata
                    FlextContext.Performance.add_operation_metadata("step_success", True)
                    FlextContext.Performance.add_operation_metadata("result_size",
                                                                   len(str(result)))

                    return {
                        "step_index": step_index,
                        "success": True,
                        "result": result,
                        "duration": perf.get("duration_seconds", 0)
                    }

                except Exception as e:
                    # Step failure with context
                    FlextContext.Performance.add_operation_metadata("step_error", str(e))

                    logger.error(f"Workflow step {step_index} failed: {e}")

                    return {
                        "step_index": step_index,
                        "success": False,
                        "error": str(e),
                        "duration": perf.get("duration_seconds", 0)
                    }
```

### Step 3: Performance Monitoring Integration

#### 3.1 Operation Timing and Metrics

Implement comprehensive performance monitoring:

```python
class PerformanceMonitoringService:
    """Service with comprehensive performance monitoring."""

    def process_data_with_monitoring(self, data: dict):
        """Process data with detailed performance monitoring."""

        with FlextContext.Performance.timed_operation("data_processing") as perf:
            try:
                # Add initial metadata
                FlextContext.Performance.add_operation_metadata("input_size", len(data))
                FlextContext.Performance.add_operation_metadata("data_type",
                                                               data.get("type", "unknown"))

                # Phase 1: Data validation
                with FlextContext.Performance.timed_operation("validation") as validation_perf:
                    validation_result = self.validate_data(data)
                    FlextContext.Performance.add_operation_metadata("validation_passed",
                                                                   validation_result.success)

                    if not validation_result.success:
                        FlextContext.Performance.add_operation_metadata("validation_errors",
                                                                       validation_result.errors)
                        return validation_result

                # Phase 2: Data transformation
                with FlextContext.Performance.timed_operation("transformation") as transform_perf:
                    transformed_data = self.transform_data(data)
                    FlextContext.Performance.add_operation_metadata("transformation_output_size",
                                                                   len(transformed_data))

                # Phase 3: Data persistence
                with FlextContext.Performance.timed_operation("persistence") as persistence_perf:
                    persistence_result = self.persist_data(transformed_data)
                    FlextContext.Performance.add_operation_metadata("records_persisted",
                                                                   persistence_result.record_count)

                # Aggregate performance metrics
                total_duration = perf.get("duration_seconds", 0)
                FlextContext.Performance.add_operation_metadata("total_duration", total_duration)
                FlextContext.Performance.add_operation_metadata("throughput_rps",
                                                               len(data) / max(total_duration, 0.001))

                # Performance summary
                performance_summary = {
                    "total_duration": total_duration,
                    "validation_duration": validation_perf.get("duration_seconds", 0),
                    "transformation_duration": transform_perf.get("duration_seconds", 0),
                    "persistence_duration": persistence_perf.get("duration_seconds", 0),
                    "records_processed": len(data),
                    "throughput_rps": len(data) / max(total_duration, 0.001)
                }

                logger.info("Data processing completed", extra={"performance": performance_summary})

                return FlextResult.ok(persistence_result)

            except Exception as e:
                # Error tracking with performance context
                error_duration = perf.get("duration_seconds", 0)
                FlextContext.Performance.add_operation_metadata("error", str(e))
                FlextContext.Performance.add_operation_metadata("error_duration", error_duration)

                logger.error(f"Data processing failed after {error_duration}s: {e}")

                return FlextResult.fail(f"Processing failed: {e}")

    async def monitor_external_service_call(self, service_url: str, request_data: dict):
        """Monitor external service call with detailed metrics."""

        with FlextContext.Performance.timed_operation("external_service_call") as perf:
            try:
                # Add request metadata
                FlextContext.Performance.add_operation_metadata("service_url", service_url)
                FlextContext.Performance.add_operation_metadata("request_size",
                                                               len(str(request_data)))

                # Get correlation context for headers
                headers = FlextContext.Serialization.get_correlation_context()
                headers.update({
                    "Content-Type": "application/json",
                    "X-Client-Service": FlextContext.Service.get_service_name()
                })

                # Make service call with timing
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        service_url,
                        json=request_data,
                        headers=headers,
                        timeout=30.0
                    )

                # Add response metadata
                FlextContext.Performance.add_operation_metadata("response_status",
                                                               response.status_code)
                FlextContext.Performance.add_operation_metadata("response_size",
                                                               len(response.content))
                FlextContext.Performance.add_operation_metadata("response_time",
                                                               response.elapsed.total_seconds())

                # Success metrics
                duration = perf.get("duration_seconds", 0)
                FlextContext.Performance.add_operation_metadata("call_success", True)
                FlextContext.Performance.add_operation_metadata("total_latency", duration)

                logger.info(f"External service call completed in {duration}s",
                           extra={"service_url": service_url, "status": response.status_code})

                return response

            except httpx.TimeoutException as e:
                # Timeout tracking
                FlextContext.Performance.add_operation_metadata("timeout_error", True)
                FlextContext.Performance.add_operation_metadata("timeout_duration", 30.0)

                logger.error(f"External service call timed out: {service_url}")
                raise

            except Exception as e:
                # General error tracking
                FlextContext.Performance.add_operation_metadata("call_error", str(e))
                FlextContext.Performance.add_operation_metadata("error_type", type(e).__name__)

                logger.error(f"External service call failed: {service_url}, error: {e}")
                raise
```

### Step 4: Cross-Service Communication

#### 4.1 HTTP Service Integration

Implement context-aware HTTP service communication:

```python
class ContextAwareHttpClient:
    """HTTP client with automatic context propagation."""

    def __init__(self, base_url: str, service_name: str):
        self.base_url = base_url
        self.service_name = service_name
        self.client = httpx.AsyncClient(base_url=base_url)

    async def call_service_with_context(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        extra_headers: dict = None
    ):
        """Make service call with full context propagation."""

        with FlextContext.Performance.timed_operation(f"{method.lower()}_{endpoint}") as perf:
            try:
                # Build headers with context
                headers = FlextContext.Serialization.get_correlation_context()

                # Add service identification
                headers.update({
                    "X-Client-Service": FlextContext.Service.get_service_name(),
                    "X-Client-Version": FlextContext.Service.get_service_version(),
                    "X-Operation-Name": FlextContext.Request.get_operation_name()
                })

                # Add extra headers
                if extra_headers:
                    headers.update(extra_headers)

                # Add request metadata
                FlextContext.Performance.add_operation_metadata("target_service", self.service_name)
                FlextContext.Performance.add_operation_metadata("endpoint", endpoint)
                FlextContext.Performance.add_operation_metadata("method", method.upper())

                if data:
                    FlextContext.Performance.add_operation_metadata("request_size", len(str(data)))

                # Make the request
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=data if data else None,
                    headers=headers
                )

                # Add response metadata
                FlextContext.Performance.add_operation_metadata("response_status",
                                                               response.status_code)
                FlextContext.Performance.add_operation_metadata("response_size",
                                                               len(response.content))

                # Log successful call
                duration = perf.get("duration_seconds", 0)
                context_summary = FlextContext.Utilities.get_context_summary()

                logger.info(f"Service call completed in {duration}s: {method} {endpoint}",
                           extra={"context": context_summary, "status": response.status_code})

                return response

            except Exception as e:
                # Error handling with context
                FlextContext.Performance.add_operation_metadata("call_error", str(e))
                FlextContext.Performance.add_operation_metadata("error_type", type(e).__name__)

                error_context = FlextContext.Serialization.get_full_context()
                logger.error(f"Service call failed: {method} {endpoint}, error: {e}",
                           extra={"context": error_context})

                raise

    async def get_with_context(self, endpoint: str, **kwargs):
        """GET request with context."""
        return await self.call_service_with_context("GET", endpoint, **kwargs)

    async def post_with_context(self, endpoint: str, data: dict, **kwargs):
        """POST request with context."""
        return await self.call_service_with_context("POST", endpoint, data, **kwargs)

    async def put_with_context(self, endpoint: str, data: dict, **kwargs):
        """PUT request with context."""
        return await self.call_service_with_context("PUT", endpoint, data, **kwargs)

    async def delete_with_context(self, endpoint: str, **kwargs):
        """DELETE request with context."""
        return await self.call_service_with_context("DELETE", endpoint, **kwargs)
```

#### 4.2 Message Queue Integration

Implement context propagation through message queues:

```python
class ContextAwareMessageQueue:
    """Message queue with context propagation support."""

    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self.publisher = self._create_publisher()
        self.consumer = self._create_consumer()

    async def publish_message_with_context(self, message_data: dict, routing_key: str = None):
        """Publish message with full context propagation."""

        with FlextContext.Performance.timed_operation("message_publish") as perf:
            try:
                # Get full context for message metadata
                context_data = FlextContext.Serialization.get_full_context()

                # Build message with context
                message = {
                    "data": message_data,
                    "context": context_data,
                    "metadata": {
                        "correlation_id": FlextContext.Correlation.get_correlation_id(),
                        "service_name": FlextContext.Service.get_service_name(),
                        "service_version": FlextContext.Service.get_service_version(),
                        "user_id": FlextContext.Request.get_user_id(),
                        "operation_name": FlextContext.Request.get_operation_name(),
                        "timestamp": datetime.utcnow().isoformat(),
                        "publisher": FlextContext.Service.get_service_name()
                    }
                }

                # Add message metadata to performance tracking
                FlextContext.Performance.add_operation_metadata("message_size", len(str(message)))
                FlextContext.Performance.add_operation_metadata("queue_name", self.queue_name)
                FlextContext.Performance.add_operation_metadata("routing_key", routing_key)

                # Publish message
                await self.publisher.publish(
                    message=message,
                    routing_key=routing_key or self.queue_name
                )

                # Log successful publish
                duration = perf.get("duration_seconds", 0)
                correlation_id = FlextContext.Correlation.get_correlation_id()

                logger.info(f"Message published in {duration}s",
                           extra={
                               "correlation_id": correlation_id,
                               "queue_name": self.queue_name,
                               "message_size": len(str(message))
                           })

            except Exception as e:
                # Error handling
                FlextContext.Performance.add_operation_metadata("publish_error", str(e))

                logger.error(f"Message publish failed: {e}",
                           extra={"queue_name": self.queue_name})

                raise

    async def consume_message_with_context(self, message):
        """Consume message and restore context."""

        try:
            # Extract context from message
            message_context = message.get("context", {})
            message_metadata = message.get("metadata", {})
            message_data = message.get("data", {})

            # Restore context from message
            if message_context:
                FlextContext.Serialization.set_from_context(message_context)

            # Set additional context from metadata
            if message_metadata.get("correlation_id"):
                FlextContext.Correlation.set_correlation_id(
                    message_metadata["correlation_id"]
                )

            # Process message with context
            with FlextContext.Performance.timed_operation("message_processing") as perf:
                # Add processing metadata
                FlextContext.Performance.add_operation_metadata("queue_name", self.queue_name)
                FlextContext.Performance.add_operation_metadata("message_size",
                                                               len(str(message_data)))
                FlextContext.Performance.add_operation_metadata("publisher",
                                                               message_metadata.get("publisher"))

                # Process the message
                result = await self.process_message(message_data)

                # Add success metadata
                FlextContext.Performance.add_operation_metadata("processing_success", True)

                duration = perf.get("duration_seconds", 0)
                correlation_id = FlextContext.Correlation.get_correlation_id()

                logger.info(f"Message processed in {duration}s",
                           extra={
                               "correlation_id": correlation_id,
                               "queue_name": self.queue_name
                           })

                return result

        except Exception as e:
            # Error handling with context
            FlextContext.Performance.add_operation_metadata("processing_error", str(e))

            error_context = FlextContext.Serialization.get_full_context()
            logger.error(f"Message processing failed: {e}",
                       extra={"context": error_context})

            raise

    async def process_message(self, message_data: dict):
        """Process message data - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_message")
```

### Step 5: Testing and Validation

#### 5.1 Context Testing Patterns

```python
import pytest
from unittest.mock import patch

class TestFlextContextIntegration:
    """Test suite for FlextContext integration."""

    def setup_method(self):
        """Setup test environment."""
        # Clear context before each test
        FlextContext.Utilities.clear_context()

    def teardown_method(self):
        """Cleanup after each test."""
        # Clear context after each test
        FlextContext.Utilities.clear_context()

    def test_correlation_id_generation(self):
        """Test correlation ID generation and propagation."""

        # Test correlation ID generation
        correlation_id = FlextContext.Correlation.generate_correlation_id()
        assert correlation_id is not None
        assert len(correlation_id) > 0

        # Test correlation ID retrieval
        retrieved_id = FlextContext.Correlation.get_correlation_id()
        assert retrieved_id == correlation_id

        # Test correlation context manager
        with FlextContext.Correlation.new_correlation() as new_id:
            assert new_id != correlation_id
            assert FlextContext.Correlation.get_correlation_id() == new_id

        # Context should be restored
        assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_nested_correlation(self):
        """Test nested correlation with parent tracking."""

        with FlextContext.Correlation.new_correlation() as parent_id:
            assert FlextContext.Correlation.get_parent_correlation_id() is None

            with FlextContext.Correlation.new_correlation() as child_id:
                # Child should have parent reference
                assert FlextContext.Correlation.get_parent_correlation_id() == parent_id
                assert FlextContext.Correlation.get_correlation_id() == child_id

                # Nested child
                with FlextContext.Correlation.new_correlation() as grandchild_id:
                    assert FlextContext.Correlation.get_parent_correlation_id() == child_id
                    assert FlextContext.Correlation.get_correlation_id() == grandchild_id

                # Back to child
                assert FlextContext.Correlation.get_correlation_id() == child_id

            # Back to parent
            assert FlextContext.Correlation.get_correlation_id() == parent_id

    def test_service_context_management(self):
        """Test service context setup and management."""

        service_name = "test-service"
        service_version = "v1.2.3"

        # Test service context manager
        with FlextContext.Service.service_context(service_name, service_version):
            assert FlextContext.Service.get_service_name() == service_name
            assert FlextContext.Service.get_service_version() == service_version

        # Context should be cleared after manager exits
        assert FlextContext.Service.get_service_name() is None
        assert FlextContext.Service.get_service_version() is None

    def test_request_context_management(self):
        """Test request context setup and management."""

        user_id = "user_123"
        operation_name = "test_operation"
        request_id = "req_456"
        metadata = {"source": "test", "version": "1.0"}

        with FlextContext.Request.request_context(
            user_id=user_id,
            operation_name=operation_name,
            request_id=request_id,
            metadata=metadata
        ):
            assert FlextContext.Request.get_user_id() == user_id
            assert FlextContext.Request.get_operation_name() == operation_name
            assert FlextContext.Request.get_request_id() == request_id

            operation_metadata = FlextContext.Performance.get_operation_metadata()
            assert operation_metadata == metadata

        # Context should be cleared
        assert FlextContext.Request.get_user_id() is None
        assert FlextContext.Request.get_operation_name() is None

    def test_performance_timing(self):
        """Test performance timing functionality."""

        import time

        with FlextContext.Performance.timed_operation("test_operation") as perf:
            # Add some metadata during operation
            FlextContext.Performance.add_operation_metadata("test_key", "test_value")

            # Simulate work
            time.sleep(0.01)  # 10ms

            # Check metadata is available
            metadata = FlextContext.Performance.get_operation_metadata()
            assert "test_key" in metadata
            assert metadata["test_key"] == "test_value"

        # Check timing was recorded
        assert "duration_seconds" in perf
        assert perf["duration_seconds"] >= 0.01
        assert "start_time" in perf
        assert "end_time" in perf

    def test_context_serialization(self):
        """Test context serialization and deserialization."""

        # Set up context
        correlation_id = "test_correlation_123"
        service_name = "test-service"
        user_id = "user_456"

        FlextContext.Correlation.set_correlation_id(correlation_id)
        FlextContext.Service.set_service_name(service_name)
        FlextContext.Request.set_user_id(user_id)

        # Test full context serialization
        full_context = FlextContext.Serialization.get_full_context()
        assert full_context["correlation_id"] == correlation_id
        assert full_context["service_name"] == service_name
        assert full_context["user_id"] == user_id

        # Test correlation context for headers
        correlation_headers = FlextContext.Serialization.get_correlation_context()
        assert "X-Correlation-Id" in correlation_headers
        assert correlation_headers["X-Correlation-Id"] == correlation_id
        assert "X-Service-Name" in correlation_headers
        assert correlation_headers["X-Service-Name"] == service_name

        # Test context restoration
        FlextContext.Utilities.clear_context()
        assert FlextContext.Correlation.get_correlation_id() is None

        FlextContext.Serialization.set_from_context(correlation_headers)
        assert FlextContext.Correlation.get_correlation_id() == correlation_id
        assert FlextContext.Service.get_service_name() == service_name

    @patch('httpx.AsyncClient')
    async def test_http_client_context_propagation(self, mock_client):
        """Test HTTP client with context propagation."""

        # Setup context
        FlextContext.Correlation.set_correlation_id("test_correlation")
        FlextContext.Service.set_service_name("test-service")

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"result": "success"}'
        mock_response.elapsed.total_seconds.return_value = 0.1

        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Create HTTP client and make request
        client = ContextAwareHttpClient("http://test-service", "test-service")

        response = await client.post_with_context("/test", {"data": "test"})

        # Verify context headers were added
        call_args = mock_client_instance.post.call_args
        headers = call_args.kwargs["headers"]

        assert "X-Correlation-Id" in headers
        assert headers["X-Correlation-Id"] == "test_correlation"
        assert "X-Client-Service" in headers
        assert headers["X-Client-Service"] == "test-service"

    def test_context_utilities(self):
        """Test context utility functions."""

        # Test ensure_correlation_id
        assert not FlextContext.Utilities.has_correlation_id()

        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        assert correlation_id is not None
        assert FlextContext.Utilities.has_correlation_id()
        assert FlextContext.Correlation.get_correlation_id() == correlation_id

        # Test context summary
        FlextContext.Service.set_service_name("test-service")
        FlextContext.Request.set_user_id("user_123")

        summary = FlextContext.Utilities.get_context_summary()
        assert "test-service" in summary
        assert "user_123" in summary

        # Test clear context
        FlextContext.Utilities.clear_context()
        assert not FlextContext.Utilities.has_correlation_id()
        assert FlextContext.Service.get_service_name() is None
        assert FlextContext.Request.get_user_id() is None
```

---

## ⚡ Performance Optimization

### Optimization Techniques

#### 1. **Context Caching**

```python
class OptimizedContextService:
    def __init__(self):
        self._context_cache = {}

    def get_cached_context(self, cache_key: str):
        """Get cached context for repeated operations."""
        if cache_key not in self._context_cache:
            self._context_cache[cache_key] = FlextContext.Serialization.get_full_context()
        return self._context_cache[cache_key]
```

#### 2. **Batch Context Operations**

```python
class BatchContextProcessor:
    async def process_batch_with_shared_context(self, items: list):
        """Process batch items with shared context setup."""

        # Set context once for the entire batch
        with FlextContext.Service.service_context("batch-processor", "v1.0.0"):
            with FlextContext.Performance.timed_operation("batch_processing"):
                results = []

                for item in items:
                    # Individual item processing with minimal context overhead
                    with FlextContext.Request.request_context(
                        operation_name=f"process_item_{item.get('id')}"
                    ):
                        result = await self.process_item(item)
                        results.append(result)

                return results
```

#### 3. **Selective Context Propagation**

```python
class SelectiveContextPropagation:
    def propagate_minimal_context(self):
        """Propagate only essential context for performance."""

        # Only include correlation and service information
        essential_context = {
            "X-Correlation-Id": FlextContext.Correlation.get_correlation_id(),
            "X-Service-Name": FlextContext.Service.get_service_name()
        }

        return {k: v for k, v in essential_context.items() if v is not None}
```

---

## 🚨 Common Pitfalls and Solutions

### Pitfall 1: Context Not Propagating

#### Problem

```python
# Context lost in async operations
async def problematic_async_operation():
    correlation_id = FlextContext.Correlation.get_correlation_id()

    # Context lost in new thread/task
    result = await some_async_operation()  # Context not available here
```

#### Solution

```python
# Proper context propagation in async
async def correct_async_operation():
    # Context automatically propagated in async/await
    with FlextContext.Correlation.inherit_correlation():
        result = await some_async_operation()  # Context available

    # For explicit thread operations, copy context
    import asyncio
    context = contextvars.copy_context()
    await asyncio.create_task(some_async_operation(), context=context)
```

### Pitfall 2: Context Memory Leaks

#### Problem

```python
# Not clearing context in long-running services
class ProblematicService:
    def process_requests(self):
        for request in self.request_stream:
            FlextContext.Request.set_user_id(request.user_id)
            self.process_request(request)
            # Context never cleared - memory leak!
```

#### Solution

```python
# Proper context lifecycle management
class CorrectService:
    def process_requests(self):
        for request in self.request_stream:
            # Use context managers for automatic cleanup
            with FlextContext.Request.request_context(user_id=request.user_id):
                self.process_request(request)
            # Context automatically cleared after each request
```

### Pitfall 3: Performance Impact of Large Context

#### Problem

```python
# Adding too much metadata
FlextContext.Performance.add_operation_metadata("large_data", huge_object)  # Bad!
```

#### Solution

```python
# Add summarized or essential metadata only
FlextContext.Performance.add_operation_metadata("data_size", len(huge_object))
FlextContext.Performance.add_operation_metadata("data_type", type(huge_object).__name__)
```

---

## 📋 Implementation Checklist

### Pre-Implementation

- [ ] **Identify Service Boundaries**: Define which services need context tracking
- [ ] **Plan Correlation Strategy**: Design correlation ID flow across services
- [ ] **Define Performance Metrics**: Identify which operations to monitor
- [ ] **Design Context Propagation**: Plan HTTP headers and message queue integration
- [ ] **Set Up Testing Environment**: Prepare context testing scenarios

### Implementation Phase

- [ ] **Basic Context Setup**: Implement service identification and correlation
- [ ] **Request Context**: Add user and operation context tracking
- [ ] **Performance Monitoring**: Integrate operation timing and metrics
- [ ] **Cross-Service Propagation**: Implement HTTP and message queue context passing
- [ ] **Error Handling**: Add context-aware error handling and logging

### Testing Phase

- [ ] **Unit Tests**: Test individual context operations
- [ ] **Integration Tests**: Test cross-service context propagation
- [ ] **Performance Tests**: Validate context overhead is minimal
- [ ] **Load Tests**: Test context under high concurrency
- [ ] **End-to-End Tests**: Validate complete request tracing

### Post-Implementation

- [ ] **Monitoring Setup**: Configure context-based monitoring and alerting
- [ ] **Documentation**: Document context patterns and usage guidelines
- [ ] **Training**: Train team members on context management patterns
- [ ] **Performance Tuning**: Optimize context operations based on usage patterns
- [ ] **Maintenance Plan**: Plan for ongoing context system maintenance

This implementation guide provides comprehensive coverage of FlextContext integration patterns, from basic setup through advanced cross-service communication and performance optimization. Follow these patterns to achieve enterprise-grade distributed tracing and observability throughout the FLEXT ecosystem.
