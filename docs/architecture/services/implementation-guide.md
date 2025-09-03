# FlextServices Implementation Guide

**Version**: 0.9.0  
**Target Audience**: FLEXT Developers, Service Architects  
**Implementation Time**: 2-3 weeks per service  
**Complexity**: Intermediate to Advanced

## 📖 Overview

This guide provides comprehensive instructions for implementing the Template Method service architecture using `FlextServices` across FLEXT applications. The service framework offers Template Method patterns for boilerplate elimination, service orchestration, performance monitoring, and enterprise-grade service management.

### Prerequisites

- Understanding of Template Method pattern and generic programming [TRequest, TDomain, TResult]
- Familiarity with railway-oriented programming (FlextResult patterns)
- Knowledge of service orchestration and enterprise architecture patterns
- Experience with dependency injection and Clean Architecture principles

### Implementation Benefits

- 📊 **80% boilerplate code elimination** through Template Method pattern
- 🔗 **Enterprise service orchestration** with workflow coordination and performance monitoring
- ⚡ **Comprehensive observability** with automatic metrics collection and correlation tracking
- 🔧 **Service registry and discovery** with health monitoring and load balancing
- 🌐 **Contract validation** with service boundary enforcement and SLA monitoring

---

## 🚀 Quick Start

### Basic Template Method Service

```python
from flext_core.services import FlextServices
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Template Method service with generic type parameters
class UserRegistrationService(
    FlextServices.ServiceProcessor[UserRequest, User, UserResponse]
):
    """User registration using Template Method pattern."""

    def process(self, request: UserRequest) -> FlextResult[User]:
        """Process request into domain object (business logic only)."""

        # Business validation
        if not request.email or "@" not in request.email:
            return FlextResult[User].fail("Invalid email address")

        # Create domain object
        user = User(
            email=request.email.lower().strip(),
            name=request.name.strip(),
            created_at=datetime.utcnow()
        )

        return FlextResult[User].ok(user)

    def build(self, user: User, *, correlation_id: str) -> UserResponse:
        """Build final result (pure function)."""
        return UserResponse(
            user_id=user.id,
            email=user.email,
            name=user.name,
            correlation_id=correlation_id,
            timestamp=datetime.utcnow()
        )

# Usage with automatic metrics and correlation
user_service = UserRegistrationService()
result = user_service.run_with_metrics("user_registration", request)

if result.success:
    print(f"User registered: {result.value.user_id}")
else:
    print(f"Registration failed: {result.error}")
```

### Service Orchestration Setup

```python
# Service orchestration with multiple services
class OrderFulfillmentOrchestrator:
    def __init__(self):
        self.orchestrator = FlextServices.ServiceOrchestrator()
        self.registry = FlextServices.ServiceRegistry()
        self.metrics = FlextServices.ServiceMetrics()

        # Register services
        self._setup_services()

    def _setup_services(self):
        """Register services for orchestration."""
        payment_service = PaymentProcessor()
        self.orchestrator.register_service("payment", payment_service)

        inventory_service = InventoryManager()
        self.orchestrator.register_service("inventory", inventory_service)

# Workflow execution with orchestration
workflow = {
    "workflow_id": "order_fulfillment_123",
    "steps": [
        {"step": "validate_payment", "service": "payment", "required": True},
        {"step": "reserve_inventory", "service": "inventory", "required": True}
    ]
}

result = orchestrator.orchestrate_workflow(workflow)
```

---

## 📚 Step-by-Step Implementation

### Step 1: Understanding Template Method Architecture

#### Generic Type Parameters Pattern

```python
from flext_core.services import FlextServices

# Template Method with three generic parameters
class DataProcessingService[TRequest, TDomain, TResult](
    FlextServices.ServiceProcessor[TRequest, TDomain, TResult]
):
    """
    Template Method pattern with generic types:
    - TRequest: Input request type
    - TDomain: Business domain object type
    - TResult: Final response type
    """

    def process(self, request: TRequest) -> FlextResult[TDomain]:
        """Abstract method: Convert request to domain object."""
        # Implement business logic here
        pass

    def build(self, domain: TDomain, *, correlation_id: str) -> TResult:
        """Abstract method: Build final result from domain object."""
        # Implement result building here
        pass

# The Template Method automatically provides:
# - run_with_metrics(): Complete processing pipeline
# - process_json(): JSON processing with validation
# - run_batch(): Batch processing with error collection
# - Automatic correlation ID generation
# - Performance tracking and metrics
```

#### Template Method Benefits

```python
class OrderProcessingService(
    FlextServices.ServiceProcessor[OrderRequest, Order, OrderResponse]
):
    """Order processing showing Template Method benefits."""

    # Template Method eliminates this boilerplate:
    # ❌ Manual correlation ID generation
    # ❌ Manual performance tracking
    # ❌ Manual error handling patterns
    # ❌ Manual JSON parsing and validation
    # ❌ Manual batch processing logic

    def process(self, request: OrderRequest) -> FlextResult[Order]:
        """Focus on business logic only."""

        # Business validation
        if request.amount <= 0:
            return FlextResult[Order].fail("Amount must be positive")

        # Business logic
        order = Order(
            customer_id=request.customer_id,
            amount=request.amount,
            items=request.items,
            status="processing"
        )

        # Business rule validation
        if order.amount > 10000:  # High-value order
            order.approval_required = True

        return FlextResult[Order].ok(order)

    def build(self, order: Order, *, correlation_id: str) -> OrderResponse:
        """Pure function - no side effects."""
        return OrderResponse(
            order_id=order.id,
            status=order.status,
            amount=order.amount,
            correlation_id=correlation_id,  # Template Method provides this
            created_at=order.created_at
        )

# Template Method usage patterns
order_service = OrderProcessingService()

# 1. Standard processing with metrics
result = order_service.run_with_metrics("order_processing", order_request)

# 2. JSON processing with validation
json_result = order_service.process_json(
    json_string,
    OrderRequest,  # Pydantic model for validation
    lambda req: order_service.run_with_metrics("json_order", req)
)

# 3. Batch processing with error collection
batch_requests = [OrderRequest(...), OrderRequest(...)]
successes, errors = order_service.run_batch(
    batch_requests,
    lambda req: order_service.run_with_metrics("batch_order", req)
)
```

### Step 2: Implementing Service Orchestration Patterns

#### Basic Service Orchestration

```python
class ServiceOrchestrationService:
    """Service orchestration implementation patterns."""

    def __init__(self):
        self.orchestrator = FlextServices.ServiceOrchestrator()
        self.registry = FlextServices.ServiceRegistry()
        self.metrics = FlextServices.ServiceMetrics()

    def setup_microservices_orchestration(self):
        """Setup microservices for orchestration."""

        # Register individual services
        auth_service = AuthenticationService()
        payment_service = PaymentProcessingService()
        notification_service = NotificationService()

        # Register with orchestrator
        self.orchestrator.register_service("auth", auth_service)
        self.orchestrator.register_service("payment", payment_service)
        self.orchestrator.register_service("notification", notification_service)

        # Register with service registry for discovery
        self.registry.register({
            "name": "auth_service",
            "type": "authentication",
            "endpoint": "https://auth.company.com",
            "version": "2.1.0"
        })

        self.registry.register({
            "name": "payment_service",
            "type": "payment",
            "endpoint": "https://payment.company.com",
            "version": "1.5.0"
        })

    def execute_complex_workflow(
        self,
        workflow_data: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Execute complex multi-service workflow."""

        workflow_definition = {
            "workflow_id": f"complex_workflow_{self._generate_id()}",
            "steps": [
                {
                    "step": "authenticate_user",
                    "service": "auth",
                    "input": workflow_data.get("auth_data"),
                    "required": True,
                    "timeout_ms": 5000
                },
                {
                    "step": "process_payment",
                    "service": "payment",
                    "input": workflow_data.get("payment_data"),
                    "required": True,
                    "depends_on": ["authenticate_user"],
                    "timeout_ms": 10000
                },
                {
                    "step": "send_confirmation",
                    "service": "notification",
                    "input": workflow_data.get("notification_data"),
                    "required": False,  # Optional step
                    "depends_on": ["process_payment"],
                    "timeout_ms": 3000
                }
            ]
        }

        # Execute with performance tracking
        start_time = time.time()
        workflow_result = self.orchestrator.orchestrate_workflow(workflow_definition)
        duration_ms = (time.time() - start_time) * 1000

        # Track orchestration metrics
        self.metrics.track_service_call(
            "complex_workflow_orchestrator",
            "execute_workflow",
            duration_ms
        )

        return workflow_result

    def implement_compensation_patterns(
        self,
        workflow_id: str,
        failed_step: str
    ) -> FlextResult[dict[str, object]]:
        """Implement compensation patterns for failed workflows."""

        compensation_actions = {
            "authenticate_user": [],  # No compensation needed
            "process_payment": ["refund_payment", "release_funds"],
            "send_confirmation": ["send_failure_notification"]
        }

        actions = compensation_actions.get(failed_step, [])
        compensation_results = {}

        for action in actions:
            # Execute compensation action
            action_result = self._execute_compensation_action(action, workflow_id)
            compensation_results[action] = action_result.success

        return FlextResult[dict[str, object]].ok(compensation_results)
```

#### Advanced Service Coordination

```python
class AdvancedServiceCoordination:
    """Advanced service coordination patterns."""

    def __init__(self):
        self.orchestrator = FlextServices.ServiceOrchestrator()

    def implement_saga_pattern(
        self,
        saga_definition: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Implement Saga pattern for distributed transactions."""

        saga_steps = saga_definition["steps"]
        saga_id = saga_definition["saga_id"]
        executed_steps = []

        try:
            # Execute saga steps
            for step in saga_steps:
                step_result = self._execute_saga_step(step)

                if step_result.is_failure:
                    # Compensation required
                    compensation_result = self._compensate_saga_steps(
                        executed_steps, saga_id
                    )
                    return FlextResult[dict[str, object]].fail(
                        f"Saga failed at step {step['name']}: {step_result.error}. "
                        f"Compensation: {compensation_result}"
                    )

                executed_steps.append(step)

            return FlextResult[dict[str, object]].ok({
                "saga_id": saga_id,
                "status": "completed",
                "executed_steps": len(executed_steps)
            })

        except Exception as e:
            # Handle unexpected errors
            compensation_result = self._compensate_saga_steps(executed_steps, saga_id)
            return FlextResult[dict[str, object]].fail(
                f"Saga failed with exception: {e}. Compensation: {compensation_result}"
            )

    def implement_circuit_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        timeout_ms: int = 60000
    ) -> FlextResult[dict[str, object]]:
        """Implement circuit breaker pattern for service resilience."""

        circuit_breaker_config = {
            "service_name": service_name,
            "failure_threshold": failure_threshold,
            "timeout_ms": timeout_ms,
            "current_failures": 0,
            "state": "closed",  # closed, open, half_open
            "last_failure_time": None
        }

        # Store circuit breaker configuration
        self._circuit_breakers[service_name] = circuit_breaker_config

        return FlextResult[dict[str, object]].ok(circuit_breaker_config)
```

### Step 3: Service Registry and Discovery Implementation

#### Service Registry Setup

```python
class EnterpriseServiceRegistry:
    """Enterprise service registry implementation."""

    def __init__(self):
        self.registry = FlextServices.ServiceRegistry()
        self.metrics = FlextServices.ServiceMetrics()
        self.validator = FlextServices.ServiceValidation()

        # Service storage and monitoring
        self._service_health: dict[str, dict[str, object]] = {}
        self._load_balancing_weights: dict[str, dict[str, float]] = {}

    def register_microservice_with_health_monitoring(
        self,
        service_info: dict[str, object]
    ) -> FlextResult[str]:
        """Register microservice with comprehensive health monitoring."""

        # Validate service registration
        validation_schema = lambda info: (
            FlextResult[dict].ok(info)
            if all(key in info for key in ["name", "endpoint", "type", "version"])
            else FlextResult[dict].fail("Missing required service information")
        )

        validation_result = self.validator.validate_input(service_info, validation_schema)
        if validation_result.is_failure:
            return FlextResult[str].fail(validation_result.error)

        # Register with FlextServices registry
        registration_result = self.registry.register(service_info)
        if registration_result.is_failure:
            return registration_result

        service_id = registration_result.value
        service_name = service_info["name"]

        # Initialize health monitoring
        self._service_health[service_name] = {
            "status": "registered",
            "health_checks": 0,
            "last_health_check": None,
            "response_time_ms": 0,
            "error_count": 0,
            "uptime_percentage": 100.0
        }

        # Initialize load balancing
        self._load_balancing_weights[service_name] = {
            "current_weight": 100,
            "base_weight": 100,
            "health_multiplier": 1.0,
            "performance_multiplier": 1.0
        }

        return FlextResult[str].ok(service_id)

    def discover_services_with_load_balancing(
        self,
        service_type: str,
        load_balancing_strategy: str = "weighted_round_robin"
    ) -> FlextResult[list[dict[str, object]]]:
        """Discover services with load balancing information."""

        # Discover services by type
        all_services = []
        for service_name, health_info in self._service_health.items():
            service_result = self.registry.discover(service_name)

            if service_result.success:
                service_data = service_result.value

                if service_data.get("type") == service_type:
                    # Add health and load balancing info
                    service_entry = {
                        **service_data,
                        "health_status": health_info["status"],
                        "response_time_ms": health_info["response_time_ms"],
                        "uptime_percentage": health_info["uptime_percentage"],
                        "load_balancing_weight": self._load_balancing_weights[service_name]["current_weight"]
                    }
                    all_services.append(service_entry)

        if not all_services:
            return FlextResult[list[dict[str, object]]].fail(
                f"No services found for type: {service_type}"
            )

        # Apply load balancing strategy
        if load_balancing_strategy == "weighted_round_robin":
            # Sort by weight (highest first)
            all_services.sort(key=lambda s: s["load_balancing_weight"], reverse=True)
        elif load_balancing_strategy == "least_response_time":
            # Sort by response time (lowest first)
            all_services.sort(key=lambda s: s["response_time_ms"])
        elif load_balancing_strategy == "highest_uptime":
            # Sort by uptime (highest first)
            all_services.sort(key=lambda s: s["uptime_percentage"], reverse=True)

        return FlextResult[list[dict[str, object]]].ok(all_services)

    def perform_comprehensive_health_checks(self) -> FlextResult[dict[str, object]]:
        """Perform comprehensive health checks with detailed reporting."""

        health_summary = {
            "total_services": len(self._service_health),
            "healthy_services": 0,
            "unhealthy_services": 0,
            "degraded_services": 0,
            "average_response_time": 0,
            "services_detail": {}
        }

        total_response_time = 0

        for service_name, health_info in self._service_health.items():
            # Perform health check
            health_start = time.time()
            health_result = self._perform_health_check(service_name)
            health_duration = (time.time() - health_start) * 1000

            # Update health information
            if health_result.success:
                status = "healthy"
                health_summary["healthy_services"] += 1
            else:
                status = "unhealthy"
                health_summary["unhealthy_services"] += 1
                health_info["error_count"] += 1

            # Update metrics
            health_info["health_checks"] += 1
            health_info["last_health_check"] = datetime.utcnow().isoformat()
            health_info["response_time_ms"] = health_duration
            health_info["status"] = status

            # Calculate uptime percentage
            total_checks = health_info["health_checks"]
            error_count = health_info["error_count"]
            health_info["uptime_percentage"] = ((total_checks - error_count) / total_checks) * 100

            # Update load balancing weights
            self._update_load_balancing_weight(service_name, health_info)

            # Track metrics
            self.metrics.track_service_call(service_name, "health_check", health_duration)

            total_response_time += health_duration

            # Store detailed health information
            health_summary["services_detail"][service_name] = {
                "status": status,
                "response_time_ms": health_duration,
                "uptime_percentage": health_info["uptime_percentage"],
                "total_checks": total_checks,
                "error_count": error_count,
                "load_balancing_weight": self._load_balancing_weights[service_name]["current_weight"]
            }

        # Calculate average response time
        if health_summary["total_services"] > 0:
            health_summary["average_response_time"] = total_response_time / health_summary["total_services"]

        health_summary["health_check_timestamp"] = datetime.utcnow().isoformat()

        return FlextResult[dict[str, object]].ok(health_summary)
```

### Step 4: Performance Monitoring and Metrics Implementation

#### Comprehensive Performance Monitoring

```python
class ServicePerformanceManager:
    """Comprehensive service performance monitoring."""

    def __init__(self):
        self.metrics = FlextServices.ServiceMetrics()

        # Performance data storage
        self._performance_history: dict[str, list[dict[str, object]]] = {}
        self._performance_baselines: dict[str, dict[str, float]] = {}
        self._alert_thresholds: dict[str, dict[str, float]] = {}
        self._anomaly_detection: dict[str, object] = {}

    def track_comprehensive_service_metrics(
        self,
        service_name: str,
        operation_name: str,
        duration_ms: float,
        success: bool,
        metadata: dict[str, object] | None = None
    ) -> FlextResult[None]:
        """Track comprehensive service metrics with anomaly detection."""

        # Track with FlextServices metrics
        tracking_result = self.metrics.track_service_call(
            service_name,
            operation_name,
            duration_ms
        )

        if tracking_result.is_failure:
            return tracking_result

        # Store detailed performance record
        performance_record = {
            "service_name": service_name,
            "operation_name": operation_name,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Add to performance history
        service_key = f"{service_name}.{operation_name}"
        if service_key not in self._performance_history:
            self._performance_history[service_key] = []

        self._performance_history[service_key].append(performance_record)

        # Keep only recent history (last 1000 records)
        if len(self._performance_history[service_key]) > 1000:
            self._performance_history[service_key] = self._performance_history[service_key][-1000:]

        # Update performance baselines
        self._update_performance_baselines(service_key, performance_record)

        # Detect performance anomalies
        anomaly_result = self._detect_performance_anomaly(service_key, performance_record)

        return FlextResult[None].ok(None)

    def generate_comprehensive_performance_report(
        self,
        service_name: str,
        time_window_hours: int = 24
    ) -> FlextResult[dict[str, object]]:
        """Generate comprehensive performance analysis report."""

        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Collect performance data for all operations
        service_operations = {}
        overall_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0
        }

        for service_key, history in self._performance_history.items():
            if service_key.startswith(f"{service_name}."):
                operation_name = service_key.split(".", 1)[1]

                # Filter by time window
                recent_calls = [
                    call for call in history
                    if datetime.fromisoformat(call["timestamp"]) >= cutoff_time
                ]

                if recent_calls:
                    # Calculate operation statistics
                    durations = [call["duration_ms"] for call in recent_calls]
                    successes = [call for call in recent_calls if call["success"]]
                    failures = [call for call in recent_calls if not call["success"]]

                    operation_stats = {
                        "operation_name": operation_name,
                        "total_calls": len(recent_calls),
                        "successful_calls": len(successes),
                        "failed_calls": len(failures),
                        "success_rate": len(successes) / len(recent_calls),
                        "average_duration_ms": sum(durations) / len(durations),
                        "min_duration_ms": min(durations),
                        "max_duration_ms": max(durations),
                        "median_duration_ms": sorted(durations)[len(durations) // 2],
                        "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
                        "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)],
                        "throughput_per_hour": len(recent_calls) / time_window_hours,
                        "error_rate": len(failures) / len(recent_calls),
                        "performance_trend": self._calculate_performance_trend(recent_calls),
                        "anomalies_detected": self._get_operation_anomalies(service_key, time_window_hours)
                    }

                    service_operations[operation_name] = operation_stats

                    # Update overall statistics
                    overall_stats["total_calls"] += len(recent_calls)
                    overall_stats["successful_calls"] += len(successes)
                    overall_stats["failed_calls"] += len(failures)
                    overall_stats["total_duration_ms"] += sum(durations)

        if not service_operations:
            return FlextResult[dict[str, object]].fail(
                f"No performance data found for service {service_name}"
            )

        # Generate comprehensive report
        report = {
            "service_name": service_name,
            "time_window_hours": time_window_hours,
            "report_generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_operations": len(service_operations),
                "total_calls": overall_stats["total_calls"],
                "success_rate": overall_stats["successful_calls"] / overall_stats["total_calls"],
                "error_rate": overall_stats["failed_calls"] / overall_stats["total_calls"],
                "average_duration_ms": overall_stats["total_duration_ms"] / overall_stats["total_calls"],
                "throughput_per_hour": overall_stats["total_calls"] / time_window_hours,
                "health_score": self._calculate_service_health_score(service_name),
                "performance_grade": self._calculate_performance_grade(service_operations)
            },
            "operations": service_operations,
            "baselines": self._performance_baselines.get(service_name, {}),
            "thresholds": self._alert_thresholds.get(service_name, {}),
            "recommendations": self._generate_performance_recommendations(service_name, service_operations),
            "alerts": self._generate_performance_alerts(service_name, service_operations)
        }

        return FlextResult[dict[str, object]].ok(report)

# Usage examples
performance_manager = ServicePerformanceManager()

# Track service performance
performance_manager.track_comprehensive_service_metrics(
    "user-service",
    "create_user",
    125.3,  # duration in ms
    True,   # success
    {"request_size": "medium", "cache_hit": False}
)

# Generate performance report
report_result = performance_manager.generate_comprehensive_performance_report(
    "user-service",
    time_window_hours=6
)

if report_result.success:
    report = report_result.value
    print(f"Performance Report for {report['service_name']}:")
    print(f"  Health score: {report['summary']['health_score']}")
    print(f"  Performance grade: {report['summary']['performance_grade']}")
    print(f"  Success rate: {report['summary']['success_rate']:.1%}")
    print(f"  Average duration: {report['summary']['average_duration_ms']:.1f}ms")
```

---

## ⚡ Advanced Implementation Patterns

### Pattern 1: Complex ETL Service with Template Method

```python
class ETLDataProcessingService(
    FlextServices.ServiceProcessor[ETLRequest, ETLPipeline, ETLResponse]
):
    """Advanced ETL processing using Template Method pattern."""

    def __init__(self):
        super().__init__()
        self.orchestrator = FlextServices.ServiceOrchestrator()
        self._setup_etl_components()

    def _setup_etl_components(self):
        """Setup ETL component services."""
        tap_service = MeltanoTapService()
        target_service = MeltanoTargetService()
        transform_service = DBTTransformService()

        self.orchestrator.register_service("tap", tap_service)
        self.orchestrator.register_service("target", target_service)
        self.orchestrator.register_service("transform", transform_service)

    def process(self, request: ETLRequest) -> FlextResult[ETLPipeline]:
        """Process ETL request with comprehensive validation."""

        # ETL configuration validation
        if not request.source_config or not request.target_config:
            return FlextResult[ETLPipeline].fail("Source and target configuration required")

        # Singer schema validation
        schema_result = self._validate_singer_schema(request.singer_schema)
        if schema_result.is_failure:
            return schema_result

        # Create ETL pipeline
        pipeline = ETLPipeline(
            id=self._generate_pipeline_id(),
            source_type=request.source_type,
            target_type=request.target_type,
            singer_schema=request.singer_schema,
            transformations=request.transformations,
            schedule=request.schedule,
            status="created"
        )

        # ETL business rule validation
        compatibility_result = self._validate_etl_compatibility(pipeline)
        if compatibility_result.is_failure:
            return compatibility_result

        return FlextResult[ETLPipeline].ok(pipeline)

    def build(self, pipeline: ETLPipeline, *, correlation_id: str) -> ETLResponse:
        """Build ETL response with execution metadata."""
        return ETLResponse(
            pipeline_id=pipeline.id,
            source_type=pipeline.source_type,
            target_type=pipeline.target_type,
            status=pipeline.status,
            estimated_records=pipeline.estimated_records,
            estimated_duration_minutes=pipeline.estimated_duration_minutes,
            correlation_id=correlation_id,
            created_at=pipeline.created_at
        )

    def execute_etl_pipeline(self, pipeline_id: str) -> FlextResult[ETLExecutionResult]:
        """Execute ETL pipeline using service orchestration."""

        etl_workflow = {
            "workflow_id": f"etl_{pipeline_id}",
            "steps": [
                {"step": "extract", "service": "tap", "timeout_ms": 300000},
                {"step": "transform", "service": "transform", "depends_on": ["extract"]},
                {"step": "load", "service": "target", "depends_on": ["transform"]}
            ]
        }

        workflow_result = self.orchestrator.orchestrate_workflow(etl_workflow)

        if workflow_result.success:
            return FlextResult[ETLExecutionResult].ok(
                ETLExecutionResult(
                    pipeline_id=pipeline_id,
                    status="completed",
                    records_processed=workflow_result.value.get("total_records", 0),
                    execution_time_ms=workflow_result.value.get("duration_ms", 0)
                )
            )
        else:
            return FlextResult[ETLExecutionResult].fail(workflow_result.error)
```

### Pattern 2: Microservice Integration with Service Registry

```python
class MicroserviceIntegrationManager:
    """Microservice integration using FlextServices registry patterns."""

    def __init__(self):
        self.registry = FlextServices.ServiceRegistry()
        self.orchestrator = FlextServices.ServiceOrchestrator()
        self.validator = FlextServices.ServiceValidation()

    def register_microservice_ecosystem(self) -> FlextResult[dict[str, str]]:
        """Register complete microservice ecosystem."""

        microservices = [
            {
                "name": "auth-service",
                "type": "authentication",
                "endpoint": "https://auth.company.com",
                "version": "2.1.0",
                "capabilities": ["jwt", "oauth2", "saml"]
            },
            {
                "name": "user-service",
                "type": "user_management",
                "endpoint": "https://users.company.com",
                "version": "1.8.0",
                "capabilities": ["crud", "profile", "preferences"]
            },
            {
                "name": "notification-service",
                "type": "messaging",
                "endpoint": "https://notifications.company.com",
                "version": "1.5.0",
                "capabilities": ["email", "sms", "push"]
            }
        ]

        registration_results = {}

        for service_info in microservices:
            # Validate service configuration
            validation_result = self._validate_microservice_config(service_info)
            if validation_result.is_failure:
                return FlextResult[dict[str, str]].fail(
                    f"Service {service_info['name']} validation failed: {validation_result.error}"
                )

            # Register with service registry
            registration_result = self.registry.register(service_info)
            if registration_result.success:
                registration_results[service_info["name"]] = registration_result.value

                # Also register with orchestrator
                self.orchestrator.register_service(
                    service_info["name"],
                    self._create_service_proxy(service_info)
                )

        return FlextResult[dict[str, str]].ok(registration_results)

    def execute_cross_service_workflow(
        self,
        workflow_request: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Execute workflow spanning multiple microservices."""

        # Discover required services
        auth_service = self.registry.discover("auth-service")
        user_service = self.registry.discover("user-service")
        notification_service = self.registry.discover("notification-service")

        if any(service.is_failure for service in [auth_service, user_service, notification_service]):
            return FlextResult[dict[str, object]].fail("Required services not available")

        # Define cross-service workflow
        cross_service_workflow = {
            "workflow_id": f"cross_service_{self._generate_id()}",
            "steps": [
                {
                    "step": "authenticate_request",
                    "service": "auth-service",
                    "input": workflow_request.get("credentials"),
                    "required": True
                },
                {
                    "step": "create_user_profile",
                    "service": "user-service",
                    "input": workflow_request.get("user_data"),
                    "required": True,
                    "depends_on": ["authenticate_request"]
                },
                {
                    "step": "send_welcome_notification",
                    "service": "notification-service",
                    "input": workflow_request.get("notification_data"),
                    "required": False,
                    "depends_on": ["create_user_profile"]
                }
            ]
        }

        # Execute with comprehensive error handling
        workflow_result = self.orchestrator.orchestrate_workflow(cross_service_workflow)

        return workflow_result
```

---

## 📋 Implementation Checklist

### Pre-Implementation

- [ ] **Service Architecture Analysis**: Analyze current service patterns and identify Template Method opportunities
- [ ] **Type System Design**: Define generic type parameters [TRequest, TDomain, TResult] for each service
- [ ] **Integration Points**: Plan FlextServices integration with existing codebase
- [ ] **Performance Requirements**: Determine orchestration and monitoring requirements

### Core Template Method Implementation

- [ ] **ServiceProcessor Extension**: Implement Template Method pattern with proper generic types
- [ ] **Business Logic Separation**: Move business logic to process() method
- [ ] **Pure Function Building**: Implement build() method as pure function
- [ ] **Error Handling**: Use FlextResult throughout processing pipeline
- [ ] **Correlation Tracking**: Leverage automatic correlation ID generation

### Service Architecture Enhancement

- [ ] **Service Orchestration**: Implement ServiceOrchestrator for workflow coordination
- [ ] **Service Registry**: Add ServiceRegistry for service discovery and health monitoring
- [ ] **Performance Monitoring**: Integrate ServiceMetrics for comprehensive observability
- [ ] **Service Validation**: Add ServiceValidation for contract enforcement

### Advanced Features

- [ ] **Batch Processing**: Implement batch processing patterns using run_batch()
- [ ] **JSON Integration**: Add JSON processing capabilities with process_JSON()
- [ ] **Circuit Breakers**: Implement resilience patterns with circuit breakers
- [ ] **Load Balancing**: Add service discovery with load balancing capabilities

### Validation and Testing

- [ ] **Unit Testing**: Test Template Method implementation with comprehensive coverage
- [ ] **Integration Testing**: Test service orchestration and registry functionality
- [ ] **Performance Testing**: Validate performance monitoring and metrics collection
- [ ] **Contract Testing**: Test service validation and contract enforcement

### Production Readiness

- [ ] **Performance Monitoring**: Deploy comprehensive service metrics and monitoring
- [ ] **Health Checks**: Implement service health monitoring and alerting
- [ ] **Error Handling**: Validate error handling and recovery patterns
- [ ] **Documentation**: Update service documentation with Template Method patterns
- [ ] **Team Training**: Train team on FlextServices architecture and patterns

This implementation guide provides comprehensive coverage of FlextServices Template Method patterns, from basic service processing through advanced orchestration and monitoring, ensuring consistent service architecture and boilerplate elimination across all FLEXT applications.
