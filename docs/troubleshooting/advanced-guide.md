# Advanced Troubleshooting Guide

This guide covers advanced troubleshooting scenarios for FLEXT Core, including common issues, debugging techniques, and performance optimization.

## 🔍 Diagnostic Tools and Techniques

### FlextResult Error Analysis

When working with FlextResult, errors can be complex. Here are debugging techniques:

```python
from flext_core import FlextResult
import logging

def debug_flext_result(result: FlextResult[any], operation_name: str) -> None:
    """Advanced FlextResult debugging."""
    if result.is_failure:
        logging.error(f"Operation '{operation_name}' failed: {result.error}")
        
        # Add stack trace context for debugging
        import traceback
        logging.debug(f"Stack trace for '{operation_name}':\n{traceback.format_stack()}")
        
        # Log additional context if available
        if hasattr(result, '_debug_context'):
            logging.debug(f"Debug context: {result._debug_context}")

def enhanced_operation() -> FlextResult[str]:
    """Example operation with enhanced debugging."""
    try:
        # Your operation logic here
        return FlextResult.ok("success")
    except Exception as e:
        # Enhanced error context
        error_result = FlextResult.fail(f"Operation failed: {e}")
        # Add debug context (custom extension)
        error_result._debug_context = {
            "exception_type": type(e).__name__,
            "exception_args": e.args,
            "operation": "enhanced_operation"
        }
        return error_result

# Usage with debugging
result = enhanced_operation()
debug_flext_result(result, "enhanced_operation")
```

### FlextContainer Debugging

Debug dependency injection issues:

```python
from flext_core import get_flext_container, FlextContainer
import json

def diagnose_container_issues():
    """Comprehensive container diagnostics."""
    container = get_flext_container()
    
    # Get all registered services
    services = container.list_services()
    print(f"Registered services: {services}")
    
    # Get detailed service information
    service_info = container.get_service_info()
    print(f"Service details: {json.dumps(service_info, indent=2)}")
    
    # Check for missing dependencies
    missing_services = []
    for service_name in services:
        service_result = container.get(service_name)
        if service_result.is_failure:
            missing_services.append({
                "service": service_name,
                "error": service_result.error
            })
    
    if missing_services:
        print("Missing or broken services:")
        for service in missing_services:
            print(f"  - {service['service']}: {service['error']}")

def validate_container_health() -> FlextResult[dict]:
    """Validate container health with detailed diagnostics."""
    container = get_flext_container()
    
    health_report = {
        "total_services": 0,
        "healthy_services": 0,
        "failed_services": [],
        "circular_dependencies": [],
        "memory_usage": {}
    }
    
    services = container.list_services()
    health_report["total_services"] = len(services)
    
    for service_name in services:
        try:
            result = container.get(service_name)
            if result.is_success:
                health_report["healthy_services"] += 1
                
                # Check memory usage
                import sys
                service_size = sys.getsizeof(result.data)
                health_report["memory_usage"][service_name] = service_size
            else:
                health_report["failed_services"].append({
                    "name": service_name,
                    "error": result.error
                })
        except RecursionError:
            health_report["circular_dependencies"].append(service_name)
        except Exception as e:
            health_report["failed_services"].append({
                "name": service_name,
                "error": f"Unexpected error: {e}"
            })
    
    if health_report["failed_services"] or health_report["circular_dependencies"]:
        return FlextResult.fail(f"Container health issues detected: {health_report}")
    
    return FlextResult.ok(health_report)
```

### Configuration Debugging

Debug configuration issues with detailed analysis:

```python
from flext_core.config import FlextCoreSettings
from pydantic import ValidationError
import os
import json

class DiagnosticSettings(FlextCoreSettings):
    """Settings class with enhanced diagnostics."""
    
    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            self._diagnose_validation_error(e)
            raise
    
    def _diagnose_validation_error(self, error: ValidationError):
        """Provide detailed diagnosis of validation errors."""
        print("Configuration Validation Error Diagnosis:")
        print("=" * 50)
        
        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            error_msg = err["msg"]
            input_value = err.get("input", "N/A")
            
            print(f"Field: {field_path}")
            print(f"Error Type: {error_type}")
            print(f"Message: {error_msg}")
            print(f"Input Value: {input_value}")
            
            # Check environment variable
            env_var_name = self._get_env_var_name(field_path)
            env_value = os.getenv(env_var_name)
            print(f"Environment Variable: {env_var_name} = {env_value}")
            print("-" * 30)
    
    def _get_env_var_name(self, field_path: str) -> str:
        """Generate environment variable name for field."""
        # This would need to match your env_prefix logic
        prefix = getattr(self.model_config, "env_prefix", "FLEXT_")
        return f"{prefix}{field_path.upper().replace(' -> ', '_')}"
    
    @classmethod
    def diagnose_environment(cls):
        """Diagnose environment variable configuration."""
        print("Environment Variable Diagnosis:")
        print("=" * 40)
        
        # List all FLEXT-related environment variables
        flext_vars = {k: v for k, v in os.environ.items() if k.startswith("FLEXT_")}
        
        if flext_vars:
            print("Found FLEXT environment variables:")
            for var, value in flext_vars.items():
                # Mask sensitive values
                if any(sensitive in var.lower() for sensitive in ["password", "key", "token", "secret"]):
                    display_value = "***MASKED***"
                else:
                    display_value = value
                print(f"  {var} = {display_value}")
        else:
            print("No FLEXT environment variables found.")
        
        print()
        print("Expected environment variables for this configuration:")
        # This would list expected variables based on the model fields
        print("  FLEXT_ENVIRONMENT")
        print("  FLEXT_LOG_LEVEL")
        print("  FLEXT_DEBUG")
        print("  ... (others based on your configuration)")
```

## 🚨 Common Issues and Solutions

### Issue 1: FlextResult Chain Failures

**Problem**: Complex FlextResult chains failing unexpectedly.

**Symptoms**:

```python
result = (
    operation1()
    .flat_map(operation2)
    .flat_map(operation3)
    .map(final_transform)
)
# Result is failure but unclear which step failed
```

**Solution**: Add debugging to each step.

```python
def debug_chain_operation():
    """Debug complex FlextResult chains."""
    
    def debug_step(step_name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                print(f"Executing step: {step_name}")
                result = func(*args, **kwargs)
                if result.is_failure:
                    print(f"Step '{step_name}' failed: {result.error}")
                else:
                    print(f"Step '{step_name}' succeeded")
                return result
            return wrapper
        return decorator
    
    @debug_step("operation1")
    def operation1() -> FlextResult[str]:
        return FlextResult.ok("step1")
    
    @debug_step("operation2")
    def operation2(data: str) -> FlextResult[str]:
        return FlextResult.ok(f"{data}_step2")
    
    @debug_step("operation3")
    def operation3(data: str) -> FlextResult[str]:
        if len(data) > 20:  # Simulated failure condition
            return FlextResult.fail("Data too long")
        return FlextResult.ok(f"{data}_step3")
    
    @debug_step("final_transform")
    def final_transform(data: str) -> str:
        return data.upper()
    
    # Execute with debugging
    result = (
        operation1()
        .flat_map(operation2)
        .flat_map(operation3)
        .map(final_transform)
    )
    
    return result
```

### Issue 2: Memory Leaks in FlextContainer

**Problem**: Services accumulating in container causing memory issues.

**Symptoms**:

- Increasing memory usage over time
- OutOfMemoryError in long-running applications
- Container performance degradation

**Solution**: Implement container cleanup and monitoring.

```python
import gc
import psutil
from typing import Dict, Any
from flext_core import get_flext_container

class ContainerMemoryMonitor:
    """Monitor and manage container memory usage."""
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
        self.service_counts = {}
    
    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def monitor_container_memory(self) -> Dict[str, Any]:
        """Monitor container memory usage."""
        container = get_flext_container()
        current_memory = self._get_memory_usage()
        
        services = container.list_services()
        service_info = container.get_service_info()
        
        report = {
            "baseline_memory_mb": self.baseline_memory,
            "current_memory_mb": current_memory,
            "memory_increase_mb": current_memory - self.baseline_memory,
            "service_count": len(services),
            "services": service_info,
            "gc_stats": {
                "collections": gc.get_stats(),
                "objects": len(gc.get_objects()),
            }
        }
        
        return report
    
    def cleanup_container(self) -> FlextResult[Dict[str, Any]]:
        """Clean up container and force garbage collection."""
        container = get_flext_container()
        
        before_memory = self._get_memory_usage()
        before_services = len(container.list_services())
        
        # Clear container (this would need to be implemented in FlextContainer)
        # container.clear()  # Hypothetical method
        
        # Force garbage collection
        collected = gc.collect()
        
        after_memory = self._get_memory_usage()
        after_services = len(container.list_services())
        
        cleanup_report = {
            "memory_before_mb": before_memory,
            "memory_after_mb": after_memory,
            "memory_freed_mb": before_memory - after_memory,
            "services_before": before_services,
            "services_after": after_services,
            "gc_collected": collected
        }
        
        return FlextResult.ok(cleanup_report)

# Usage
monitor = ContainerMemoryMonitor()

# Regular monitoring
memory_report = monitor.monitor_container_memory()
if memory_report["memory_increase_mb"] > 100:  # 100MB threshold
    print("High memory usage detected, initiating cleanup...")
    cleanup_result = monitor.cleanup_container()
    if cleanup_result.is_success:
        print(f"Cleanup successful: {cleanup_result.data}")
```

### Issue 3: Configuration Environment Conflicts

**Problem**: Different environments loading wrong configuration values.

**Symptoms**:

- Production using development settings
- Environment variables not being respected
- Settings validation passing but values incorrect

**Solution**: Environment-specific validation and debugging.

```python
from flext_core.config import FlextCoreSettings
from flext_core.constants import FlextEnvironment
import os
from typing import Any, Dict

class EnvironmentDiagnostics:
    """Diagnose environment configuration issues."""
    
    @staticmethod
    def diagnose_environment_detection() -> Dict[str, Any]:
        """Diagnose how environment is being detected."""
        
        # Check all possible environment indicators
        env_indicators = {
            "FLEXT_ENVIRONMENT": os.getenv("FLEXT_ENVIRONMENT"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT"),
            "ENV": os.getenv("ENV"),
            "NODE_ENV": os.getenv("NODE_ENV"),  # Common in web apps
            "PYTHONPATH": os.getenv("PYTHONPATH"),
            "PWD": os.getenv("PWD"),
        }
        
        # Detect actual environment
        try:
            settings = FlextCoreSettings()
            detected_env = settings.environment
        except Exception as e:
            detected_env = f"Error: {e}"
        
        return {
            "environment_variables": env_indicators,
            "detected_environment": detected_env,
            "flext_environment_values": [e.value for e in FlextEnvironment],
        }
    
    @staticmethod
    def validate_environment_consistency(settings_class: type[FlextCoreSettings]) -> FlextResult[Dict[str, Any]]:
        """Validate that environment settings are consistent."""
        
        try:
            settings = settings_class()
            
            consistency_report = {
                "environment": settings.environment.value,
                "debug_mode": settings.debug,
                "log_level": settings.log_level.value,
                "issues": []
            }
            
            # Check for common inconsistencies
            if settings.environment == FlextEnvironment.PRODUCTION:
                if settings.debug:
                    consistency_report["issues"].append(
                        "CRITICAL: Debug mode enabled in production"
                    )
                
                if settings.log_level.value in ["DEBUG", "TRACE"]:
                    consistency_report["issues"].append(
                        f"WARNING: Debug logging ({settings.log_level.value}) in production"
                    )
            
            elif settings.environment == FlextEnvironment.DEVELOPMENT:
                if not settings.debug:
                    consistency_report["issues"].append(
                        "INFO: Debug mode disabled in development (unusual but not critical)"
                    )
            
            # Check environment variable consistency
            env_var_env = os.getenv("FLEXT_ENVIRONMENT", "").lower()
            if env_var_env and env_var_env != settings.environment.value:
                consistency_report["issues"].append(
                    f"CRITICAL: Environment variable ({env_var_env}) doesn't match detected environment ({settings.environment.value})"
                )
            
            return FlextResult.ok(consistency_report)
            
        except Exception as e:
            return FlextResult.fail(f"Environment validation failed: {e}")

# Usage
def diagnose_configuration_issues():
    """Complete configuration diagnostics."""
    
    print("Environment Detection Diagnosis:")
    print("=" * 40)
    env_diagnosis = EnvironmentDiagnostics.diagnose_environment_detection()
    
    for key, value in env_diagnosis.items():
        print(f"{key}: {value}")
    
    print("\nEnvironment Consistency Check:")
    print("=" * 40)
    
    consistency_result = EnvironmentDiagnostics.validate_environment_consistency(FlextCoreSettings)
    
    if consistency_result.is_success:
        report = consistency_result.data
        print(f"Environment: {report['environment']}")
        print(f"Debug Mode: {report['debug_mode']}")
        print(f"Log Level: {report['log_level']}")
        
        if report["issues"]:
            print("\nIssues Found:")
            for issue in report["issues"]:
                print(f"  - {issue}")
        else:
            print("\nNo issues found.")
    else:
        print(f"Consistency check failed: {consistency_result.error}")
```

### Issue 4: Performance Degradation

**Problem**: FLEXT Core operations becoming slow over time.

**Solution**: Performance monitoring and optimization.

```python
import time
import functools
from typing import Callable, Any
from flext_core import FlextResult

class PerformanceMonitor:
    """Monitor FLEXT Core performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def time_operation(self, operation_name: str):
        """Decorator to time operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    if isinstance(result, FlextResult):
                        success = result.is_success
                except Exception as e:
                    result = e
                    success = False
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    # Record metrics
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = {
                            "total_calls": 0,
                            "successful_calls": 0,
                            "total_time": 0.0,
                            "min_time": float('inf'),
                            "max_time": 0.0,
                            "avg_time": 0.0
                        }
                    
                    metrics = self.metrics[operation_name]
                    metrics["total_calls"] += 1
                    if success:
                        metrics["successful_calls"] += 1
                    
                    metrics["total_time"] += execution_time
                    metrics["min_time"] = min(metrics["min_time"], execution_time)
                    metrics["max_time"] = max(metrics["max_time"], execution_time)
                    metrics["avg_time"] = metrics["total_time"] / metrics["total_calls"]
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "total_operations": len(self.metrics),
            "operations": {}
        }
        
        for operation, metrics in self.metrics.items():
            success_rate = (metrics["successful_calls"] / metrics["total_calls"]) * 100
            
            report["operations"][operation] = {
                **metrics,
                "success_rate_percent": success_rate,
                "calls_per_second": metrics["total_calls"] / metrics["total_time"] if metrics["total_time"] > 0 else 0
            }
        
        return report
    
    def identify_slow_operations(self, threshold_ms: float = 100.0) -> list[str]:
        """Identify operations slower than threshold."""
        slow_operations = []
        
        for operation, metrics in self.metrics.items():
            if metrics["avg_time"] * 1000 > threshold_ms:  # Convert to ms
                slow_operations.append(operation)
        
        return slow_operations

# Usage
monitor = PerformanceMonitor()

class OptimizedUserService:
    """User service with performance monitoring."""
    
    @monitor.time_operation("get_user")
    def get_user(self, user_id: str) -> FlextResult[User]:
        # Your implementation here
        return FlextResult.ok(User(id=user_id, name="Test User"))
    
    @monitor.time_operation("create_user")  
    def create_user(self, user_data: dict) -> FlextResult[User]:
        # Your implementation here
        return FlextResult.ok(User(**user_data))

# After running operations
performance_report = monitor.get_performance_report()
slow_operations = monitor.identify_slow_operations(threshold_ms=50.0)

if slow_operations:
    print(f"Slow operations detected: {slow_operations}")
    for op in slow_operations:
        metrics = performance_report["operations"][op]
        print(f"  {op}: avg {metrics['avg_time']*1000:.2f}ms, max {metrics['max_time']*1000:.2f}ms")
```

## 🔧 Advanced Debugging Techniques

### FlextResult Stack Traces

```python
import traceback
from flext_core import FlextResult

class DebugFlextResult(FlextResult[T]):
    """Enhanced FlextResult with stack trace information."""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.is_failure:
            self._stack_trace = traceback.format_stack()
    
    @classmethod
    def fail_with_trace(cls, error: str) -> "DebugFlextResult[T]":
        """Create failure result with stack trace."""
        result = cls(success=False, error=error, data=None)
        result._stack_trace = traceback.format_stack()
        return result
    
    def get_stack_trace(self) -> list[str] | None:
        """Get stack trace if available."""
        return getattr(self, '_stack_trace', None)
    
    def print_debug_info(self):
        """Print comprehensive debug information."""
        print(f"FlextResult Debug Info:")
        print(f"  Success: {self.success}")
        print(f"  Error: {self.error}")
        print(f"  Data Type: {type(self.data).__name__ if self.data else 'None'}")
        
        if hasattr(self, '_stack_trace'):
            print("  Stack Trace:")
            for line in self._stack_trace[-5:]:  # Last 5 stack frames
                print(f"    {line.strip()}")
```

### Container Dependency Visualization

```python
import json
from flext_core import get_flext_container

def visualize_container_dependencies():
    """Create a visual representation of container dependencies."""
    container = get_flext_container()
    services = container.list_services()
    service_info = container.get_service_info()
    
    # Create dependency graph
    dependency_graph = {
        "nodes": [],
        "edges": []
    }
    
    for service_name in services:
        service_result = container.get(service_name)
        
        node = {
            "id": service_name,
            "status": "healthy" if service_result.is_success else "failed",
            "type": type(service_result.data).__name__ if service_result.is_success else "unknown"
        }
        
        if service_result.is_failure:
            node["error"] = service_result.error
        
        dependency_graph["nodes"].append(node)
        
        # Analyze dependencies (this would require introspection)
        # For now, we'll add a placeholder for actual dependency analysis
        # In a real implementation, you'd analyze constructor parameters, etc.
    
    return dependency_graph

# Generate and save dependency graph
dep_graph = visualize_container_dependencies()
with open("container_dependencies.json", "w") as f:
    json.dump(dep_graph, f, indent=2)

print("Dependency graph saved to container_dependencies.json")
```

## 📊 Monitoring and Observability

### Custom Metrics Collection

```python
from typing import Dict, Any
import time
from collections import defaultdict

class FlextMetricsCollector:
    """Collect custom metrics for FLEXT Core operations."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = {}
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._make_key(name, tags or {})
        self.counters[key] += value
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        key = self._make_key(name, tags or {})
        self.histograms[key].append(value)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value."""
        key = self._make_key(name, tags or {})
        self.gauges[key] = value
    
    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create a metric key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        for key, values in self.histograms.items():
            if values:
                summary["histograms"][key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted(values)[len(values) // 2],
                    "p95": sorted(values)[int(len(values) * 0.95)],
                    "p99": sorted(values)[int(len(values) * 0.99)]
                }
        
        return summary

# Global metrics collector
metrics = FlextMetricsCollector()

# Usage in your code
def monitored_operation():
    start_time = time.perf_counter()
    
    try:
        # Your operation here
        result = some_flext_operation()
        
        if result.is_success:
            metrics.increment_counter("flext.operations.success", tags={"operation": "some_operation"})
        else:
            metrics.increment_counter("flext.operations.failure", tags={"operation": "some_operation"})
        
        return result
    
    finally:
        duration = time.perf_counter() - start_time
        metrics.record_histogram("flext.operations.duration", duration, tags={"operation": "some_operation"})

# Get metrics report
metrics_summary = metrics.get_metrics_summary()
print(json.dumps(metrics_summary, indent=2))
```

## 🚀 Performance Optimization

### Optimization Strategies

1. **Container Optimization**:
   - Use singleton pattern for expensive services
   - Implement lazy initialization
   - Clear unused services periodically

2. **FlextResult Optimization**:
   - Avoid deep nesting of flat_map operations
   - Use early returns when possible
   - Cache successful results when appropriate

3. **Configuration Optimization**:
   - Load configuration once at startup
   - Use environment variable caching
   - Validate configuration early

4. **Memory Optimization**:
   - Use weak references for circular dependencies
   - Implement proper cleanup in long-running services
   - Monitor memory usage regularly

This troubleshooting guide provides comprehensive solutions for advanced FLEXT Core issues, helping developers maintain robust and performant applications.
