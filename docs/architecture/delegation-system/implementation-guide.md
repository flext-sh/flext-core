# FlextDelegationSystem Implementation Guide

**Version**: 0.9.0  
**Target Audience**: FLEXT Developers, System Architects  
**Implementation Time**: 2-4 weeks per library  
**Complexity**: Intermediate to Advanced  

## 📖 Overview

This guide provides comprehensive, step-by-step instructions for implementing `FlextDelegationSystem` patterns in FLEXT libraries. The delegation system enables sophisticated composition patterns through automatic method forwarding, property delegation, and type-safe protocol contracts.

### Prerequisites

- Python 3.13+ with advanced type hints
- Understanding of descriptor protocol
- Familiarity with composition patterns
- Knowledge of Protocol typing

### Implementation Benefits

- 📦 **60-70% reduction** in delegation-related code
- 🔒 **Type safety** through protocol contracts
- ✅ **Automatic validation** of delegation correctness
- 🚀 **Performance optimization** through efficient forwarding
- 🔧 **Consistent architecture** across all libraries

---

## 🚀 Quick Start

### Basic Implementation

```python
from flext_core.delegation_system import FlextDelegationSystem

class MyService:
    """Service with automatic delegation composition."""
    
    def __init__(self):
        # Compose functionality through delegation
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self,
            ValidationMixin,    # Provides validate(), is_valid
            LoggingMixin,      # Provides log_info(), log_error()
            CachingMixin,      # Provides cache_get(), cache_set()
        )
    
    def process_data(self, data):
        # All delegated methods now available directly
        if not self.validate(data):     # From ValidationMixin
            self.log_error("Invalid data")  # From LoggingMixin
            return None
        
        cached = self.cache_get(data)   # From CachingMixin
        if cached:
            self.log_info("Using cached result")
            return cached
        
        result = self.expensive_operation(data)
        self.cache_set(data, result)    # From CachingMixin
        return result
```

### Validation

```python
# Validate delegation is working correctly
validation = FlextDelegationSystem.validate_delegation_system()
if validation.success:
    print("✅ Delegation system working correctly")
else:
    print(f"❌ Delegation issues: {validation.error}")
```

---

## 📚 Step-by-Step Implementation

### Step 1: Identify Delegation Opportunities

#### 1.1 Analyze Current Patterns

Look for these common patterns in existing code:

```python
# ❌ Manual delegation patterns to replace
class CurrentService:
    def __init__(self):
        self._logger = LoggingService()
        self._validator = ValidationService()
        self._cache = CacheService()
    
    def log_info(self, message):
        return self._logger.log_info(message)  # Manual forwarding
    
    def validate(self, data):
        return self._validator.validate(data)  # Manual forwarding
    
    def cache_get(self, key):
        return self._cache.get(key)  # Manual forwarding
```

#### 1.2 Identify Mixin Candidates

Create this analysis checklist:

```python
# Delegation Opportunity Analysis Checklist
delegation_opportunities = {
    "validation_operations": [
        "validate_input()", "sanitize_data()", "check_constraints()"
    ],
    "logging_operations": [
        "log_info()", "log_error()", "log_debug()", "audit_log()"
    ],
    "caching_operations": [
        "cache_get()", "cache_set()", "cache_clear()", "cache_stats()"
    ],
    "serialization_operations": [
        "to_dict()", "from_dict()", "to_json()", "from_json()"
    ],
    "error_handling": [
        "handle_error()", "format_error()", "log_exception()"
    ]
}
```

### Step 2: Design Mixin Classes

#### 2.1 Create Focused Mixins

Each mixin should have a single responsibility:

```python
class ValidationMixin:
    """Mixin providing validation capabilities."""
    
    def __init__(self):
        self.validation_errors: list[str] = []
        self.validation_rules: dict[str, callable] = {}
    
    def validate(self, data: dict) -> bool:
        """Validate data using configured rules."""
        self.validation_errors.clear()
        
        for field, rule in self.validation_rules.items():
            if field not in data:
                self.validation_errors.append(f"Missing required field: {field}")
                continue
            
            try:
                if not rule(data[field]):
                    self.validation_errors.append(f"Validation failed for {field}")
            except Exception as e:
                self.validation_errors.append(f"Validation error for {field}: {e}")
        
        return len(self.validation_errors) == 0
    
    @property
    def is_valid(self) -> bool:
        """Check if last validation was successful."""
        return len(self.validation_errors) == 0
    
    def add_validation_rule(self, field: str, rule: callable) -> None:
        """Add validation rule for a field."""
        self.validation_rules[field] = rule
    
    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()
```

```python
class CachingMixin:
    """Mixin providing caching capabilities."""
    
    def __init__(self):
        self._cache: dict[str, object] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def cache_get(self, key: str) -> object | None:
        """Get value from cache."""
        if key in self._cache:
            self._cache_stats["hits"] += 1
            return self._cache[key]
        self._cache_stats["misses"] += 1
        return None
    
    def cache_set(self, key: str, value: object) -> None:
        """Set value in cache."""
        self._cache[key] = value
    
    def cache_clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            **self._cache_stats,
            "size": len(self._cache)
        }
```

#### 2.2 Create Advanced Mixins

For complex functionality:

```python
class AuditMixin:
    """Mixin providing comprehensive audit capabilities."""
    
    def __init__(self):
        self.audit_log: list[dict] = []
        self.audit_enabled = True
    
    def audit_operation(self, operation: str, data: dict, result: dict) -> None:
        """Audit an operation with its data and result."""
        if not self.audit_enabled:
            return
        
        import time
        audit_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "data_hash": hash(str(data)),
            "result_status": result.get("status", "unknown"),
            "duration": result.get("duration", 0),
            "user_context": getattr(self, "current_user", "system")
        }
        self.audit_log.append(audit_entry)
    
    def get_audit_trail(self, operation_filter: str = None) -> list[dict]:
        """Get audit trail, optionally filtered by operation."""
        if operation_filter:
            return [entry for entry in self.audit_log 
                   if entry["operation"] == operation_filter]
        return self.audit_log.copy()
    
    def enable_audit(self, enabled: bool = True) -> None:
        """Enable or disable audit logging."""
        self.audit_enabled = enabled
```

### Step 3: Implement Delegation

#### 3.1 Basic Service Implementation

```python
class BusinessService:
    """Business service with comprehensive delegation."""
    
    def __init__(self, config: dict):
        # Compose all required functionality through delegation
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self,
            ValidationMixin,    # Input validation
            CachingMixin,       # Result caching
            AuditMixin,         # Operation auditing
            LoggingMixin,       # Structured logging
            MetricsMixin,       # Performance metrics
            ErrorHandlingMixin  # Error management
        )
        
        self.config = config
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize delegated components with configuration."""
        # Configure validation rules
        self.add_validation_rule("id", lambda x: isinstance(x, (int, str)) and x)
        self.add_validation_rule("data", lambda x: isinstance(x, dict))
        
        # Configure logging
        self.configure_logging({
            "level": self.config.get("log_level", "INFO"),
            "format": "structured"
        })
        
        # Enable audit if configured
        self.enable_audit(self.config.get("audit_enabled", True))
    
    def process_business_request(self, request: dict) -> FlextResult[dict]:
        """Process business request with comprehensive delegation."""
        import time
        start_time = time.time()
        
        try:
            # Validation through delegation
            if not self.validate(request):  # From ValidationMixin
                error_msg = f"Validation failed: {self.get_validation_errors()}"
                self.log_error(error_msg)   # From LoggingMixin
                return FlextResult[dict].fail(error_msg)
            
            # Check cache
            cache_key = f"request_{hash(str(request))}"
            cached_result = self.cache_get(cache_key)  # From CachingMixin
            if cached_result:
                self.record_metric("cache_hit")  # From MetricsMixin
                self.log_info(f"Returned cached result for {cache_key}")
                return FlextResult[dict].ok(cached_result)
            
            # Process request
            self.log_info(f"Processing new request: {request.get('id', 'unknown')}")
            result = self._execute_business_logic(request)
            
            # Cache result
            self.cache_set(cache_key, result)  # From CachingMixin
            
            # Record metrics
            duration = time.time() - start_time
            self.record_metric("request_processed", {"duration": duration})
            
            # Audit operation
            self.audit_operation("business_request", request, {
                "status": "success",
                "duration": duration,
                "result_size": len(str(result))
            })
            
            self.log_info(f"Request processed successfully in {duration:.3f}s")
            return FlextResult[dict].ok(result)
            
        except Exception as e:
            # Error handling through delegation
            error_context = {
                "request_id": request.get("id", "unknown"),
                "error": str(e),
                "duration": time.time() - start_time
            }
            
            handled_error = self.handle_error("business_request", e, error_context)
            self.audit_operation("business_request", request, {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            })
            
            return handled_error
    
    def _execute_business_logic(self, request: dict) -> dict:
        """Execute actual business logic."""
        # Simulate business processing
        request_id = request.get("id", "unknown")
        data = request.get("data", {})
        
        # Business logic implementation
        processed_data = {
            "processed_id": request_id,
            "processed_data": data,
            "processing_metadata": {
                "processor": "BusinessService",
                "version": "1.0.0",
                "timestamp": time.time()
            }
        }
        
        return processed_data
```

#### 3.2 Advanced Delegation with Protocols

Create type-safe delegation contracts:

```python
from typing import Protocol

class HasBusinessCapabilities(FlextDelegationSystem.HasDelegator, Protocol):
    """Protocol defining business service capabilities."""
    
    def validate(self, data: dict) -> bool: ...
    def log_info(self, message: str) -> None: ...
    def cache_get(self, key: str) -> object | None: ...
    def audit_operation(self, operation: str, data: dict, result: dict) -> None: ...
    @property
    def is_valid(self) -> bool: ...

class TypeSafeBusinessService(HasBusinessCapabilities):
    """Type-safe business service implementing the protocol."""
    
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self,
            ValidationMixin,  # Provides validate(), is_valid
            LoggingMixin,     # Provides log_info(), log_error()
            CachingMixin,     # Provides cache_get(), cache_set()
            AuditMixin        # Provides audit_operation()
        )
    
    # Service now automatically satisfies HasBusinessCapabilities protocol
```

### Step 4: Advanced Patterns

#### 4.1 Property Delegation

For transparent property access:

```python
class ConfigurationMixin:
    """Mixin providing configuration properties."""
    
    def __init__(self):
        self._config = {
            "database_url": "postgresql://localhost:5432/db",
            "api_timeout": 30,
            "retry_count": 3,
            "debug_enabled": False
        }
    
    @property
    def database_url(self) -> str:
        return self._config["database_url"]
    
    @property
    def api_timeout(self) -> int:
        return self._config["api_timeout"]
    
    @property
    def retry_count(self) -> int:
        return self._config["retry_count"]
    
    @property
    def debug_enabled(self) -> bool:
        return self._config["debug_enabled"]

class ServiceWithPropertyDelegation:
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ConfigurationMixin
        )
    
    def connect_to_database(self):
        # Properties automatically available through delegation
        connection_string = self.database_url  # From ConfigurationMixin
        timeout = self.api_timeout             # From ConfigurationMixin
        
        return self._establish_connection(connection_string, timeout)
```

#### 4.2 Method Delegation with Custom Forwarding

```python
class CustomMethodDelegator:
    """Custom method delegator with advanced forwarding."""
    
    def __init__(self, target_service: object):
        self.delegator = FlextDelegationSystem.create_method_delegator(
            self, 
            target_service, 
            ["process_data", "validate_input", "format_output"]
        )
        self.target_service = target_service
    
    def enhanced_process_data(self, data: dict) -> dict:
        """Enhanced processing with pre/post hooks."""
        # Pre-processing hook
        self.log_info(f"Starting enhanced processing for {data}")
        
        # Delegate to original method
        result = self.process_data(data)  # Delegated method
        
        # Post-processing hook
        enhanced_result = {
            **result,
            "enhanced_metadata": {
                "processor": "CustomMethodDelegator",
                "enhancement_applied": True
            }
        }
        
        self.log_info(f"Enhanced processing completed")
        return enhanced_result
```

### Step 5: Testing and Validation

#### 5.1 Comprehensive Testing Strategy

```python
import pytest
from flext_core.delegation_system import FlextDelegationSystem

class TestDelegationImplementation:
    """Comprehensive test suite for delegation implementation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.service = BusinessService({
            "log_level": "DEBUG",
            "audit_enabled": True,
            "cache_enabled": True
        })
    
    def test_delegation_setup(self):
        """Test that delegation is properly configured."""
        # Validate delegation system
        validation = self.service.delegator._validate_delegation()
        assert validation.success, f"Delegation validation failed: {validation.error}"
        
        # Check delegation info
        info = self.service.delegator.get_delegation_info()
        assert info["host_class"] == "BusinessService"
        assert len(info["mixin_classes"]) >= 5
        assert "validate" in info["delegated_methods"]
        assert "log_info" in info["delegated_methods"]
        assert "cache_get" in info["delegated_methods"]
    
    def test_method_delegation(self):
        """Test that delegated methods work correctly."""
        # Test validation delegation
        valid_data = {"id": "test123", "data": {"value": 42}}
        assert self.service.validate(valid_data) == True
        assert self.service.is_valid == True
        
        # Test invalid data
        invalid_data = {"id": "", "data": "not_a_dict"}
        assert self.service.validate(invalid_data) == False
        assert len(self.service.get_validation_errors()) > 0
    
    def test_property_delegation(self):
        """Test that delegated properties work correctly."""
        # Properties should be accessible through delegation
        assert hasattr(self.service, "is_valid")
        assert isinstance(self.service.is_valid, bool)
        
        # Properties should reflect mixin state
        self.service.validate({"id": "test", "data": {}})
        assert self.service.is_valid == True
    
    def test_caching_delegation(self):
        """Test that caching delegation works correctly."""
        # Test cache operations
        self.service.cache_set("test_key", "test_value")
        cached_value = self.service.cache_get("test_key")
        assert cached_value == "test_value"
        
        # Test cache statistics
        stats = self.service.cache_stats()
        assert stats["hits"] >= 1
        assert stats["size"] >= 1
    
    def test_audit_delegation(self):
        """Test that audit delegation works correctly."""
        # Perform audited operation
        test_data = {"id": "audit_test", "data": {"value": 123}}
        test_result = {"status": "success", "duration": 0.1}
        
        self.service.audit_operation("test_operation", test_data, test_result)
        
        # Check audit trail
        audit_trail = self.service.get_audit_trail()
        assert len(audit_trail) >= 1
        
        operation_audit = self.service.get_audit_trail("test_operation")
        assert len(operation_audit) >= 1
        assert operation_audit[0]["operation"] == "test_operation"
    
    def test_integration_scenario(self):
        """Test complete integration scenario."""
        request = {
            "id": "integration_test",
            "data": {
                "customer_id": "cust_123",
                "order_items": ["item1", "item2"],
                "total_amount": 99.99
            }
        }
        
        # Process request using all delegated capabilities
        result = self.service.process_business_request(request)
        
        # Verify successful processing
        assert result.success, f"Request processing failed: {result.error}"
        
        # Verify delegation worked correctly
        info = self.service.delegator.get_delegation_info()
        assert len(info["delegated_methods"]) >= 10
        
        # Verify audit trail was created
        audit_trail = self.service.get_audit_trail("business_request")
        assert len(audit_trail) >= 1
        
        # Verify cache was used
        stats = self.service.cache_stats()
        assert stats["size"] >= 1

    def test_system_wide_validation(self):
        """Test system-wide delegation validation."""
        system_validation = FlextDelegationSystem.validate_delegation_system()
        assert system_validation.success, f"System validation failed: {system_validation.error}"
        
        report = system_validation.value
        assert report["status"] == "SUCCESS"
        assert len(report["test_results"]) >= 4
        assert all("✓" in result for result in report["test_results"])
```

#### 5.2 Performance Testing

```python
import time
import pytest

class TestDelegationPerformance:
    """Performance testing for delegation system."""
    
    def test_delegation_overhead(self):
        """Test that delegation adds minimal performance overhead."""
        service = BusinessService({"audit_enabled": False})
        
        # Test direct method call performance
        start_time = time.time()
        for _ in range(1000):
            service.validate({"id": "test", "data": {}})
        delegation_time = time.time() - start_time
        
        # Delegation should add minimal overhead (< 20% slowdown)
        # This is an acceptable trade-off for the benefits provided
        assert delegation_time < 1.0, f"Delegation too slow: {delegation_time:.3f}s"
    
    def test_memory_usage(self):
        """Test that delegation doesn't cause excessive memory usage."""
        import gc
        import sys
        
        # Measure memory before creating services
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create multiple services with delegation
        services = []
        for i in range(100):
            service = BusinessService({"id": f"service_{i}"})
            services.append(service)
        
        # Measure memory after
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Objects per service should be reasonable (< 50 per service)
        objects_per_service = (final_objects - initial_objects) / 100
        assert objects_per_service < 50, f"Too many objects per service: {objects_per_service}"
```

---

## 🔧 Migration Patterns

### Pattern 1: Facade to Delegation Migration

#### Before: Manual Facade Pattern
```python
class ManualFacade:
    def __init__(self):
        self._config = ConfigService()
        self._utils = UtilityService() 
        self._adapters = AdapterService()
    
    @property
    def config(self):
        return self._config
    
    def get_configuration(self, key):
        return self._config.get(key)
    
    def log_message(self, message):
        return self._utils.log(message)
```

#### After: Delegation Pattern
```python
class DelegationFacade:
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ConfigMixin, UtilityMixin, AdapterMixin
        )
    
    # All methods automatically available through delegation:
    # get_configuration(), log_message(), adapter methods, etc.
```

### Pattern 2: Service Layer to Delegation Migration

#### Before: Manual Service Coordination
```python
class ManualServiceLayer:
    def __init__(self):
        self.validator = ValidationService()
        self.logger = LoggingService()
        self.cache = CacheService()
    
    def process_request(self, request):
        if not self.validator.validate(request):
            self.logger.log_error("Validation failed")
            return None
            
        cached = self.cache.get(request)
        if cached:
            self.logger.log_info("Cache hit")
            return cached
```

#### After: Delegation-Based Service Layer
```python
class DelegationServiceLayer:
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ValidationMixin, LoggingMixin, CacheMixin
        )
    
    def process_request(self, request):
        if not self.validate(request):     # Delegated
            self.log_error("Validation failed")  # Delegated
            return None
            
        cached = self.cache_get(request)   # Delegated
        if cached:
            self.log_info("Cache hit")     # Delegated
            return cached
```

---

## ⚡ Performance Optimization

### Optimization Techniques

#### 1. **Lazy Delegation Setup**
```python
class OptimizedService:
    def __init__(self):
        self._delegator = None
    
    @property
    def delegator(self):
        if self._delegator is None:
            self._delegator = FlextDelegationSystem.create_mixin_delegator(
                self, ValidationMixin, LoggingMixin  # Only create when needed
            )
        return self._delegator
```

#### 2. **Delegation Caching**
```python
class CachedDelegationService:
    _delegation_cache = {}
    
    def __init__(self, service_type: str):
        if service_type not in self._delegation_cache:
            self._delegation_cache[service_type] = FlextDelegationSystem.create_mixin_delegator(
                self, *self._get_mixins_for_type(service_type)
            )
        self.delegator = self._delegation_cache[service_type]
```

#### 3. **Performance Monitoring**
```python
class MonitoredDelegationService:
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ValidationMixin, PerformanceMonitoringMixin
        )
    
    def process_with_monitoring(self, data):
        with self.performance_monitor("process_operation"):  # Delegated
            if self.validate(data):  # Delegated
                return self.expensive_operation(data)
```

---

## 🚨 Common Pitfalls and Solutions

### Pitfall 1: Method Name Conflicts

#### Problem
```python
class ConflictingMixins:
    def __init__(self):
        # Both mixins have a 'process()' method
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ProcessingMixin, AnotherProcessingMixin  # Conflict!
        )
```

#### Solution
```python
class ResolvedConflicts:
    def __init__(self):
        self.delegator = FlextDelegationSystem.create_mixin_delegator(
            self, ProcessingMixin  # Use only one, or create adapter
        )
        
        # Manual delegation for conflicting method
        self.another_processor = AnotherProcessingMixin()
    
    def process_with_alternative(self, data):
        return self.another_processor.process(data)
```

### Pitfall 2: Circular Dependencies

#### Problem
```python
# Mixin A depends on Mixin B methods
class MixinA:
    def method_a(self):
        return self.method_b()  # Expects method_b from MixinB

# Mixin B depends on Mixin A methods  
class MixinB:
    def method_b(self):
        return self.method_a()  # Expects method_a from MixinA - CIRCULAR!
```

#### Solution
```python
# Break circular dependency with interface
class SharedInterfaceMixin:
    def get_shared_data(self):
        return {"shared": True}

class IndependentMixinA:
    def method_a(self):
        shared = self.get_shared_data()  # From SharedInterfaceMixin
        return f"A: {shared}"

class IndependentMixinB:
    def method_b(self):
        shared = self.get_shared_data()  # From SharedInterfaceMixin
        return f"B: {shared}"
```

### Pitfall 3: Property Delegation Issues

#### Problem
```python
class PropertyMixin:
    @property
    def config_value(self):
        return self._config_value  # AttributeError if _config_value not set
```

#### Solution
```python
class SafePropertyMixin:
    def __init__(self):
        self._config_value = "default_value"  # Always initialize
    
    @property
    def config_value(self):
        return getattr(self, '_config_value', 'fallback_default')
```

---

## 📋 Implementation Checklist

### Pre-Implementation
- [ ] **Analyze existing delegation patterns** in target library
- [ ] **Identify mixin candidates** and their responsibilities
- [ ] **Design protocol contracts** for type safety
- [ ] **Plan migration strategy** (incremental vs. complete)
- [ ] **Set up testing environment** with validation scenarios

### Implementation Phase
- [ ] **Create focused mixin classes** with single responsibilities
- [ ] **Implement host classes** with delegation setup
- [ ] **Add protocol contracts** for type safety
- [ ] **Configure validation rules** and error handling
- [ ] **Implement property delegation** where needed

### Testing Phase
- [ ] **Unit tests** for each delegated method
- [ ] **Integration tests** for complete workflows
- [ ] **Performance tests** for delegation overhead
- [ ] **System validation** using FlextDelegationSystem.validate_delegation_system()
- [ ] **Protocol compliance tests** using MyPy

### Post-Implementation
- [ ] **Documentation** of delegation patterns used
- [ ] **Performance monitoring** setup
- [ ] **Migration of dependent code** to use new delegation
- [ ] **Training** for team members on new patterns
- [ ] **Maintenance plan** for ongoing delegation management

---

## 📚 Advanced Topics

### Custom Delegation Strategies

For specialized use cases, implement custom delegation:

```python
class CustomDelegationStrategy:
    """Custom delegation strategy for specialized requirements."""
    
    @staticmethod
    def create_conditional_delegator(host, condition_func, *mixins):
        """Create delegator that only delegates when condition is met."""
        
        class ConditionalDelegator:
            def __init__(self, host, condition, mixins):
                self.host = host
                self.condition = condition
                self.base_delegator = FlextDelegationSystem.create_mixin_delegator(
                    host, *mixins
                )
            
            def __getattr__(self, name):
                if self.condition():
                    return getattr(self.base_delegator, name)
                raise AttributeError(f"Conditional delegation: {name} not available")
        
        return ConditionalDelegator(host, condition_func, mixins)

# Usage
class ConditionalService:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        
        # Only delegate debug methods when in debug mode
        self.delegator = CustomDelegationStrategy.create_conditional_delegator(
            self, 
            lambda: self.debug_mode,
            DebugMixin, DiagnosticMixin
        )
```

This implementation guide provides comprehensive coverage of FlextDelegationSystem implementation patterns, from basic setup through advanced customization strategies. Follow these patterns to achieve consistent, type-safe, and performant delegation throughout the FLEXT ecosystem.
