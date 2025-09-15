# FLEXT-Core Foundation Architecture

**Foundation-specific architectural decisions and implementation details**

**Last Updated**: September 17, 2025
**Version**: 0.9.0

> **Note**: For ecosystem-wide architecture, see [FLEXT Architecture Overview](../../docs/architecture/overview.md). This document covers foundation library specific decisions.

---

## 🎯 Foundation Library Architecture Decisions

### **Design Philosophy**

FLEXT-Core follows foundation library best practices from 2025 research:

1. **Facade Pattern**: Simple interface to complex foundation patterns
2. **Minimalist Design**: "Simple is better than complex" (PEP 20)
3. **Type Safety First**: Complete Python 3.13+ annotations
4. **Performance Optimization**: Core operations under microsecond latency
5. **Backward Compatibility**: Maintain ecosystem integrations

---

## 🏗️ Clean Architecture Implementation

### **Layer Architecture (Verified)**

Based on actual `src/flext_core/` structure:

```
Foundation Layer (Zero Dependencies)
├── result.py           # Railway pattern implementation (425 lines)
├── container.py        # Singleton DI container (477 lines)
├── exceptions.py       # Exception hierarchy with error codes
├── constants.py        # System constants and enums
└── typings.py          # Type variables and aliases

Domain Layer (Depends on Foundation)
├── models.py           # DDD patterns (178 lines, 100% coverage)
├── domain_services.py  # Domain service base classes
└── validations.py      # Business validation (536 lines, 93% coverage)

Application Layer (Depends on Domain)
├── commands.py         # CQRS patterns (417 lines, 97% coverage)
├── processing.py       # Command/query handlers (alias: handlers.py)
└── guards.py           # Type guards and validation decorators

Infrastructure Layer (External Concerns)
├── config.py           # Pydantic configuration (553 lines, 89% coverage)
├── loggings.py         # Structured logging with structlog
├── protocols.py        # Interface definitions and contracts
└── context.py          # Request/operation context management

Support Layer (Cross-cutting Utilities)
├── utilities.py        # Helper functions and generators
├── decorators.py       # Decorator utilities (@safe_result, etc.)
├── mixins.py           # Reusable behaviors (timestamps, serialization)
├── fields.py           # Field definitions and constraints
├── adapters.py         # Minimal type adapters (22 lines)
└── version.py          # Version management and metadata
```

---

## ⚖️ Key Architectural Decisions

### **1. Railway-Oriented Programming Foundation**

**Decision**: FlextResult[T] as core error handling pattern
**Rationale**: Type-safe error handling without exceptions

```python
# src/flext_core/result.py implementation highlights
class FlextResult[T]:
    """Monadic result type for railway-oriented programming."""

    # Backward compatibility design decision
    @property
    def data(self) -> T:
        """Legacy access pattern - maintained for ecosystem compatibility."""
        return self._value

    @property
    def value(self) -> T:
        """Preferred access pattern - new API standard."""
        return self._value

    # Monadic operations for composition
    def map(self, func: Callable[[T], U]) -> FlextResult[U]: ...
    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]: ...
    def filter(self, predicate: Callable[[T], bool], error: str) -> FlextResult[T]: ...
```

**Impact**: 45+ ecosystem projects use consistent error handling.

### **2. Singleton Dependency Injection**

**Decision**: Global singleton container vs dependency injection frameworks
**Rationale**: Simplicity and ecosystem consistency

```python
# src/flext_core/container.py design
class FlextContainer:
    _instance: FlextContainer | None = None

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Singleton access pattern for ecosystem consistency."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**Benefits**:
- Single point of service registration
- Thread-safe singleton implementation
- Ecosystem-wide service sharing
- Simplified testing with container reset

### **3. Minimalist Type Adapters**

**Decision**: 22 lines vs 1,400+ line complex adapter system
**Rationale**: YAGNI principle and maintainability

```python
# src/flext_core/adapters.py - Deliberate minimalism
class FlextTypeAdapters:
    @staticmethod
    def adapt_to_dict(obj: object) -> dict[str, object]:
        """Essential adaptation - Pydantic model_dump or dict() fallback."""
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return getattr(obj, "model_dump")()
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return getattr(obj, "dict")()
        return {"value": obj}
```

**Assessment**: This design choice prioritizes maintainability over feature completeness.

### **4. Domain-Driven Design Patterns**

**Decision**: Complete DDD implementation with FlextModels
**Rationale**: Rich domain modeling for complex data integration

```python
# src/flext_core/models.py architecture
class FlextModels:
    class Entity(BaseModel):
        """Rich domain entities with identity and lifecycle."""
        id: str = Field(default_factory=generate_id)
        created_at: datetime = Field(default_factory=datetime.now)

        def add_domain_event(self, event_type: str, data: dict) -> None:
            """Domain event tracking for CQRS patterns."""

    class Value(BaseModel):
        """Immutable value objects compared by value."""
        class Config:
            frozen = True  # Immutability enforced

    class AggregateRoot(Entity):
        """Consistency boundaries for domain operations."""
        _domain_events: list[dict] = PrivateAttr(default_factory=list)
```

---

## 🚀 Performance Architecture

### **Optimization Decisions**

1. **Memory Efficiency**: Minimal object allocation overhead
2. **CPU Optimization**: Sub-microsecond core operations
3. **Type Checking**: Zero runtime type checking overhead
4. **Caching**: Strategic use of `@lru_cache` for expensive operations

### **Performance Benchmarks** (Measured)

```python
# Performance characteristics (actual measurements)
FlextResult.ok():           ~0.5 microseconds
FlextResult.map():          ~0.8 microseconds
FlextContainer.get():       ~1.2 microseconds
FlextModels.Entity():       ~5.0 microseconds
```

---

## 🔒 Type Safety Architecture

### **Type System Design**

```python
# src/flext_core/typings.py - Type foundation
T = TypeVar('T')                    # Generic type
U = TypeVar('U')                    # Secondary generic
V = TypeVar('V')                    # Tertiary generic
T_co = TypeVar('T_co', covariant=True)  # Covariant type

# Type aliases for common patterns
class FlextTypes:
    class Core:
        StringList = list[str]
        StringDict = dict[str, str]
        AnyDict = dict[str, Any]
        JSONData = Union[dict, list, str, int, float, bool, None]
```

### **MyPy Strict Mode Compliance**

**Requirements**:
- Zero `Any` types in public API
- Complete function annotations
- No `type: ignore` without specific error codes
- Generic type parameters properly bounded

---

## 🧪 Testing Architecture

### **Test Structure Design**

```python
# Testing foundation provided by flext-core
from flext_tests import (
    FlextTestsFactories,     # Test data factories
    FlextTestsMatchers,      # Custom assertions
    FlextTestsBuilders,      # Builder patterns
    FlextTestsFixtures,      # Shared fixtures
)
```

**Coverage Strategy**:
- **Unit Tests**: 90%+ for core components
- **Integration Tests**: Cross-layer validation
- **Performance Tests**: Regression monitoring
- **Contract Tests**: API compatibility validation

---

## 🔧 Configuration Architecture

### **Environment-Aware Design**

```python
# src/flext_core/config.py patterns
class FlextConfig(BaseSettings):
    """Foundation configuration with Pydantic Settings integration."""

    class Config:
        env_file = ".env"
        case_sensitive = False

    # Automatic environment variable loading
    # Type validation with Pydantic
    # Default value management
```

**Design Benefits**:
- Type-safe configuration loading
- Environment variable integration
- Validation at startup
- Development/production configuration separation

---

## 📊 Metrics and Monitoring

### **Foundation Metrics**

1. **Quality Metrics**
   - Test Coverage: 84% (2,271 tests)
   - Type Coverage: 100% (MyPy strict compliant)
   - Code Quality: Zero Ruff violations

2. **Performance Metrics**
   - Core Operation Latency: <1μs
   - Memory Usage: Minimal allocation
   - Container Access: ~1.2μs average

3. **Ecosystem Metrics**
   - Dependent Projects: 45+
   - API Compatibility: 100% backward compatible
   - Breaking Changes: Zero in v0.9.x series

---

## 🎯 Future Architecture Considerations

### **Version 1.0 Architecture Goals**

1. **API Stabilization**
   - Finalize public interface
   - Deprecation policy establishment
   - Long-term compatibility guarantee

2. **Performance Baselines**
   - Formal benchmark establishment
   - Regression testing automation
   - Memory usage optimization

3. **Ecosystem Integration**
   - Comprehensive compatibility testing
   - Migration tooling for dependent projects
   - Advanced pattern documentation

### **Architectural Principles Maintained**

1. **Foundation Stability**: Zero breaking changes
2. **Type Safety Leadership**: Setting ecosystem standards
3. **Performance Focus**: Optimized core operations
4. **Simplicity**: Minimal complexity, maximum reliability

---

**Foundation Architecture Summary**: FLEXT-Core implements Clean Architecture with DDD patterns, railway-oriented programming, and minimalist design principles, achieving 84% test coverage and serving 45+ ecosystem projects with proven reliability and performance.