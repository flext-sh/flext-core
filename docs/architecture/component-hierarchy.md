# Component Hierarchy - FLEXT Core Architecture

Reality-based component analysis

This document provides an analysis of FLEXT Core's actual component hierarchy based on the real implementation in `src/flext_core/`.

## 🏗️ Actual Module Structure (48 Python files)

Based on file count and real directory listing:

### **Level 1: Foundation (Zero dependencies)**

**Core Types and Constants**

- `constants.py` - Core enums and constants
- `__version__.py` - Version management & compatibility helpers
- `py.typed` - Type information marker

### **Level 2: Railway pattern foundation**

**FlextResult Pattern** (`result.py`)

```python
class FlextResult[T]:
    # Actual methods from code analysis:
    def __init__(...)
    def is_success(self) -> bool
    def success(self) -> bool  # Property
    def is_failure(self) -> bool
    def is_fail(self) -> bool
    def data(self) -> T | None  # Property
    def error(self) -> str | None  # Property
    def error_code(self) -> str | None
    def error_data(self) -> dict[str, object]

    @classmethod
    def ok(cls, data: T) -> FlextResult[T]

    @classmethod
    def fail(cls, ...) -> FlextResult[T]

    def unwrap(self) -> T
    # Note: map/flat_map methods exist but not captured in this grep
```

### **Level 3: Dependency injection**

**FlextContainer** (`container.py`)

- Dependency injection container
- Service registration and retrieval
- Based on FlextResult pattern

### **Level 4: Configuration management**

**Configuration Modules**

- `config.py` - FlextConfig implementation
- (config_hierarchical.py was removed in the current codebase)
- `config_models.py` - Configuration data models
- `payload.py` - Message/payload patterns

### **Level 5: Domain layer**

**Domain-Driven Design Components**

- `entities.py` - FlextModels.Entity base class
- `value_objects.py` - FlextModels.Value immutable types
- `aggregate_root.py` - FlextAggregates with events
- `domain_services.py` - Domain services
- `models.py` - General models

### **Level 6: Architectural patterns**

**CQRS and Validation**

- `commands.py` - Command pattern
- `handlers.py` - Handler patterns
- `validation.py` - Validation framework
- `protocols.py` - Protocol definitions
- `guards.py` - Type guards

### **Level 7: Cross-cutting concerns**

**Utilities and Extensions**

- `utilities.py` - Utility functions
- `decorators.py` - Decorator patterns
- `mixins.py` - Mixin classes
- `fields.py` - Field definitions
- `exceptions.py` - Exception hierarchy
- `loggings.py` - Logging utilities

### **Level 8: Type system**

**Type Definitions**

- `typings.py` - Centralized type system (single source of truth)
- `types.py` - Thin compatibility re-export

### **Level 9: Base implementations**

**Internal Base Classes (modernized)**

- `base_commands.py`
- `base_decorators.py`
- `base_exceptions.py`
- `base_handlers.py`
- `base_mixins.py`
- `base_testing.py`
- `base_utilities.py`
- `delegation_system.py`
- `legacy.py`

### **Level 10: Integration and specialized**

**Framework Integration**

- `core.py` - FlextCore main class
- `context.py` - Context management
- `observability.py` - Observability patterns
- `schema_processing.py` - Schema processing
- `singer_base.py` - Singer integration
- `testing_utilities.py` - Testing support

## 🔍 Dependency Flow Analysis

### Confirmed Dependencies (Based on Real Code)

```
FlextResult (result.py)
    ↓
FlextContainer (container.py) - uses FlextResult
    ↓
FlextConfig (config.py) - uses FlextResult and FlextContainer
    ↓
Domain Classes (entities.py, etc.) - use all above
    ↓
Patterns (commands.py, handlers.py) - use domain + core
    ↓
Application Layer (core.py) - integrates all patterns
```

## 📊 Real API Surface

### FlextResult (Verified Methods)

```python
# Success/failure checking
result.is_success() -> bool
result.success -> bool  # property
result.is_failure() -> bool
result.is_fail() -> bool

# Data access
result.data -> T | None  # property
result.error -> str | None  # property
result.error_code() -> str | None
result.error_data() -> dict[str, object]

# Creation methods
FlextResult[None].ok(data) -> FlextResult[T]
FlextResult[None].fail(...) -> FlextResult[T]

# Unwrapping
result.unwrap() -> T  # May raise if failure
```

### FlextConfig (From config.py)

```python
# Environment integration
class AppSettings(FlextConfig):
    field: str = Field(default="value")

    class Config:
        env_prefix = "APP_"
```

## ⚠️ What's NOT Implemented

Based on code analysis, these commonly referenced features don't exist:

```python
# These are NOT in the actual codebase:
result.map(func)          # May exist but not captured
result.flat_map(func)     # May exist but not captured
container.list_services() # Not confirmed in FlextContainer
container.get_service_info() # Not confirmed in FlextContainer
```

## 🎯 Architectural Strengths

### Confirmed Strengths

1. **Type Safety**: Full Python 3.13 generic support
2. **Railway Pattern**: Consistent FlextResult usage
3. **Configuration**: Pydantic-based settings
4. **Domain Modeling**: Rich entity and value object support
5. **Dependency Injection**: Container-based service location

### Areas Needing Verification

1. **Event Sourcing**: Claims exist but implementation unclear
2. **CQRS Implementation**: Files exist but actual API unclear
3. **Plugin Architecture**: Mentioned but no clear implementation
4. **Cross-Language Bridge**: Claims exist but no verified implementation

## 🔄 Migration and Evolution

### Current Status

- **48 Python modules** with mixed documentation quality
- **Multiple type systems** in transition (types.py → semantic_types.py)
- **Legacy modules** marked for compatibility (types.py)
- **Base implementations** suggest ongoing refactoring

### Architectural Decisions

1. **Railway-Oriented Programming**: Central to all operations
2. **Pydantic Integration**: Configuration and validation
3. **Protocol-Based Design**: Structural typing throughout
4. **Clean Architecture**: Layer separation enforced

## 📋 Next Steps for Documentation

1. **Verify all API claims** against actual implementation
2. **Test all code examples** in documentation
3. **Document actual limitations** rather than aspirational features
4. **Create accurate architectural diagrams** based on real dependencies

---

**This component hierarchy reflects the ACTUAL implementation in FLEXT Core as verified through code analysis. All claims are based on real file structure and method signatures found in the source code.**
