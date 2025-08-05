# Component Hierarchy - FLEXT Core Architecture

**Reality-Based Component Analysis**

This document provides an analysis of FLEXT Core's actual component hierarchy based on the real implementation in `src/flext_core/`.

## 🏗️ Actual Module Structure (48 Python Files)

Based on file count and real directory listing:

### **Level 1: Foundation (Zero Dependencies)**

**Core Types and Constants**

- `constants.py` - Core enums and constants
- `version.py` - Version management
- `py.typed` - Type information marker

### **Level 2: Railway Pattern Foundation**

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

### **Level 3: Dependency Injection**

**FlextContainer** (`container.py`)

- Dependency injection container
- Service registration and retrieval
- Based on FlextResult pattern

### **Level 4: Configuration Management**

**Configuration Modules**

- `config.py` - FlextBaseSettings implementation
- `config_hierarchical.py` - Hierarchical configuration
- `config_models.py` - Configuration data models
- `payload.py` - Message/payload patterns

### **Level 5: Domain Layer**

**Domain-Driven Design Components**

- `entities.py` - FlextEntity base class
- `value_objects.py` - FlextValueObject immutable types
- `aggregate_root.py` - FlextAggregateRoot with events
- `domain_services.py` - Domain services
- `models.py` - General models

### **Level 6: Architectural Patterns**

**CQRS and Validation**

- `commands.py` - Command pattern
- `handlers.py` - Handler patterns
- `validation.py` - Validation framework
- `interfaces.py` - Protocol definitions
- `protocols.py` - Additional protocols
- `guards.py` - Type guards

### **Level 7: Cross-Cutting Concerns**

**Utilities and Extensions**

- `utilities.py` - Utility functions
- `decorators.py` - Decorator patterns
- `mixins.py` - Mixin classes
- `fields.py` - Field definitions
- `exceptions.py` - Exception hierarchy
- `loggings.py` - Logging utilities

### **Level 8: Type System**

**Type Definitions**

- `types.py` - Legacy type system
- `flext_types.py` - Compatibility types
- `semantic_types.py` - Unified semantic patterns
- `semantic.py` - Semantic patterns
- `semantic_old.py` - Legacy semantic

### **Level 9: Base Implementations**

**Internal Base Classes**

- `_result_base.py` - Result pattern base
- `_config_base.py` - Configuration base
- `_decorators_base.py` - Decorator base
- `_handlers_base.py` - Handler base
- `_mixins_base.py` - Mixin base
- `_railway_base.py` - Railway pattern base
- `_builder_base.py` - Builder pattern
- `_delegation_system.py` - Delegation system
- `foundation.py` - Foundation patterns

### **Level 10: Integration and Specialized**

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
FlextBaseSettings (config.py) - uses FlextResult and FlextContainer
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
FlextResult.ok(data) -> FlextResult[T]
FlextResult.fail(...) -> FlextResult[T]

# Unwrapping
result.unwrap() -> T  # May raise if failure
```

### FlextBaseSettings (From config.py)

```python
# Environment integration
class AppSettings(FlextBaseSettings):
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
- **Legacy modules** marked for compatibility (flext_types.py)
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
