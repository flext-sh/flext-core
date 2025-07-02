# FLX-Core Component Hierarchy Guide

**Document Type**: Component Dependency Analysis
**Project**: FLX-Core Framework
**Created**: 2025-06-29
**Purpose**: Understand which components are most core/internal vs external

---

## 🎯 Hierarchy Overview

This document answers the question: **"Which components are most core/internal to the FLX-Core project?"**

The answer is clear: **`domain/pydantic_base.py`** is the absolute core, with all other components building upon it in layers.

## 📊 Dependency Pyramid (Most Core → Most External)

```
                            🏆 ABSOLUTE CORE
                        domain/pydantic_base.py
                      Foundation for ALL components
                    (DomainBaseModel, ServiceResult, etc.)

                           ↑ EVERYTHING DEPENDS ON ↑

                            🎯 TYPE FOUNDATION
                    domain/base.py + advanced_types.py
                      Fundamental types and Python 3.13
                        (DomainId, UserId, PipelineName)

                           ↑ DOMAIN MODELS DEPEND ON ↑

                            🏗️ BUSINESS MODELS
                   domain/entities.py + value_objects.py
                    Core business concepts and aggregates
                      (Pipeline, PipelineExecution, Plugin)

                           ↑ BOUNDARIES DEPEND ON ↑

                          🔌 ARCHITECTURAL BOUNDARIES
                               domain/ports.py
                        Clean architecture interfaces
                     (Primary ports, Secondary ports, Protocols)

                           ↑ EVENTS SYSTEM DEPENDS ON ↑

                            ⚡ EVENT FOUNDATION
                             events/event_bus.py
                        Event-driven architecture support
                        (EventBusProtocol, Lato integration)

                           ↑ CONFIG DEPENDS ON ↑

                            ⚙️ CONFIGURATION LAYER
                           config/domain_config.py
                        Centralized configuration management
                        (Environment config, Business constants)

                           ↑ APPLICATION DEPENDS ON ↑

                            🎯 USE CASE LAYER
                              application/*
                        Command handlers and domain services
                        (Commands, Handlers, Application services)

                           ↑ INFRASTRUCTURE DEPENDS ON ↑

                           🗃️ EXTERNAL ADAPTERS
                            infrastructure/*
                        Database, external APIs, file system
                        (Repositories, ORM models, Unit of work)
```

## 🔍 Detailed Component Analysis

### 🏆 Level 1: Absolute Core - `pydantic_base.py`

**Why this is most core:**

- Every other component inherits from classes defined here
- Removing this breaks the entire system
- Zero dependencies on other FLX components
- Provides fundamental abstractions for everything else

**Key Components:**

```python
DomainBaseModel     # Foundation Pydantic model
DomainValueObject   # Immutable objects (frozen=True)
DomainEntity        # Identity-based objects
DomainAggregateRoot # Event sourcing aggregates
DomainEvent         # Immutable domain events
ServiceResult[T]    # Result type for operations
```

**Dependencies:** Only external libraries (Pydantic, Python stdlib)

### 🎯 Level 2: Type Foundation

**Components:** `base.py` + `advanced_types.py`

**Why this level:**

- Builds directly on pydantic_base.py
- Defines fundamental types used throughout
- Business-agnostic type definitions
- Required by all domain models

**Key Types:**

```python
DomainId           # Base identifier type
UserId, TenantId   # Business-specific IDs
PipelineName       # Domain value types
ServiceResult      # Re-exported from base
```

**Dependencies:** pydantic_base.py

### 🏗️ Level 3: Business Models

**Components:** `entities.py` + `value_objects.py`

**Why this level:**

- Implements actual business concepts
- Uses foundation types and base classes
- Defines core aggregates and value objects
- Business logic starts here

**Key Models:**

```python
Pipeline           # Core aggregate root
PipelineExecution  # Execution tracking
Plugin             # Plugin management
ExecutionStatus    # Status enumeration
Duration           # Time value object
```

**Dependencies:** pydantic_base.py, base.py, advanced_types.py

### 🔌 Level 4: Architectural Boundaries

**Component:** `ports.py`

**Why this level:**

- Defines interfaces for Clean Architecture
- Uses domain models but doesn't implement business logic
- Separates domain from application/infrastructure
- Critical for dependency inversion

**Key Interfaces:**

```python
PipelineManagementPort  # Primary port
PluginManagementPort    # Primary port
PipelineRepository      # Secondary port
EventBusPort           # Event integration
```

**Dependencies:** All domain models and types

### ⚡ Level 5: Event Foundation

**Component:** `events/event_bus.py`

**Why this level:**

- Implements event-driven architecture
- Provides infrastructure for domain events
- Uses domain models and ports
- Enables reactive programming

**Key Components:**

```python
EventBusProtocol   # Event bus interface
Lato integration   # DI container events
Domain event routing # Event distribution
```

**Dependencies:** Domain models, ports, external libraries (Lato)

### ⚙️ Level 6: Configuration

**Component:** `config/domain_config.py`

**Why this level:**

- Configures behavior of all other components
- Environment-aware settings
- Business rule configuration
- System-wide constants

**Key Features:**

```python
Environment config     # Development/production
Business constants     # Domain parameters
Type-safe validation   # Configuration validation
```

**Dependencies:** All lower levels for type definitions

### 🎯 Level 7: Application Layer

**Components:** `application/*`

**Why this level:**

- Orchestrates domain objects for use cases
- Implements business workflows
- Uses all domain components
- Bridges to external world

**Key Components:**

```python
Command handlers       # Business operations
Domain services        # Complex logic coordination
Application services   # Use case orchestration
CQRS implementation   # Command/Query separation
```

**Dependencies:** All domain layer, events, configuration

### 🗃️ Level 8: Infrastructure (Most External)

**Components:** `infrastructure/*`

**Why this is most external:**

- Adapts to external systems (databases, APIs, etc.)
- Implements interfaces defined by domain
- Can be replaced without affecting business logic
- Highest level of dependencies

**Key Components:**

```python
Repository implementations  # Database integration
SQLAlchemy models          # ORM mapping
Unit of work pattern       # Transaction management
Session management         # Database sessions
```

**Dependencies:** Everything - implements all domain interfaces

## 🧭 Navigation Guide

### Finding the Core for Different Purposes

**If you need to understand:**

- **Basic concepts** → Start with `pydantic_base.py`
- **Business logic** → Look at `entities.py` and `value_objects.py`
- **Interfaces** → Check `ports.py`
- **Configuration** → Examine `domain_config.py`
- **Data persistence** → Study `infrastructure/persistence/`

### Modification Impact Analysis

**Changing `pydantic_base.py`:**

- 🚨 **CRITICAL IMPACT** - Affects everything
- Requires testing entire system
- Likely breaking changes across all modules

**Changing `entities.py`:**

- 🔶 **HIGH IMPACT** - Affects business logic
- Requires testing all use cases
- May affect API contracts

**Changing `infrastructure/*`:**

- 🟢 **LOW IMPACT** - Isolated to adapters
- Can be changed without affecting business logic
- Mainly affects external integrations

## 📋 Practical Guidelines

### When Adding New Features

1. **Start from the core** - Do you need new base types?
2. **Define domain models** - What business concepts are involved?
3. **Add interfaces** - What ports do you need?
4. **Implement use cases** - How do components work together?
5. **Add infrastructure** - How do you connect to external systems?

### When Debugging Issues

1. **Check foundation** - Are base classes working correctly?
2. **Verify domain logic** - Are business rules implemented correctly?
3. **Examine interfaces** - Are contracts being fulfilled?
4. **Test infrastructure** - Are external connections working?

### When Refactoring

1. **Core changes** - Plan carefully, high impact
2. **Domain changes** - Focus on business logic clarity
3. **Infrastructure changes** - Usually safe, isolated impact

## 🎯 Summary

The FLX-Core architecture follows a clear hierarchical structure where **`domain/pydantic_base.py`** forms the absolute foundation. Each layer builds upon the previous ones, creating a stable and maintainable architecture where changes to external layers don't affect the core business logic.

This hierarchy ensures:

- **Stability**: Core business logic is protected from external changes
- **Testability**: Each layer can be tested independently
- **Maintainability**: Clear dependencies make refactoring safer
- **Extensibility**: New features can be added without affecting existing core
