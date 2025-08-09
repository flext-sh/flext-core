# FLEXT Core Documentation

**Foundation library for clean architecture patterns**

Welcome to FLEXT Core documentation. This library provides foundational patterns for type-safe error handling, dependency injection, domain modeling, and configuration management in Python applications.

## 🎯 What is FLEXT Core

FLEXT Core is a **foundation library** that establishes architectural patterns for Python applications, providing:

- **Type-Safe Error Handling**: FlextResult[T] pattern for explicit error management
- **Dependency Injection**: FlextContainer for service management
- **Domain-Driven Design**: Entities, value objects, and aggregates
- **Configuration Management**: Environment-aware settings with validation
- **Structured Logging**: Built-in logging patterns
- **Command Patterns**: Basic CQRS foundation

## 🚀 Quick Start

### Installation

```bash
pip install flext-core
# or
poetry add flext-core
```

### Basic Usage

```python
from flext_core import FlextResult, FlextContainer

# Type-safe error handling
def process_data(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult.fail("Data cannot be empty")
    return FlextResult.ok(data.upper())

# Railway-oriented programming
result = (
    process_data("hello")
    .map(lambda x: f"Processed: {x}")
)

if result.success:
    print(f"Success: {result.data}")
else:
    print(f"Error: {result.error}")
```

### Domain Modeling (models API)

```python
from flext_core.models import FlextEntity
from flext_core import FlextResult

class User(FlextEntity):
    id: str
    name: str
    email: str
    is_active: bool = False

    def validate_business_rules(self) -> FlextResult[None]:
        if "@" not in self.email:
            return FlextResult.fail("Invalid email")
        if not self.name.strip():
            return FlextResult.fail("Name required")
        return FlextResult.ok(None)
```

## 📖 Documentation Sections

### 🚀 **Getting Started**

New to FLEXT Core? Start here for installation and basic usage.

- [**Installation Guide**](getting-started/installation.md) - Setup and dependencies
- [**Quick Start Guide**](getting-started/quickstart.md) - Core patterns and examples

### 🏗️ **Architecture**

Understand the design principles and patterns.

- [**Architecture Overview**](architecture/overview.md) - Clean Architecture patterns
- [**Component Hierarchy**](architecture/component-hierarchy.md) - Module organization

### 📚 **API Reference**

Complete reference for FLEXT Core components.

- [**Core API**](api/core.md) - FlextResult, FlextContainer, FlextSettings
- [**Patterns API**](api/patterns.md) - Commands, Handlers, Validation

### ⚙️ **Configuration**

Environment-aware configuration management.

- [**Configuration Overview**](configuration/overview.md) - Settings patterns
- Secrets Management (planned)

### 🛠️ **Development**

Best practices for development with FLEXT Core.

- [**Best Practices**](development/best-practices.md) - Development guidelines

### 💡 **Examples**

Real-world usage patterns and applications.

- [**Examples Overview**](examples/overview.md) - Working implementations

### 🔧 **Troubleshooting**

Problem resolution and debugging.

- [**Advanced Guide**](troubleshooting/advanced-guide.md) - Common issues

## 🏗️ Core Architecture

FLEXT Core implements Clean Architecture principles:

```
┌─────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                    │
│                Commands, Handlers                       │
├─────────────────────────────────────────────────────────┤
│                     DOMAIN LAYER                        │
│              Entities, Value Objects                    │
├─────────────────────────────────────────────────────────┤
│                INFRASTRUCTURE LAYER                     │
│          Configuration, Logging, Persistence           │
├═════════════════════════════════════════════════════════┤
│                  FOUNDATION LAYER                       │
│         FlextResult | FlextContainer | Types           │
└─────────────────────────────────────────────────────────┘
```

## 📊 Current Status

### ✅ **Stable (v0.9.0)**

- FlextResult[T] for type-safe error handling
- FlextContainer for dependency injection
- FlextSettings for configuration
- FlextEntity for domain modeling
- Basic logging support

### 🚧 **In Development**

- Complete CQRS implementation
- Event sourcing patterns
- Advanced domain patterns
- Plugin architecture

### 📋 **Planned**

- Performance optimizations
- Advanced logging features
- Extended validation patterns

## 🎯 Use Cases

### **Application Development**

Building applications that need:

- Type-safe error handling throughout the codebase
- Dependency injection for service management
- Clean architecture patterns
- Configuration management

### **Library Development**

Creating libraries that require:

- Consistent error handling patterns
- Domain modeling capabilities
- Configuration support
- Testing utilities

### **Enterprise Applications**

Developing enterprise solutions with:

- Railway-oriented programming
- Domain-driven design
- Structured configuration
- Comprehensive logging

## 📈 Quality Standards

- **Python 3.13+** only (modern language features)
- **95% test coverage** minimum requirement
- **MyPy strict mode** with zero type errors
- **PEP8 compliance** with 79-character lines
- **Comprehensive documentation** for all public APIs

## 🤝 Contributing

FLEXT Core welcomes contributions:

### **Development Setup**

```bash
git clone <repository-url>
cd flext-core
make setup
make validate  # Run all quality checks
```

### **Areas for Contribution**

1. **Documentation** - Examples and guides
2. **Testing** - Additional test coverage
3. **Performance** - Optimization opportunities
4. **Features** - New architectural patterns

### **Guidelines**

1. Follow [**Best Practices**](development/best-practices.md)
2. Maintain quality standards (lint, type-check, test coverage)
3. Include comprehensive tests
4. Update documentation

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check this documentation first

## ⚠️ Important Notes

- This documentation reflects the current implementation in `src/flext_core/`
- Examples are aligned to real exports in `src/flext_core/__init__.py`
- Some features are in active development (see status sections)
- Check the source code for the most up-to-date API

---

**FLEXT Core** - Foundation for clean, maintainable Python applications

**Version**: 0.9.0 | **License**: MIT | **Python**: 3.13+
