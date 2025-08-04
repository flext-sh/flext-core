# FLEXT Core Documentation

**The Architectural Foundation of Enterprise Data Integration**

Welcome to the comprehensive documentation for FLEXT Core, the foundational Python library that powers the entire FLEXT data integration ecosystem. This documentation provides everything you need to understand, implement, and contribute to the architectural patterns used across all 32 projects in the FLEXT platform.

## 🎯 What is FLEXT Core

FLEXT Core is a **foundational library** that establishes consistent architectural patterns for enterprise-grade data integration. It serves as the cornerstone that enables 32 interconnected projects to work together seamlessly, providing:

- **Type-Safe Error Handling**: FlextResult[T] pattern used across all ecosystem projects
- **Enterprise Dependency Injection**: FlextContainer for consistent service location
- **Domain-Driven Design**: Rich domain entities, value objects, and aggregates
- **Configuration Management**: Environment-aware settings with validation
- **Structured Logging**: Correlation ID support for distributed tracing
- **CQRS Foundation**: Command/handler patterns for enterprise applications

## 🚀 Quick Start

### For New Users

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
    .flat_map(lambda x: save_to_database(x))
)

if result.success:
    print(f"Success: {result.data}")
else:
    print(f"Error: {result.error}")
```

### For Ecosystem Developers

```python
from flext_core import FlextEntity, FlextValueObject

class User(FlextEntity):
    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("Already activated")
        self.is_active = True
        self.add_domain_event({"type": "UserActivated"})
        return FlextResult.ok(None)
```

## 📖 Documentation Sections

### 🚀 **Getting Started**

New to FLEXT Core? Start here for installation and basic usage.

- [**Installation Guide**](getting-started/installation.md) - Environment setup and dependencies
- [**Quick Start Guide**](getting-started/quickstart.md) - Essential patterns and examples

### 🏗️ **Architecture**

Understand the foundational patterns and design principles.

- [**Architecture Overview**](architecture/overview.md) - Clean Architecture and DDD
- [**Component Hierarchy**](architecture/component-hierarchy.md) - Architectural layers

### 📚 **API Reference**

Complete reference for all FLEXT Core components.

- [**Core API**](api/core.md) - FlextResult, FlextContainer, Settings
- [**Patterns API**](api/patterns.md) - Commands, Handlers, Domain Models

### ⚙️ **Configuration**

Environment-aware configuration and secret management.

- [**Configuration Overview**](configuration/overview.md) - Type-safe settings
- [**Secrets Management**](configuration/secrets.md) - Secure data handling

### 🛠️ **Development**

Best practices for enterprise development with FLEXT Core.

- [**Best Practices**](development/best-practices.md) - Development guidelines
- [**TODO & Roadmap**](TODO.md) - Current priorities and timeline

### 💡 **Examples**

Real-world usage patterns and complete applications.

- [**Examples Overview**](examples/overview.md) - Working implementations

### 🔧 **Troubleshooting**

Problem resolution and debugging guidance.

- [**Advanced Guide**](troubleshooting/advanced-guide.md) - Common issues

## 🏗️ Ecosystem Architecture

FLEXT Core serves as the foundation for the complete FLEXT ecosystem:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLEXT ECOSYSTEM (33 Projects)                 │
├─────────────────────────────────────────────────────────────────┤
│ 🎯 Services (3): FlexCore(Go) | client-a | client-b                │
├─────────────────────────────────────────────────────────────────┤
│ 📱 Applications (6): API | Auth | Web | CLI | Quality | Plugin  │
├─────────────────────────────────────────────────────────────────┤
│ 🔧 Infrastructure (6): Oracle | LDAP | LDIF | WMS | gRPC | Melt. │
├─────────────────────────────────────────────────────────────────┤
│ 🔄 Singer Ecosystem (15): 5 Taps | 5 Targets | 4 DBT | 1 Ext.  │
├─────────────────────────────────────────────────────────────────┤
│ ⚡ Go Binaries (4): flext | cli | server | demo                 │
├═════════════════════════════════════════════════════════════════┤
│              FLEXT CORE - ARCHITECTURAL FOUNDATION               │
│         FlextResult | FlextContainer | Domain Patterns          │
└─────────────────────────────────────────────────────────────────┘
```

### **Impact Across Ecosystem**

- **33 Projects** depend on FLEXT Core patterns
- **15,000+ Function Signatures** use FlextResult[T]
- **Zero Downtime** requirement for changes
- **Enterprise Production** deployments

## 📊 Current Status

### ✅ **Production Ready (v0.9.0)**

- Type-safe error handling with FlextResult[T]
- Enterprise dependency injection with FlextContainer
- Domain modeling patterns (entities, value objects, aggregates)
- Configuration management with Pydantic validation
- Structured logging with correlation ID support

### 🚧 **In Development for 1.0.0 (December 2025)**

- Complete Event Sourcing with persistence
- Advanced CQRS with Query Bus and middleware
- Plugin architecture for ecosystem extensibility
- Python-Go bridge for FlexCore integration
- Distributed tracing and enterprise observability

## 🎯 Use Cases

### **For Library Developers**

Building infrastructure libraries (Oracle, LDAP, gRPC) that need:

- Consistent error handling across all operations
- Type-safe dependency injection
- Enterprise configuration patterns
- Domain modeling for business logic

### **For Service Developers**

Creating services (FlexCore, client-a, client-b) that require:

- Clean Architecture implementation
- CQRS patterns for scalability
- Event-driven communication
- Cross-language integration

### **For Application Developers**

Building applications (API, Web, CLI) that need:

- Railway-oriented programming
- Configuration management
- Structured logging
- Enterprise development patterns

### **For Data Integration Teams**

Developing Singer taps, targets, and DBT transformations with:

- Consistent error handling in data pipelines
- Type-safe configuration management
- Domain modeling for data entities
- Observability and monitoring patterns

## 🤝 Contributing

FLEXT Core welcomes contributions that enhance the foundational patterns:

### **High-Impact Areas**

1. **Event Sourcing Implementation** - Complete the event store foundation
2. **Performance Optimization** - Improve container and handler performance
3. **Documentation** - Real-world examples and architectural guides
4. **Testing** - Integration coverage and ecosystem validation

### **Getting Started**

1. Read [**Best Practices**](development/best-practices.md) for guidelines
2. Check [**TODO & Roadmap**](TODO.md) for current priorities
3. Review [**Architecture Overview**](architecture/overview.md) for context
4. Submit pull requests with ecosystem impact assessment

## 📞 Support

- **Documentation Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Community Discussion**: [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
- **Security Reports**: Contact maintainers privately
- **Enterprise Support**: Available for production deployments

---

**Mission**: Provide the architectural foundation that enables the FLEXT ecosystem to deliver reliable, scalable, and maintainable data integration solutions for enterprise environments.

**Version**: 0.9.0 Beta | **Target 1.0.0**: December 2025 | **Ecosystem**: 33 Projects
