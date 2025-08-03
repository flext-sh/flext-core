# FLEXT Core

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Version 0.9.0](https://img.shields.io/badge/version-0.9.0-orange.svg)](https://github.com/flext-sh/flext-core)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/flext-sh/flext-core)

**The Architectural Foundation of the FLEXT Data Integration Ecosystem**

FLEXT Core is the foundational Python library that establishes consistent architectural patterns, type-safe error handling, and enterprise-grade design patterns used across all 32 projects in the FLEXT ecosystem. As a pure library with zero CLI dependencies, it serves as the cornerstone that enables the FLEXT platform to deliver reliable, scalable data integration solutions.

## 🎯 Project Mission

**Enable enterprise-grade data integration through consistent architectural patterns**

FLEXT Core exists to solve the fundamental challenge of maintaining architectural consistency across a complex ecosystem of 32 interconnected projects. By providing battle-tested patterns for error handling, dependency injection, domain modeling, and configuration management, it ensures that every component in the FLEXT ecosystem follows the same enterprise-grade standards.

## 🏗️ Architecture Role in FLEXT Ecosystem

### **Foundation Layer**

FLEXT Core sits at the foundation of the entire FLEXT ecosystem, providing essential patterns for:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLEXT ECOSYSTEM (32 Projects)                 │
├─────────────────────────────────────────────────────────────────┤
│ Services (3): FlexCore(Go) | ALGAR | GrupoNos                    │
├─────────────────────────────────────────────────────────────────┤
│ Applications (6): API | Auth | Web | CLI | Quality | Plugin      │
├─────────────────────────────────────────────────────────────────┤
│ Infrastructure (6): Oracle | LDAP | LDIF | WMS | gRPC | Meltano  │
├─────────────────────────────────────────────────────────────────┤
│ Singer Ecosystem (15): 5 Taps | 5 Targets | 4 DBT | 1 Extension │
├─────────────────────────────────────────────────────────────────┤
│ Go Binaries (4): flext | cli | server | demo                     │
├═════════════════════════════════════════════════════════════════┤
│              FLEXT CORE - ARCHITECTURAL FOUNDATION               │
│  FlextResult | FlextContainer | Domain Patterns | Config        │
└─────────────────────────────────────────────────────────────────┘
```

### **Core Responsibilities**

1. **Type-Safe Error Handling**: FlextResult[T] pattern eliminates exceptions across all 32 projects
2. **Dependency Injection**: FlextContainer provides consistent service location
3. **Domain Modeling**: Enterprise DDD patterns (entities, value objects, aggregates)
4. **Configuration Management**: Environment-aware settings with validation
5. **Structured Logging**: Correlation ID support for distributed tracing
6. **CQRS Foundation**: Command/handler patterns for enterprise applications

## 📊 Current Status (v0.9.0 Beta)

### ✅ **Production-Ready Components**

- **FlextResult[T]**: Complete railway-oriented programming implementation
- **FlextContainer**: Enterprise dependency injection with type safety
- **Domain Patterns**: Entities, value objects, and aggregate roots
- **Configuration System**: Pydantic-based settings with environment variables
- **Structured Logging**: Advanced logging with context management
- **Validation Framework**: Comprehensive input validation with guards

### 🚧 **In Development for 1.0.0 (December 2025)**

- **Event Sourcing**: Complete event store with persistence and replay
- **Advanced CQRS**: Query bus, middleware pipeline, handler discovery
- **Plugin Architecture**: Hot-swappable components for ecosystem extensibility
- **Cross-Language Bridge**: Python-Go integration for FlexCore services
- **Enterprise Observability**: Distributed tracing and correlation propagation

## 🚀 Quick Start

### Installation

```bash
pip install flext-core
```

**Requirements**: Python 3.13+ only (no backward compatibility)  
**Dependencies**: Minimal (pydantic, pydantic-settings, structlog)

### Basic Usage

```python
from flext_core import FlextResult, FlextContainer

# Type-safe error handling
def divide(a: int, b: int) -> FlextResult[float]:
    if b == 0:
        return FlextResult.fail("Division by zero")
    return FlextResult.ok(a / b)

# Railway-oriented programming
result = (
    divide(10, 2)
    .map(lambda x: x * 2)
    .flat_map(lambda x: divide(x, 3))
)

if result.is_success:
    print(f"Result: {result.data}")  # Result: 3.33...
else:
    print(f"Error: {result.error}")

# Dependency injection
container = FlextContainer()
result = container.register("calculator", CalculatorService())
assert result.is_success

calculator = container.get("calculator").unwrap()
```

### Domain Modeling

```python
from flext_core import FlextEntity, FlextValueObject, FlextResult

class User(FlextEntity):
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        # Domain events collected (persistence in 1.0.0)
        self.add_domain_event({"type": "UserActivated", "user_id": self.id})
        return FlextResult.ok(None)

class Email(FlextValueObject):
    address: str

    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError("Invalid email format")
```

## 🛠️ Development

### Setup

```bash
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup  # Install dependencies and pre-commit hooks
```

### Quality Gates (Required)

```bash
make validate      # Complete validation pipeline (all must pass)
make check         # Quick lint + type check
make test          # Run tests (95% coverage required)
make format        # Code formatting
make security      # Security scanning
```

### Testing

```bash
make test                    # Full test suite with coverage
make test-unit               # Unit tests only
make test-integration        # Integration tests only
pytest -m "not slow"         # Fast tests only
pytest -m core               # Core framework tests
pytest -m ddd                # Domain-driven design tests
```

## 📈 Impact Metrics

### **Quality Standards**

- **Test Coverage**: 95% minimum (currently 95%+)
- **Type Safety**: Strict MyPy with zero tolerance
- **Security**: Bandit + pip-audit scanning
- **Linting**: Ruff with comprehensive rules
- **Line Length**: 79 characters (strict PEP8)

### **Ecosystem Integration**

- **32 Projects**: Direct dependencies on FLEXT Core patterns
- **Zero Downtime**: Changes require ecosystem-wide validation
- **Consistent API**: FlextResult[T] used in 15,000+ function signatures
- **Enterprise Ready**: Used in production data integration pipelines

## 🗓️ Development Roadmap

### **Current Sprint (August 2025)**

- ✅ Quality gate fixes (formatting, security)
- 🔄 Enhanced test coverage and integration testing
- 🔄 Security audit and vulnerability assessment

### **Core Architecture (Sep-Oct 2025)**

- Event Sourcing with PostgreSQL/SQLite persistence
- Query Bus and CQRS middleware pipeline
- Plugin architecture foundation with registry
- Container performance optimization (10x improvement target)

### **Enterprise Features (November 2025)**

- Python-Go bridge for FlexCore integration
- Distributed tracing with OpenTelemetry
- Advanced observability and metrics collection
- Comprehensive integration test suites

### **Production Readiness (December 2025)**

- Performance benchmarking across ecosystem
- Security hardening and audit
- Complete documentation with real-world examples
- 1.0.0 Release candidate and community validation

## 🌐 Ecosystem Projects Using FLEXT Core

### **Infrastructure Libraries**

- **flext-db-oracle**: Oracle database connectivity with enterprise patterns
- **flext-ldap**: LDAP directory services with domain modeling
- **flext-ldif**: LDIF file processing with validation frameworks
- **flext-oracle-wms**: Warehouse Management System integration
- **flext-grpc**: gRPC communication with type-safe error handling

### **Application Services**

- **flext-api**: REST API services built on FlextResult patterns
- **flext-auth**: Authentication services with domain entities
- **flext-web**: Web interface using core configuration patterns
- **flext-cli**: Command-line tools with consistent error handling

### **Data Integration (Singer Ecosystem)**

- **15 Projects**: All Singer taps, targets, and DBT transformations use FLEXT Core
- **Consistent Patterns**: FlextResult for data pipeline error handling
- **Type Safety**: Domain models with Pydantic integration

### **Core Services**

- **FlexCore (Go)**: Integrates via Python bridge for business logic
- **ALGAR**: Oracle Unified Directory migration using core patterns
- **GrupoNos**: Meltano-native implementation with FLEXT patterns

## 📚 Documentation

- [**Getting Started**](docs/getting-started/) - Installation and quickstart guides
- [**Architecture Guide**](docs/architecture/) - Clean Architecture implementation
- [**API Reference**](docs/api/) - Complete API documentation
- [**Configuration**](docs/configuration/) - Settings and environment management
- [**Examples**](examples/) - 17 comprehensive working examples
- [**Development Guide**](docs/development/) - Contributing and best practices
- [**TODO & Roadmap**](docs/TODO.md) - Current gaps and development timeline

## 🤝 Contributing

FLEXT Core welcomes contributions that enhance the foundational patterns used across the ecosystem:

### **Priority Areas for Contribution**

1. **Event Sourcing Implementation**: Help complete the event store foundation
2. **Performance Optimization**: Container and handler performance improvements
3. **Documentation**: Real-world examples and architectural decision records
4. **Testing**: Integration test coverage and ecosystem validation

### **Contribution Process**

1. Fork repository and create feature branch
2. Follow development standards: `make validate` must pass
3. Maintain 95% test coverage with comprehensive test suites
4. Submit pull request with ecosystem impact assessment

### **Guidelines**

- **Python 3.13 only** - no backward compatibility
- **Type safety first** - comprehensive type hints required
- **Railway-oriented** - use FlextResult for all error handling
- **Ecosystem aware** - consider impact on 32 dependent projects

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/flext-sh/flext-core/discussions)
- **Security**: Report security issues privately to the maintainers

---

**FLEXT Core v0.9.0** - The architectural foundation enabling enterprise-grade data integration across 32 interconnected projects. Beta software with solid foundations and clear roadmap to production readiness in December 2025.

**Mission**: Provide the architectural foundation that enables the FLEXT ecosystem to deliver reliable, scalable, and maintainable data integration solutions for enterprise environments.
