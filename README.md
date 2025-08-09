# flext-core

**Type**: Foundation Library | **Status**: Development | **Dependencies**: None

Architectural foundation library providing consistent patterns, type-safe error handling, and enterprise design patterns for the FLEXT ecosystem.

> **⚠️ Development Status**: Core patterns stable (FlextResult, FlextContainer), domain patterns active development, 95% test coverage target

## Quick Start

```bash
# Install dependencies
poetry install

# Test basic functionality
python -c "from flext_core import FlextResult; result = FlextResult.ok('test'); print('✅ Working')"

# Development setup
make setup
```

## Current Reality

**What Actually Works:**

- FlextResult[T] pattern for type-safe error handling
- FlextContainer dependency injection system
- Domain entities (FlextEntity, FlextValueObject, FlextAggregateRoot)
- Configuration management with FlextSettings
- Structured logging with correlation IDs

**What Needs Work:**

- Event Sourcing implementation incomplete
- CQRS patterns (Command/Query Bus missing)
- Plugin architecture foundation missing
- Cross-language bridge patterns (Python-Go)

## Architecture Role in FLEXT Ecosystem

### **Foundation Component**

FLEXT Core provides base patterns used by all ecosystem projects:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLEXT ECOSYSTEM (32 Projects)                 │
├─────────────────────────────────────────────────────────────────┤
│ Services: FlexCore(Go) | FLEXT Service(Go/Python) | Clients     │
├─────────────────────────────────────────────────────────────────┤
│ Applications: API | Auth | Web | CLI | Quality | Observability  │
├─────────────────────────────────────────────────────────────────┤
│ Infrastructure: Oracle | LDAP | LDIF | gRPC | Plugin | WMS      │
├─────────────────────────────────────────────────────────────────┤
│ Singer Ecosystem: Taps(5) | Targets(5) | DBT(4) | Extensions(1) │
├═════════════════════════════════════════════════════════════════┤
│ Foundation: [FLEXT-CORE] (Patterns | Types | Domain Base)       │
└─────────────────────────────────────────────────────────────────┘
```

### **Core Responsibilities**

1. **Pattern Foundation**: FlextResult, FlextContainer, domain patterns
2. **Type System**: Base types and interfaces for ecosystem
3. **Domain Modeling**: Clean Architecture and DDD base classes

## Key Features

### **Current Capabilities**

- **FlextResult[T]**: Railway-oriented programming for error handling
- **FlextContainer**: Enterprise dependency injection container
- **Domain Patterns**: FlextEntity, FlextValueObject with business validation
- **Configuration**: Environment-aware settings with Pydantic

### **Pattern Examples**

```python
from flext_core import FlextResult, FlextContainer, FlextEntity

# Type-safe error handling
def process_data(data: str) -> FlextResult[ProcessedData]:
    if not data:
        return FlextResult.fail("Empty data provided")
    return FlextResult.ok(ProcessedData(data))

# Dependency injection
container = FlextContainer()
container.register("service", UserService())
service = container.get("service").unwrap()

# Domain entities with validation
class User(FlextEntity):
    name: str
    email: str

    def validate_domain_rules(self) -> FlextResult[None]:
        if "@" not in self.email:
            return FlextResult.fail("Invalid email")
        return FlextResult.ok(None)
```

## Installation & Usage

### Installation

```bash
# Clone and install
cd /path/to/flext-core
poetry install

# Development setup
make setup
```

### Basic Usage

```python
from flext_core import FlextResult, FlextContainer

# Railway-oriented programming
result = (
    validate_input(data)
    .flat_map(process_data)
    .map(format_output)
)

if result.success:
    print(f"Result: {result.data}")
else:
    print(f"Error: {result.error}")
```

## Development Commands

### Quality Gates (Zero Tolerance)

```bash
# Complete validation pipeline (run before commits)
make validate              # Full validation (lint + type + security + test)
make check                 # Quick lint + type check + test
make test                  # Run all tests (95% coverage requirement)
make lint                  # Code linting
make type-check            # Type checking
make format                # Code formatting
make security              # Security scanning
```

### Testing

```bash
# Test categories
make test-unit             # Unit tests only
make test-integration      # Integration tests only
make coverage-html         # Generate HTML coverage report

# Specific test patterns
pytest -m unit
pytest -m integration
pytest -m "not slow"       # Fast tests for quick feedback
```

## Configuration

### Environment Variables

```bash
# Core library configuration
export FLEXT_LOG_LEVEL="INFO"
export FLEXT_DEBUG="false"

# Development settings
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

## Quality Standards

### **Zero Tolerance Quality Gates**

- **Coverage**: 95% test coverage enforced
- **Type Safety**: Strict MyPy configuration
- **Linting**: Ruff with comprehensive rules
- **Security**: Bandit + pip-audit scanning

## Integration with FLEXT Ecosystem

### **Pattern Usage Across Ecosystem**

```python
# All ecosystem projects use these patterns
from flext_core import FlextResult, FlextContainer, FlextEntity

# Service operations return FlextResult
async def service_operation() -> FlextResult[Data]:
    return FlextResult.ok(processed_data)

# Dependency injection throughout ecosystem
container = get_flext_container()
service = container.get("service_name").unwrap()
```

### **Foundation for All Projects**

- **29 FLEXT Libraries**: Use FlextResult, domain patterns
- **FlexCore (Go)**: Integrates via Python bridge
- **Services**: Built on foundation patterns

## Current Status

**Version**: 0.9.0 (Development)

**Completed**:

- ✅ FlextResult pattern with railway-oriented programming
- ✅ FlextContainer dependency injection system
- ✅ Domain entity patterns (FlextEntity, FlextValueObject)
- ✅ Configuration management with Pydantic

**In Progress**:

- 🔄 Event Sourcing implementation
- 🔄 CQRS patterns (Command/Query Bus)
- 🔄 Plugin architecture foundation

**Planned**:

- 📋 Cross-language bridge patterns (Python-Go)
- 📋 Advanced event-driven patterns
- 📋 Distributed system patterns

## Contributing

### Development Standards

- **Pure Library**: No CLI dependencies, foundation patterns only
- **Type Safety**: All code must pass MyPy strict mode
- **Testing**: Maintain 95% coverage
- **Ecosystem Impact**: Changes affect 32 dependent projects

### Development Workflow

```bash
# Setup and validate
make setup
make validate
make test
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Links

- **[CLAUDE.md](CLAUDE.md)**: Development guidance
- **[Documentation](docs/)**: Complete documentation

---

_Foundation library for the FLEXT ecosystem - Enterprise data integration platform_
