# FLEXT Core New Semantic Architecture Plan

## Objective

Reorganize flext-core to be a pure foundation library with clear semantic organization, removing concrete implementations and providing only abstract patterns and base classes.

## New Semantic Structure

```
src/flext_core/
├── foundation/              # 🏗️ ABSOLUTE FOUNDATION
│   ├── __init__.py         # Core exports only
│   ├── abstractions.py     # Pure abstractions & interfaces
│   ├── patterns.py         # Base architectural patterns
│   ├── primitives.py       # Primitive types & value objects
│   └── protocols.py        # Protocol definitions
│
├── domain/                 # 🎯 DOMAIN LAYER (Pure business logic)
│   ├── __init__.py
│   ├── entities/           # Entity base classes
│   │   ├── __init__.py
│   │   ├── base.py         # Base entity patterns
│   │   └── aggregates.py   # Aggregate root patterns
│   ├── values/             # Value objects
│   │   ├── __init__.py
│   │   ├── base.py         # Value object base
│   │   └── common.py       # Common value objects (ID types, etc)
│   ├── events/             # Domain events
│   │   ├── __init__.py
│   │   ├── base.py         # Event base classes
│   │   └── bus.py          # Event bus abstractions
│   ├── services/           # Domain services (abstractions only)
│   │   ├── __init__.py
│   │   └── base.py         # Domain service base patterns
│   └── specifications/     # Specification pattern
│       ├── __init__.py
│       └── base.py         # Specification base classes
│
├── application/            # 🎯 APPLICATION LAYER (Use cases)
│   ├── __init__.py
│   ├── commands/           # Command patterns
│   │   ├── __init__.py
│   │   ├── base.py         # Command base classes
│   │   └── handlers.py     # Command handler patterns
│   ├── queries/            # Query patterns  
│   │   ├── __init__.py
│   │   ├── base.py         # Query base classes
│   │   └── handlers.py     # Query handler patterns
│   ├── services/           # Application services
│   │   ├── __init__.py
│   │   └── base.py         # Application service patterns
│   └── workflows/          # Workflow patterns
│       ├── __init__.py
│       └── base.py         # Workflow orchestration
│
├── infrastructure/         # 🗃️ INFRASTRUCTURE LAYER (Abstractions only)
│   ├── __init__.py
│   ├── repositories/       # Repository patterns
│   │   ├── __init__.py
│   │   ├── base.py         # Repository interfaces
│   │   └── memory.py       # In-memory implementation (for testing)
│   ├── messaging/          # Messaging abstractions
│   │   ├── __init__.py
│   │   ├── base.py         # Message broker interfaces
│   │   └── patterns.py     # Messaging patterns
│   ├── persistence/        # Persistence abstractions
│   │   ├── __init__.py
│   │   ├── base.py         # Persistence interfaces
│   │   └── transactions.py # Transaction patterns
│   ├── serialization/      # Serialization abstractions
│   │   ├── __init__.py
│   │   ├── base.py         # Serializer interfaces
│   │   └── json.py         # JSON serialization patterns
│   └── external/           # External service interfaces
│       ├── __init__.py
│       ├── http.py         # HTTP client abstractions
│       └── protocols.py    # External protocol interfaces
│
├── configuration/          # ⚙️ CONFIGURATION (Abstract config patterns)
│   ├── __init__.py
│   ├── base.py             # Base configuration classes
│   ├── validation.py       # Configuration validation patterns
│   ├── secrets.py          # Secret management abstractions
│   └── profiles.py         # Configuration profile patterns
│
├── integration/            # 🔌 INTEGRATION (Abstract integration patterns)
│   ├── __init__.py
│   ├── adapters/           # Adapter pattern implementations
│   │   ├── __init__.py
│   │   └── base.py         # Adapter base classes
│   ├── protocols/          # Protocol adapters (abstractions)
│   │   ├── __init__.py
│   │   ├── rest.py         # REST protocol abstractions
│   │   ├── grpc.py         # gRPC protocol abstractions
│   │   └── messaging.py    # Messaging protocol abstractions
│   └── translation/        # Data translation patterns
│       ├── __init__.py
│       └── base.py         # Translation pattern base
│
├── observability/          # 📊 OBSERVABILITY (Abstract monitoring patterns)
│   ├── __init__.py
│   ├── logging/            # Logging abstractions
│   │   ├── __init__.py
│   │   ├── base.py         # Logger interfaces
│   │   └── structured.py   # Structured logging patterns
│   ├── metrics/            # Metrics abstractions
│   │   ├── __init__.py
│   │   ├── base.py         # Metrics interfaces
│   │   └── collectors.py   # Metric collection patterns
│   ├── tracing/            # Tracing abstractions
│   │   ├── __init__.py
│   │   └── base.py         # Tracing interfaces
│   └── health/             # Health check patterns
│       ├── __init__.py
│       └── base.py         # Health check abstractions
│
└── security/               # 🔒 SECURITY (Abstract security patterns)
    ├── __init__.py
    ├── authentication/     # Authentication abstractions
    │   ├── __init__.py
    │   ├── base.py         # Auth interfaces
    │   └── tokens.py       # Token handling patterns
    ├── authorization/      # Authorization abstractions
    │   ├── __init__.py
    │   ├── base.py         # Authorization interfaces
    │   └── policies.py     # Policy pattern implementations
    ├── cryptography/       # Crypto abstractions
    │   ├── __init__.py
    │   ├── base.py         # Crypto interfaces
    │   └── hashing.py      # Hashing pattern abstractions
    └── validation/         # Security validation patterns
        ├── __init__.py
        └── base.py         # Validation pattern base
```

## Components to Deprecate (Move to Legacy)

### Current paths → New paths mapping

1. **Configuration Adapters** (Too specific for core):
   - `config/adapters/cli.py` → Move to `flext-cli`
   - `config/adapters/django.py` → Move to `flext-web`
   - `config/adapters/singer.py` → Move to `flext-meltano`
   - `config/oracle.py` → Move to `flext-db-oracle`
   - `config/oracle_oic.py` → Move to `flext-oracle-oic-ext`

2. **Specific Utilities** (Too concrete for core):
   - `utils/ldif_writer.py` → Move to `flext-ldif`
   - `utils/config_generator.py` → Abstract pattern in `configuration/`

3. **Mixed Domain Models** (Split abstract from concrete):
   - `domain/pipeline.py` → Keep abstract pipeline concepts, move Singer specifics
   - `application/pipeline.py` → Keep application patterns, move implementations

## Migration Strategy

### Phase 1: Create New Structure

1. Create new semantic package structure
2. Implement abstract base classes in new locations
3. Add deprecation warnings to old imports

### Phase 2: Compatibility Layer

1. Keep old imports working via forwarding
2. Add warnings about deprecated paths
3. Provide clear migration guidance

### Phase 3: Documentation & Testing

1. Update all documentation with new semantic paths
2. Ensure 100% test coverage for new structure
3. Quality gates pass with zero violations

## Benefits of New Structure

### Semantic Clarity

- **foundation/**: Core building blocks (protocols, patterns, primitives)
- **domain/**: Pure business logic (entities, values, events, services)
- **application/**: Use cases and workflows (commands, queries, services)
- **infrastructure/**: External concerns (repositories, messaging, persistence)
- **configuration/**: Settings and validation patterns
- **integration/**: Adapter and protocol patterns
- **observability/**: Monitoring and logging abstractions
- **security/**: Authentication, authorization, and crypto patterns

### Quick Navigation

- Looking for base classes? → `foundation/`
- Need domain patterns? → `domain/`
- Want application patterns? → `application/`
- Infrastructure abstractions? → `infrastructure/`
- Configuration patterns? → `configuration/`
- Integration patterns? → `integration/`
- Observability setup? → `observability/`
- Security patterns? → `security/`

### Clear Separation of Concerns

- Pure abstractions in `foundation/`
- Business logic in `domain/`
- Use cases in `application/`
- External concerns in `infrastructure/`
- No concrete implementations (Oracle, LDAP, etc.)
- Framework adapters moved to appropriate modules

## Implementation Principles

1. **SOLID Compliance**: Each module has single responsibility
2. **Clean Architecture**: Strict dependency inversion
3. **DRY**: No code duplication across new structure
4. **KISS**: Simple, clear interfaces
5. **Type Safety**: 100% typed with modern Python 3.13
6. **Zero Tolerance**: No quality violations tolerated
