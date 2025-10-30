# FLEXT-Core Documentation Index

**Status**: ✅ Complete and Current  
**Last Updated**: 2025-01-21  
**Version**: v0.9.9 Release Candidate

---

## 🎯 Quick Navigation

### For First-Time Users

1. **[Getting Started Guide](./guides/getting-started.md)** - Installation and basic usage
2. **[Architecture Overview](./architecture/overview.md)** - System design and layer hierarchy
3. **[Quick Examples](../examples/)** - Working code samples (00-15)

### For Developers

- **[Railway-Oriented Programming](./guides/railway-oriented-programming.md)** - Error handling patterns
- **[Dependency Injection Advanced](./guides/dependency-injection-advanced.md)** - Service management
- **[Domain-Driven Design](./guides/domain-driven-design.md)** - Entity and value object patterns
- **[Best Practices & Anti-Patterns](./guides/anti-patterns-best-practices.md)** - Common mistakes and solutions

### For API Reference

- **[Foundation Layers (0, 0.5, 1)](./api-reference/foundation.md)** - Core types and utilities
- **[Domain Layer (2)](./api-reference/domain.md)** - Models, services, and domain patterns
- **[Application Layer (3)](./api-reference/application.md)** - Handlers, bus, and dispatchers
- **[Infrastructure Layer (4)](./api-reference/infrastructure.md)** - Config, logging, context

### For Standards & Development

- **[Development Standards](./standards/development.md)** - Code quality, testing, and quality gates
- **[Contributing Guide](./development/contributing.md)** - How to contribute to FLEXT-Core

### For Modernization

- **[Pydantic v2 Modernization Plan](./pydantic-v2-modernization/README.md)** - 9-part plan (9 parts + 8 appendices)
  - Status: Planning Complete, Execution Pending
  - Timeline: 3 weeks for ecosystem migration
  - Includes audit script, automation tools, and migration checklists

---

## 📚 Full Documentation Structure

```
docs/
├── INDEX.md                              (this file)
│
├── QUICK_START.md                       (getting started essentials)
│
├── guides/                              (learning guides)
│   ├── getting-started.md              (installation, basic usage)
│   ├── railway-oriented-programming.md  (FlextResult patterns)
│   ├── dependency-injection-advanced.md (FlextContainer patterns)
│   ├── domain-driven-design.md          (DDD with FlextModels)
│   ├── anti-patterns-best-practices.md  (lessons learned)
│   └── pydantic-v2-patterns.md         (v2 best practices)
│
├── api-reference/                       (API documentation by layer)
│   ├── foundation.md                   (Layer 0, 0.5, 1)
│   ├── domain.md                       (Layer 2)
│   ├── application.md                  (Layer 3)
│   └── infrastructure.md               (Layer 4)
│
├── architecture/                        (system design)
│   ├── overview.md                     (layer hierarchy)
│   └── INTEGRATION_PATTERNS.md         (patterns for ecosystem)
│
├── development/                         (contributing)
│   └── contributing.md                 (development guidelines)
│
├── standards/                           (project standards)
│   └── development.md                  (code quality standards)
│
└── pydantic-v2-modernization/          (modernization plan - 21 files)
    ├── README.md                       (plan overview)
    ├── 01-executive-summary.md         (current state analysis)
    ├── 02-immediate-fixes.md           (critical fixes)
    ├── 03-best-practices.md            (v2 patterns)
    ├── 04-test-fixes.md                (test migration)
    ├── 05-workspace-audit.md           (ecosystem audit)
    ├── 06-quality-gates.md             (automation setup)
    ├── 07-documentation.md             (team enablement)
    ├── 08-execution-timeline.md        (3-week roadmap)
    ├── 09-metrics-risks.md             (success criteria)
    ├── APPENDIX_A_API_REFERENCE.md     (Pydantic v2 API)
    ├── APPENDIX_B_MIGRATION_CHECKLIST.md
    ├── APPENDIX_C_COMMON_ERRORS.md
    ├── APPENDIX_D_GLOSSARY.md
    ├── APPENDIX_E_CODE_EXAMPLES.md
    ├── APPENDIX_F_FAQ.md
    ├── APPENDIX_G_TOOLS_SCRIPTS.md
    ├── APPENDIX_H_REFERENCES.md
    ├── audit_pydantic_v2.py            (automation script)
    └── (+ 3 supporting files)
```

---

## 🏗️ Architecture Layers

### Layer 0: Pure Constants (Zero Dependencies)

- **Module**: `FlextConstants` - 50+ error codes, validation patterns, configuration defaults
- **Module**: `FlextTypes` - Type system with 50+ TypeVars, protocols, type aliases
- **Module**: `FlextProtocols` - Runtime-checkable interfaces
- **Docs**: [Foundation API Reference](./api-reference/foundation.md)

### Layer 0.5: Runtime Bridge (External Libraries)

- **Module**: `FlextRuntime` - Type guards, serialization, logging utilities
- **Provides**: Structured logging, JSON serialization, email/URL validation
- **Docs**: [Foundation API Reference](./api-reference/foundation.md)

### Layer 1: Foundation (Core Patterns)

- **Module**: `FlextResult[T]` - Railway pattern for error handling
- **Module**: `FlextContainer` - Dependency injection singleton
- **Module**: `FlextExceptions` - Exception hierarchy with error codes
- **Docs**: [Foundation API Reference](./api-reference/foundation.md), [Railway Patterns](./guides/railway-oriented-programming.md)

### Layer 2: Domain (Business Logic)

- **Modules**: `FlextModels`, `FlextService`, `FlextMixins`, `FlextUtilities`
- **Patterns**: DDD entities, value objects, domain services
- **Docs**: [Domain API Reference](./api-reference/domain.md), [DDD Guide](./guides/domain-driven-design.md)

### Layer 3: Application (Use Cases)

- **Modules**: `FlextHandlers`, `FlextBus`, `FlextDispatcher`, `FlextRegistry`, `FlextProcessors`
- **Patterns**: CQRS handlers, event bus, message processing
- **Docs**: [Application API Reference](./api-reference/application.md)

### Layer 4: Infrastructure (External Resources)

- **Modules**: `FlextConfig`, `FlextLogger`, `FlextContext`, `FlextDecorators`
- **Patterns**: Configuration management, structured logging, context tracking
- **Docs**: [Infrastructure API Reference](./api-reference/infrastructure.md)

---

## 🔄 Modernization Initiatives

### Pydantic v2 Modernization (📋 Planned)

**Status**: Plan complete, awaiting execution  
**Timeline**: 3 weeks (foundation first, then ecosystem)  
**Impact**: 33 FLEXT projects, improved performance, reduced code duplication

**Key Deliverables**:

- 9-part comprehensive modernization plan
- 8 appendices with API reference, examples, troubleshooting
- Automated audit script and migration tools
- Per-project migration checklist
- Performance benchmarks (50-70% JSON improvement target)

**Where to Start**: [Pydantic v2 Modernization README](./pydantic-v2-modernization/README.md)

---

## 📖 Learning Path

### Beginner Path (4-6 hours)

1. [Getting Started](./guides/getting-started.md)
2. [Railway-Oriented Programming](./guides/railway-oriented-programming.md)
3. [Foundation API Reference](./api-reference/foundation.md)
4. Examples 01-03: Basic patterns

### Intermediate Path (8-12 hours)

1. [Dependency Injection Advanced](./guides/dependency-injection-advanced.md)
2. [Domain-Driven Design](./guides/domain-driven-design.md)
3. [Domain API Reference](./api-reference/domain.md)
4. [Application API Reference](./api-reference/application.md)
5. Examples 04-08: Intermediate patterns

### Advanced Path (12-16 hours)

1. [Best Practices & Anti-Patterns](./guides/anti-patterns-best-practices.md)
2. [Infrastructure API Reference](./api-reference/infrastructure.md)
3. [Architecture Overview](./architecture/overview.md)
4. Examples 09-15: Advanced patterns
5. [Integration Patterns](./architecture/INTEGRATION_PATTERNS.md)

### Contributing Path (4-6 hours)

1. [Development Standards](./standards/development.md)
2. [Contributing Guide](./development/contributing.md)
3. [Anti-Patterns Guide](./guides/anti-patterns-best-practices.md)

---

## 🔍 Cross-References

### By Feature

- **Error Handling**: [Railway Patterns](./guides/railway-oriented-programming.md) → [Foundation API](./api-reference/foundation.md)
- **Dependency Injection**: [DI Guide](./guides/dependency-injection-advanced.md) → [Domain API](./api-reference/domain.md)
- **Data Models**: [DDD Guide](./guides/domain-driven-design.md) → [Domain API](./api-reference/domain.md)
- **Configuration**: [Getting Started](./guides/getting-started.md) → [Infrastructure API](./api-reference/infrastructure.md)
- **Testing**: [Development Standards](./standards/development.md) → [Contributing](./development/contributing.md)

### By Use Case

- **Building a Service**: Getting Started → DI Guide → DDD Guide → Examples 02, 04-07
- **Creating a Handler**: Getting Started → Application API → Examples 07, 14
- **Configuring App**: Getting Started → Infrastructure API → Examples 04
- **Contributing Code**: Development Standards → Contributing Guide → Anti-Patterns

---

## ✅ Status Legend

- **✅ Implemented** (v0.9.9) - Available for use in production
- **🔄 In Progress** - Currently being developed
- **📋 Planned** (v1.0.0) - Scheduled for future release

---

## 📞 Quick Links

- **GitHub**: [FLEXT-Core Repository](https://github.com/flext-sh/flext-core)
- **PyPI**: [FLEXT-Core Package](https://pypi.org/project/flext-core/)
- **Examples**: [Complete Working Examples](../examples/)
- **Tests**: [Test Suite](../tests/)
- **Main README**: [Project Overview](../README.md)
- **Project Standards**: [CLAUDE.md](../CLAUDE.md)

---

**Last Updated**: 2025-01-21  
**FLEXT-Core Version**: v0.9.9 Release Candidate  
**Target**: 1.0.0 Release (October 2025)
