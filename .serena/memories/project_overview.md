# FLEXT-CORE Project Overview

## Purpose

FLEXT-CORE is the **foundation library** for the entire FLEXT ecosystem, serving 32+ dependent projects. It provides:

- Railway Pattern Foundation (FlextResult[T])
- Dependency Injection (FlextContainer)
- Domain Models (FlextModels - Entity/Value/AggregateRoot)
- Service Architecture (FlextDomainService)
- Complete type safety for ecosystem

## Current Status

- **Version**: 0.9.9 RC (Production/Stable)
- **Test Coverage**: 79% (targeting 85%+)
- **Quality**: Zero MyPy/PyRight errors in src/
- **Dependencies**: 32+ projects depend on this foundation

## Tech Stack

- **Python**: 3.13+ (latest features)
- **Pydantic**: 2.11.7+ (modern data validation)
- **StructLog**: 25.4.0+ (structured logging)
- **Poetry**: Dependency management
- **Architecture**: Clean Architecture + DDD + SOLID

## Critical Requirements

- ZERO tolerance for API breaking changes
- Zero errors in MyPy strict mode, PyRight, and Ruff for src/
- Single class per module pattern (unified classes with nested helpers)
- No wrappers, no fallbacks, no legacy access
- Use flext-core extensively, avoid code duplication
- Professional documentation for all public APIs
