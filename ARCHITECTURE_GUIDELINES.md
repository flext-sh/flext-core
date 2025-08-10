# FLEXT-CORE ARCHITECTURE GUIDELINES

**Versão**: 2.0.0  
**Status**: ATIVO  
**Data**: 2025-08-09  

## 📋 VISÃO GERAL

Este documento define os guidelines arquiteturais obrigatórios para o flext-core, a biblioteca foundation do ecosistema FLEXT que serve como base para 32+ projetos.

## 🏗️ PRINCÍPIOS ARQUITETURAIS FUNDAMENTAIS

### 1. CLEAN ARCHITECTURE + DDD + CQRS

```
┌─────────────────────────────────────┐
│         PRESENTATION LAYER          │
├─────────────────────────────────────┤
│         APPLICATION LAYER           │
│    ┌─────────────┬─────────────┐   │
│    │  COMMANDS   │   QUERIES   │   │
│    └─────────────┴─────────────┘   │
├─────────────────────────────────────┤
│           DOMAIN LAYER              │
│  ┌─────────┬──────────┬──────────┐ │
│  │ENTITIES │ V.OBJECTS│AGGREGATES│ │
│  └─────────┴──────────┴──────────┘ │
├─────────────────────────────────────┤
│       INFRASTRUCTURE LAYER          │
└─────────────────────────────────────┘
```

### 2. FLEXT RESULT PATTERN (Railway-Oriented Programming)

**OBRIGATÓRIO**: Todo método público DEVE retornar `FlextResult[T]`
```python
def process_data(data: str) -> FlextResult[ProcessedData]:
    if not data:
        return FlextResult.fail("Empty data provided")
    return FlextResult.ok(ProcessedData(data))
```

### 3. NAMING CONVENTION STRICT

**OBRIGATÓRIO**: Todos os exports públicos DEVEM usar prefixo `Flext`
```python
# ✅ CORRETO
class FlextConfig: pass
class FlextContainer: pass
def FlextUtility(): pass

# ❌ INCORRETO  
class Config: pass
class Container: pass
def utility(): pass
```

## 🔧 PADRÕES DE IMPLEMENTAÇÃO

### 1. MODULE STRUCTURE PATTERN

Cada módulo DEVE seguir esta estrutura:
```python
"""Module docstring explaining purpose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.result import FlextResult
from flext_core.exceptions import FlextError

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict

# Classes here
class FlextSomething:
    """Class with Flext prefix."""
    pass

__all__: list[str] = [
    "FlextSomething",
]
```

### 2. ABSTRACTION → IMPLEMENTATION PATTERN

```python
# config_base.py - Abstractions
class FlextAbstractConfig(ABC):
    @abstractmethod
    def validate_config(self) -> FlextResult[None]:
        ...

# config.py - Concrete implementations  
class FlextConfig(FlextAbstractConfig):
    def validate_config(self) -> FlextResult[None]:
        return FlextResult.ok(None)
```

### 3. EXCEPTION HANDLING PATTERN

**PROIBIDO**: `raise Exception` ou `except Exception` sem FlextResult
```python
# ❌ INCORRETO
def bad_function():
    raise ValueError("Something wrong")

# ✅ CORRETO
def good_function() -> FlextResult[str]:
    try:
        # logic here
        return FlextResult.ok("success")
    except ValueError as e:
        return FlextResult.fail(f"Validation error: {e}")
```

### 4. TYPE SAFETY PATTERN

**OBRIGATÓRIO**: Use tipos centralizados de `flext_core.typings`
```python
from flext_core.typings import TAnyDict, TEntityId
from flext_core.constants import FlextEntityStatus

def process_entity(
    entity_id: TEntityId,
    data: TAnyDict,
    status: FlextEntityStatus
) -> FlextResult[bool]:
    return FlextResult.ok(True)
```

## 🚫 ANTI-PATTERNS - PROIBIDOS

### 1. CIRCULAR IMPORTS
```python
# ❌ PROIBIDO
# base_handlers.py
from flext_core.handlers import FlextHandler

# handlers.py  
from flext_core.base_handlers import FlextBaseHandler
```

### 2. LARGE MODULES (>1000 linhas)
```python
# ❌ PROIBIDO - Módulos >1000 linhas
# Quebrar em módulos menores com responsabilidades específicas
```

### 3. MISSING __all__ EXPORTS
```python
# ❌ PROIBIDO
class FlextSomething: pass
# Missing __all__

# ✅ CORRETO
class FlextSomething: pass
__all__: list[str] = ["FlextSomething"]
```

### 4. COMPATIBILITY LAYERS OVERUSE
```python
# ❌ EVITAR - Muitos *_compat.py
# Prefira migrar para APIs modernas
```

## 📁 LAYERED ARCHITECTURE STRUCTURE

### Layer 0: Foundation (Sem dependências internas)
- `result.py` - FlextResult pattern
- `exceptions.py` - FlextError hierarchy  
- `typings.py` - Type definitions
- `constants.py` - Centralized constants

### Layer 1: Infrastructure 
- `config_base.py` - Configuration abstractions
- `container.py` - Dependency injection
- `utilities.py` - Utility functions

### Layer 2: Domain Models
- `entities.py` - Domain entities
- `value_objects.py` - Value objects  
- `aggregate_root.py` - DDD aggregates

### Layer 3: Application Services
- `handlers.py` - CQRS handlers
- `commands.py` - Command patterns
- `validation.py` - Validation services

### Layer 4: Interface/Compatibility
- `*_compat.py` - Backwards compatibility
- `legacy.py` - Legacy support

## 🎯 QUALITY GATES OBRIGATÓRIOS

### 1. Code Quality
```bash
make lint      # Ruff linting - ZERO errors
make type-check # MyPy strict - ZERO errors  
make test      # 95%+ coverage - ALL passing
```

### 2. Architectural Compliance
- [ ] All classes use `Flext` prefix
- [ ] All public methods return `FlextResult[T]`
- [ ] No circular imports
- [ ] All modules have `__all__` exports
- [ ] All modules have `from __future__ import annotations`

### 3. Performance
- [ ] Modules <1000 lines
- [ ] Initialization <10ms
- [ ] Memory usage <50MB

## 🔄 DEPENDENCY MANAGEMENT

### Allowed Dependencies Flow
```
Layer 0 Foundation ←── Layer 1 Infrastructure
                   ←── Layer 2 Domain Models  
                   ←── Layer 3 Application Services
                   ←── Layer 4 Interface/Compatibility
```

### Prohibited Dependencies
- Layer 0 → Any other layer
- Layer 1 → Layer 2/3/4 (except utilities → domain)
- Circular dependencies between any layers

## 🛠️ IMPLEMENTATION CHECKLIST

Para cada novo módulo:

- [ ] Docstring explaining purpose
- [ ] `from __future__ import annotations`
- [ ] Proper imports from `flext_core.*`
- [ ] All classes use `Flext` prefix
- [ ] All public methods return `FlextResult[T]`
- [ ] `__all__` export list defined
- [ ] Type hints on all functions
- [ ] Unit tests with 95%+ coverage
- [ ] No circular imports
- [ ] No exceptions without FlextResult wrapping
- [ ] Constants from `flext_core.constants`
- [ ] Types from `flext_core.typings`

## 📊 METRICS & MONITORING

### Module Health Metrics
- Line count: <1000
- Cyclomatic complexity: <10
- Import count: <20
- Test coverage: >95%

### Architecture Compliance Score
```python
def calculate_compliance_score() -> float:
    """Calculate architectural compliance percentage."""
    # Implementation tracks all guidelines above
    pass
```

## 🚀 ECOSYSTEM IMPACT CONSIDERATIONS

Este módulo serve como foundation para **32+ projetos FLEXT**:

### Breaking Change Policy
1. **Semantic Versioning**: Major.Minor.Patch
2. **Deprecation Period**: 6 months minimum
3. **Migration Guides**: Required for breaking changes
4. **Compatibility Testing**: Against all dependent projects

### API Stability Requirements
- Public APIs marked with `@final` when stable
- Abstract base classes versioned separately  
- Legacy compatibility maintained for 2+ major versions

---

**COMPLIANCE**: Este documento é OBRIGATÓRIO para todo desenvolvimento em flext-core.
**UPDATES**: Atualizado conforme evolução arquitetural do projeto.
**ENFORCEMENT**: Validado automaticamente via CI/CD pipelines.