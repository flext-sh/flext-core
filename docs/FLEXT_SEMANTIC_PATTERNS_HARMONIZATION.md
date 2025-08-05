# FLEXT Semantic Patterns Harmonization Guide

**Status**: ACTIVE HARMONIZATION  
**Version**: 1.0.0  
**Objetivo**: Eliminar duplicações e conflitos entre padrões semânticos no ecossistema FLEXT  
**Aplicação**: Todos os 33 projetos do ecossistema FLEXT  

## 🎯 PROBLEMAS IDENTIFICADOS - DUPLICAÇÕES CRÍTICAS

### ❌ PROBLEMA 1: Multiple FlextEntity Definitions

**Conflito**: 3 diferentes definições de FlextEntity

- `models.py`: FlextEntity (abstract, validate_business_rules)
- `entities.py`: FlextEntity (abstract, validate_domain_rules)
- `shared_domain.py`: ConcreteFlextEntity (implementação de teste)

**RESOLUÇÃO HARMÔNICA**:

- **ÚNICA FONTE**: `models.py` será a definição canônica
- **MÉTODO PADRÃO**: `validate_business_rules()` (não domain_rules)
- **MIGRATION**: Todas as referências migram para models.py
- **DEPRECATION**: entities.py será marcado como deprecated

### ❌ PROBLEMA 2: Type System Duplication

**Conflito**: 3 sistemas de tipos sobrepostos

- `types.py`: Legacy hierarchical types (FlextTypes)
- `semantic_types.py`: New semantic types (FlextTypes)
- `flext_types.py`: Flat type aliases (TAnyDict, etc.)

**RESOLUÇÃO HARMÔNICA**:

- **SISTEMA PRINCIPAL**: `semantic_types.py` (hierárquico moderno)
- **SISTEMA LEGACY**: `types.py` (compatibilidade temporária)
- **FLAT ALIASES**: `flext_types.py` (deprecated, apenas exports)
- **MIGRATION PATH**: Gradual migration ao semantic_types.py

### ❌ PROBLEMA 3: Configuration System Conflicts

**Conflito**: 4 sistemas de configuração independentes

- `config.py`: FlextBaseSettings (Pydantic)
- `config_models.py`: Configuration models e TypedDicts
- `config_hierarchical.py`: Hierarchical configuration manager
- `models.py`: FlextConfig (semantic pattern)

**RESOLUÇÃO HARMÔNICA**:

- **FOUNDATION**: `models.py` FlextConfig (semantic pattern base)
- **SETTINGS**: `config.py` FlextBaseSettings (environment integration)
- **MODELS**: `config_models.py` (concrete implementations)
- **HIERARCHY**: `config_hierarchical.py` (management layer)

### ❌ PROBLEMA 4: Observability Pattern Inconsistency

**Conflito**: 3 abordagens de observability

- `observability.py`: Protocol-based minimal implementations
- `loggings.py`: StructLog-based logging system
- `models.py`: FlextObs namespace (semantic pattern)

**RESOLUÇÃO HARMÔNICA**:

- **PROTOCOLS**: `observability.py` (interface contracts)
- **IMPLEMENTATION**: `loggings.py` (concrete logging)
- **NAMESPACE**: `models.py` FlextObs (semantic organization)

## 🏗️ PADRÃO SEMÂNTICO HARMONIZADO

### 1. SEMANTIC NAMING CONVENTION (Unificado)

**Padrão Universal**: `Flext[Domain][Type][Context]`

**Domínios Canônicos**:

```python
FlextCore.*       # Foundation patterns (Result, Container, etc.)
FlextData.*       # Data integration (Connection, Record, Schema)
FlextAuth.*       # Authentication/Authorization (Token, User, Policy)
FlextObs.*        # Observability (Logger, Metrics, Tracer)
FlextSinger.*     # Singer protocol (Tap, Target, Stream)
FlextConfig.*     # Configuration (Settings, Environment, Validation)
```

**Tipos Canônicos**:

```python
*.Model          # Pydantic models (mutable)
*.Value          # Value objects (immutable)
*.Entity         # Domain entities (identity-based)
*.Config         # Configuration models
*.Protocol       # Interface contracts
*.Factory        # Creation patterns
```

### 2. FOUNDATION CLASSES HARMONIZED

#### A. Universal Base Model

```python
# CANONICAL: models.py
class FlextModel(BaseModel):
    """Universal base for ALL FLEXT models."""
    metadata: dict[str, object] = Field(default_factory=dict)
    
    def validate_business_rules(self) -> FlextResult[None]:
        """STANDARD validation method name."""
        return FlextResult.ok(None)
```

#### B. Domain Entity Pattern

```python
# CANONICAL: models.py
class FlextEntity(FlextModel, ABC):
    """Identity-based entities - ÚNICA definição."""
    id: str = Field(description="Unique identifier")
    version: int = Field(default=1, description="Optimistic locking")
    domain_events: list[dict[str, object]] = Field(default_factory=list, exclude=True)
    
    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]:
        """MUST implement business validation."""
```

#### C. Value Object Pattern

```python
# CANONICAL: models.py
class FlextValue(FlextModel, ABC):
    """Immutable value objects - ÚNICA definição."""
    model_config = ConfigDict(frozen=True)
    
    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]:
        """MUST implement business validation."""
```

### 3. TYPE SYSTEM HARMONIZATION

#### A. Semantic Types (PRIMARY)

```python
# PRIMARY: semantic_types.py
class FlextTypes:
    """Hierarchical semantic type system - PRINCIPAL."""
    
    class Core:
        """Core patterns - Result, Container, etc."""
        type Result[T, E] = T | E
        type Factory[T] = Callable[[], T]
        type Predicate[T] = Callable[[T], bool]
    
    class Data:
        """Data integration types."""
        type Connection = object
        type Record = dict[str, object]
        type Schema = dict[str, object]
    
    class Auth:
        """Authentication types."""
        type Token = str
        type User = dict[str, object]
        type Policy = dict[str, object]
```

#### B. Legacy Compatibility

```python
# COMPATIBILITY: types.py (DEPRECATED)
class FlextTypesCompat:
    """Legacy aliases - TEMPORARY during migration."""
    # Aliases to semantic_types for backward compatibility
    Result = FlextTypes.Core.Result
    Connection = FlextTypes.Data.Connection
    # ... etc
```

#### C. Flat Aliases (DEPRECATED)

```python
# DEPRECATED: flext_types.py
# Export only for backward compatibility
TAnyDict = dict[str, object]  # Use FlextTypes.Data.Dict instead
TEntity = str  # Use FlextTypes.Domain.EntityId instead
```

### 4. CONFIGURATION HARMONIZATION

#### A. Semantic Configuration Base

```python
# FOUNDATION: models.py
class FlextConfig(FlextValue):
    """Semantic configuration pattern - BASE."""
    
    def validate_business_rules(self) -> FlextResult[None]:
        """Override for specific validation."""
        return FlextResult.ok(None)
```

#### B. Environment Integration

```python
# INTEGRATION: config.py
class FlextBaseSettings(FlextConfig, BaseSettings):
    """Environment-aware settings - EXTENDS semantic base."""
    
    class Config:
        env_prefix = "FLEXT_"
        case_sensitive = False
```

#### C. Concrete Models

```python
# IMPLEMENTATIONS: config_models.py
class FlextDatabaseConfig(FlextBaseSettings):
    """Database configuration - CONCRETE implementation."""
    host: str = "localhost"
    port: int = 5432
    # ... specific fields
```

### 5. OBSERVABILITY HARMONIZATION

#### A. Protocol Contracts

```python
# CONTRACTS: observability.py
@runtime_checkable
class FlextLoggerProtocol(Protocol):
    """Logger interface contract."""
    def info(self, message: str, **context: object) -> None: ...
    # ... other methods
```

#### B. Semantic Namespace

```python
# NAMESPACE: models.py
class FlextObs:
    """Observability namespace - semantic organization."""
    # Extended by projects like flext-observability
    pass
```

#### C. Concrete Implementation  

```python
# IMPLEMENTATION: loggings.py
class FlextLogger:
    """Concrete logger implementation."""
    # Implements FlextLoggerProtocol
    pass
```

## 📋 MIGRATION STRATEGY - SYSTEMATIC HARMONIZATION

### PHASE 1: Foundation Harmonization (Esta fase)

#### Step 1.1: Consolidate Entity Definitions

- [x] Manter apenas `models.py` FlextEntity como canônico
- [x] Marcar `entities.py` como deprecated
- [x] Atualizar shared_domain para usar validate_business_rules
- [x] Migrar todos os imports

#### Step 1.2: Harmonize Type Systems

- [ ] Estabelecer `semantic_types.py` como sistema principal
- [ ] Migrar aliases críticos de `types.py` para compatibilidade
- [ ] Deprecar `flext_types.py` exports desnecessários
- [ ] Atualizar imports em todos os módulos

#### Step 1.3: Unify Configuration Patterns

- [ ] Estabelecer hierarquia: FlextConfig → FlextBaseSettings → Concrete Models
- [ ] Consolidar duplicações entre config módulos
- [ ] Migrar configurações para padrão hierárquico
- [ ] Atualizar factory functions

#### Step 1.4: Standardize Observability

- [ ] Estabelecer protocols como interfaces
- [ ] Migrar implementations para padrão unificado
- [ ] Consolidar namespace semantic organization
- [ ] Atualizar logging patterns

### PHASE 2: Module-by-Module Application

- [ ] Aplicar padrões harmonizados em cada módulo flext-core
- [ ] Atualizar imports e exports  
- [ ] Validar consistência com testes
- [ ] Documentar mudanças

### PHASE 3: Ecosystem Propagation

- [ ] Aplicar em projetos de infraestrutura
- [ ] Aplicar em Singer ecosystem (15 projetos)
- [ ] Validar Go-Python bridge compatibility
- [ ] Atualizar documentação

## ✅ VALIDAÇÃO DA HARMONIZAÇÃO

### Critérios de Sucesso

1. **Zero Duplicação**: Cada conceito tem UMA definição canônica
2. **Naming Consistency**: Todos seguem Flext[Domain][Type][Context]
3. **Migration Path**: Backward compatibility mantida durante transição
4. **Type Safety**: MyPy strict compliance maintained
5. **Test Coverage**: Todos os testes passam após harmonização

### Quality Gates

```bash
# Validação completa
make validate                    # Lint + Type + Test
grep -r "validate_domain_rules"  # Must be ZERO results
grep -r "from.*entities import"  # Must migrate to models
mypy --strict                    # Zero type errors
pytest -v                       # All tests pass
```

## 🎯 IMMEDIATE ACTIONS

### CURRENT PRIORITY

1. **Finish Entity Consolidation** - Complete validate_business_rules migration
2. **Fix Failing Tests** - Resolve 58 test failures with harmonized patterns  
3. **Type System Migration** - Establish semantic_types.py as primary
4. **Configuration Unification** - Implement hierarchical config pattern

### NEXT SPRINT

1. Apply harmonized patterns throughout flext-core
2. Update all imports and exports
3. Validate with comprehensive test suite
4. Document harmonization for ecosystem projects

---

**COMMIT TO HARMONIZATION**: Esta harmonização resolve conflitos fundamentais que estavam causando confusion e duplicação no ecossistema FLEXT. Cada conceito agora tem uma única fonte de verdade clara.
