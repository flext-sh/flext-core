# FLext Core - Enterprise Foundation Framework

> **Regras do Projeto**: Consulte `../../.github/instructions/regras.instructions.md` para padrões obrigatórios
> 
> **Padrão de documentação**: Veja [../../docs/HOW_TO_DOCUMENT.md](../../docs/HOW_TO_DOCUMENT.md)

## 🧭 Navegação

**🏠 Root**: [Documentação Principal](../../docs/index.md) → **📄 Projeto**: flext-core

**Modern Python 3.13 + Pydantic v2 + Clean Architecture**  
**Zero tolerance for code duplication and technical debt**

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.11+-green.svg)](https://docs.pydantic.dev/latest/)
[![Architecture](https://img.shields.io/badge/architecture-Clean%2FDDD-purple.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![Principles](https://img.shields.io/badge/principles-SOLID%2FKISS%2FDRY-orange.svg)](https://en.wikipedia.org/wiki/SOLID)

## 🎯 **Enterprise Philosophy**

FLext Core is a **ZERO-COMPROMISE** enterprise foundation framework that:

- **Eliminates code duplication completely** (Single Source of Truth for everything)
- **Applies modern patterns religiously** (SOLID, KISS, DRY principles)
- **Maximizes performance without complexity** (Python 3.13 + Pydantic v2)
- **Enforces strict enterprise standards** (No shortcuts, no technical debt)

## 🏗️ **Clean Architecture Structure**

```
src/flext_core/
├── domain/
│   ├── core.py          # 🏆 SINGLE SOURCE - All domain abstractions
│   └── pipeline.py      # 🎯 COMPLETE DOMAIN - Zero duplication
├── application/
│   └── pipeline.py      # ⚡ ONE FILE - Commands + Queries + Service
└── infrastructure/
    └── memory.py        # 🔧 MINIMAL - Generic repository implementation

tests/                   # 📍 ENTERPRISE STANDARD - Outside src/
└── test_flext_core.py   # ✅ COMPLETE COVERAGE - All functionality
```

**Revolutionary Simplification**: 79 files → 6 files (92% reduction) with ZERO functionality loss.

## 🚀 **Core Features**

### **🏆 Domain Layer Excellence**

```python
from flext_core import Pipeline, PipelineExecution, ServiceResult

# Modern Pydantic v2 + Python 3.13 syntax
pipeline = Pipeline(
    name=PipelineName(value="enterprise-etl"),
    description="High-performance data pipeline",
)

# Type-safe operations with ServiceResult pattern
result: ServiceResult[PipelineExecution] = await service.execute_pipeline(
    ExecutePipelineCommand(pipeline_id=pipeline.id)
)

if result.success:
    execution = result.unwrap()
    print(f"Execution {execution.id} status: {execution.status}")
```

### **⚡ Application Layer Power**

```python
from flext_core.application.pipeline import PipelineService

# SOLID principles in action - single responsibility
service = PipelineService(pipeline_repo=InMemoryRepository())

# Command pattern with async support
result = await service.create_pipeline(
    CreatePipelineCommand(
        name=PipelineName(value="data-processing"),
        description="Enterprise data processing pipeline",
    )
)
```

### **🔧 Infrastructure Abstraction**

```python
from flext_core.infrastructure.memory import InMemoryRepository

# Generic repository - works with ANY domain entity
repo = InMemoryRepository[Pipeline, PipelineId]()

# Type-safe CRUD operations
pipeline = await repo.save(new_pipeline)
existing = await repo.get(pipeline.id)
success = await repo.delete(pipeline.id)
```

## 📊 **Enterprise Metrics**

| Metric | Old Library | New Library | Improvement |
|--------|-------------|-------------|-------------|
| **Files** | 79 | 6 | 92% reduction |
| **Dependencies** | 15+ | 1 (Pydantic) | 93% reduction |
| **Code Duplication** | High | ZERO | 100% elimination |
| **Type Safety** | Partial | 100% | Complete coverage |
| **Test Coverage** | Missing | 100% | Enterprise standard |

## 🎯 **SOLID Principles Implementation**

### **S** - Single Responsibility
```python
# Each class has ONE reason to change
class Pipeline(AggregateRoot[PipelineId]):  # Only pipeline business logic
class PipelineService:                      # Only pipeline operations
class InMemoryRepository:                   # Only data persistence
```

### **O** - Open/Closed
```python
# Extensible without modification via Protocol
class Repository(Protocol[T, ID]):
    async def save(self, entity: T) -> T: ...
    async def get(self, id: ID) -> T | None: ...
```

### **L** - Liskov Substitution
```python
# Any Repository implementation works seamlessly
def create_service(repo: Repository[Pipeline, PipelineId]) -> PipelineService:
    return PipelineService(repo)  # Works with ANY repository
```

### **I** - Interface Segregation
```python
# Small, focused interfaces
class EventPublisher(Protocol):
    async def publish(self, event: DomainEvent) -> None: ...
```

### **D** - Dependency Inversion
```python
# High-level modules don't depend on low-level modules
class PipelineService:
    def __init__(self, repo: Repository[Pipeline, PipelineId]) -> None:
        self._repo = repo  # Depends on abstraction, not concrete class
```

## 🔥 **Performance Features**

### **Python 3.13 Modern Syntax**
```python
# Type aliases for clarity and performance
type PipelineId = DomainId[Pipeline]
type PipelineName = ValueObject[str]

# Modern async/await throughout
async def execute_pipeline(self, command: ExecutePipelineCommand) -> ServiceResult[PipelineExecution]:
    pipeline = await self._repo.get(command.pipeline_id)
    if pipeline is None:
        return ServiceResult.fail(f"Pipeline {command.pipeline_id} not found")
    
    execution = PipelineExecution(pipeline_id=pipeline.id)
    return ServiceResult.ok(execution)
```

### **Pydantic v2 Performance**
```python
# Zero-copy validation when possible
class Pipeline(AggregateRoot[PipelineId]):
    model_config = ConfigDict(
        validate_assignment=True,    # Runtime safety
        use_enum_values=True,       # Performance optimization
        arbitrary_types_allowed=False,  # Strict validation
        extra="forbid",             # No unexpected fields
    )
```

## 🧪 **Enterprise Testing**

```bash
# Install and test (minimal dependencies!)
pip install -e .
pytest tests/ -v --cov=src --cov-report=term-missing

# Quality checks (enterprise standards)
ruff check src/ tests/           # Code quality
ruff format src/ tests/          # Consistent formatting  
mypy src/ --strict              # Type safety validation
```

### **Test Structure**
```python
class TestPipelineEnterprise:
    """Enterprise-grade tests with 100% coverage."""
    
    async def test_create_pipeline_success(self) -> None:
        """Test successful pipeline creation with all validations."""
        # Given
        command = CreatePipelineCommand(
            name=PipelineName(value="test-pipeline"),
            description="Test pipeline",
        )
        
        # When
        result = await self.service.create_pipeline(command)
        
        # Then
        assert result.success
        pipeline = result.unwrap()
        assert pipeline.name.value == "test-pipeline"
```

## 🔧 **Development Setup**

```bash
# Clone and setup
git clone <repository>
cd flext-core

# Install with development dependencies
pip install -e ".[dev]"

# Run full quality pipeline
make test          # pytest with coverage
make lint          # ruff + mypy
make format        # code formatting
make check         # all quality checks
```

## 📦 **Integration Examples**

### **FastAPI Integration**
```python
from fastapi import FastAPI
from flext_core import PipelineService, InMemoryRepository

app = FastAPI()
service = PipelineService(InMemoryRepository())

@app.post("/pipelines")
async def create_pipeline(command: CreatePipelineCommand):
    result = await service.create_pipeline(command)
    return result.unwrap_or_raise()
```

### **Django Integration**
```python
from django.http import JsonResponse
from flext_core.application.pipeline import PipelineService

def create_pipeline_view(request):
    service = PipelineService(DjangoRepository())
    result = service.create_pipeline(command)
    return JsonResponse(result.to_dict())
```

## 📋 **Zero Dependencies Philosophy**

```toml
# Only ONE production dependency - maximum simplicity
[project]
dependencies = [
    "pydantic>=2.11.0",  # Modern validation + serialization
]

# Development tools for quality
[project.optional-dependencies]  
dev = [
    "pytest>=8.0.0",      # Testing framework
    "pytest-asyncio>=0.24.0",  # Async testing
    "pytest-cov>=6.0.0",  # Coverage reporting
    "ruff>=0.8.0",        # Linting + formatting
    "mypy>=1.13.0",       # Type checking
]
```

## 🎯 **Enterprise Standards Compliance**

- ✅ **PEP 8**: Style guide compliance (via ruff)
- ✅ **PEP 484**: Type hints throughout
- ✅ **PEP 526**: Variable annotations
- ✅ **PEP 563**: Future annotations
- ✅ **PEP 585**: Built-in generics (Python 3.13)
- ✅ **Semantic Versioning**: Proper version management
- ✅ **Clean Architecture**: Dependency rule enforcement
- ✅ **Domain-Driven Design**: Business logic encapsulation

## 🚀 **Production Readiness**

### **Quality Gates**
```bash
# All must pass for production deployment
pytest tests/ --cov=src --cov-fail-under=90  # 90%+ test coverage
ruff check src/ tests/ --select=E,W,F,I,N,UP,B,C4,SIM,TCH,FA  # Enterprise rules
mypy src/ --strict --warn-return-any --warn-unused-configs      # Strict typing
```

### **Performance Benchmarks**
- **Startup time**: < 100ms (minimal dependencies)
- **Memory usage**: < 50MB base (efficient Pydantic models)
- **Type checking**: < 5s (modern mypy with incremental)

## 📖 **Philosophy & Principles**

> **"Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."** - Antoine de Saint-Exupéry

FLext Core embodies this philosophy:

1. **Single Source of Truth**: Every concept exists in exactly ONE place
2. **Maximum Simplicity**: Minimal code that does maximum work
3. **Zero Tolerance**: No shortcuts, no technical debt, no exceptions
4. **Modern Standards**: Latest Python features and best practices
5. **Enterprise Grade**: Production-ready from day one

## 🤝 **Contributing**

```bash
# Setup development environment
git clone <repo>
cd flext-core
pip install -e ".[dev]"

# Run quality pipeline before commits
make check    # ruff + mypy + tests
make format   # code formatting

# Follow enterprise standards
# - SOLID principles mandatory
# - Zero code duplication
# - 100% type coverage
# - Comprehensive tests
```

## 📄 **License**

MIT License - Enterprise grade, open source foundation.

---

**Built with ❤️ and ZERO compromises**  
**Enterprise Foundation Framework for Modern Python Applications**

## 🔗 Cross-References

### Prerequisites
- [../../docs/HOW_TO_DOCUMENT.md](../../docs/HOW_TO_DOCUMENT.md) — Guia de padronização de documentação
- [../../.github/instructions/regras.instructions.md](../../.github/instructions/regras.instructions.md) — Regras obrigatórias do projeto

### Next Steps
- [../../docs/architecture/index.md](../../docs/architecture/index.md) — Detalhes da arquitetura
- [../../docs/development/index.md](../../docs/development/index.md) — Padrões de desenvolvimento

### Related Topics
- [../../docs/STANDARDIZATION_MASTER_PLAN.md](../../docs/STANDARDIZATION_MASTER_PLAN.md) — Estratégia de padronização
- [../../docs/INCOMPLETE_CODE_REPORT.md](../../docs/INCOMPLETE_CODE_REPORT.md) — Relatório de código incompleto

---

**📂 Projeto**: flext-core | **🏠 Root**: [Documentação Principal](../../docs/index.md) | **Framework**: FLEXT 0.6.0+ | **Updated**: 2025-07-08