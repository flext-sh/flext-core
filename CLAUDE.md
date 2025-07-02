# CLAUDE.md - FLX-CORE MODULE

**Hierarchy**: PROJECT-SPECIFIC
**Project**: FLX Core - Foundation & Transformation Hub
**Status**: DEVELOPMENT
**Last Updated**: 2025-06-28

**Reference**: `/home/marlonsc/CLAUDE.md` → Universal principles
**Reference**: `/home/marlonsc/internal.invalid.md` → Cross-workspace issues
**Reference**: `../CLAUDE.md` → PyAuto workspace patterns

---

## 🎯 PROJECT-SPECIFIC CONFIGURATION

### Virtual Environment Usage

```bash
# MANDATORY: Use workspace venv
source /home/marlonsc/pyauto/.venv/bin/activate
# NOT project-specific venv
```

### Agent Coordination

```bash
# Read workspace coordination first
cat /home/marlonsc/pyauto/.token | tail -5
# Use project .token only for project-specific coordination
echo "FLX_CORE_ANALYSIS_$(date)" >> .token
```

## 🔍 CRITICAL LESSONS LEARNED

### **INVESTIGATION TRUTH: Always Verify Implementation**

**What Happened**: Made assumptions about implementation status without deep investigation
**Reality**: Claimed 0% authentication when it was 75% complete (32KB of working code)
**Lesson**: ALWAYS investigate actual code before making claims

```bash
# MANDATORY: Investigation protocol for this project
1. ls -la src/module/ to check file sizes (empty vs implemented)
2. grep -r "NotImplementedError" --include="*.py" | wc -l for real counts
3. head -100 suspected_empty_file.py to verify actual content
4. NEVER trust file names or patterns - READ THE CODE
```

### **Project Reality Check**

Based on REAL analysis of flx-meltano-enterprise:

| Component      | Initial Claim    | Actual Reality                          | Lesson                        |
| -------------- | ---------------- | --------------------------------------- | ----------------------------- |
| Domain Layer   | "Good design"    | 3,721 lines of EXCELLENT implementation | Undersold the achievement     |
| Authentication | "0% functional"  | 75% complete, 32KB UserService          | Completely wrong assessment   |
| gRPC           | "Empty stubs"    | 3,242 lines fully implemented           | Failed to check file content  |
| Plugin System  | "0% implemented" | 40% done with discovery/loader          | Missed partial implementation |

**Total NotImplementedError**: 289 (not 2,166 as claimed)

## 📁 PROJECT STRUCTURE

```
flx-core/
├── docs/
│   ├── ARCHITECTURAL_TRUTH.md     # Real findings from investigation
│   └── architecture/
│       ├── ADR-001-modularization-strategy.md   # Based on real analysis
│       ├── ADR-002-authentication-architecture.md # Corrected metrics
│       └── ADR-003-plugin-system-design.md      # Actual gaps identified
├── src/
│   └── flx_core/                  # Extracted from flx-meltano-enterprise
│       ├── domain/                # 🎯 MOST CORE LAYER
│       │   ├── pydantic_base.py   # 🏆 ABSOLUTE FOUNDATION
│       │   ├── base.py            # Domain type system
│       │   ├── advanced_types.py  # Python 3.13 types
│       │   ├── entities.py        # Core business entities
│       │   ├── value_objects.py   # Domain value objects
│       │   └── ports.py           # Clean architecture boundaries
│       ├── events/                # Event-driven foundation
│       │   └── event_bus.py       # Lato-based event system
│       ├── application/           # Use case orchestration
│       │   ├── base_application.py
│       │   ├── commands.py
│       │   └── handlers.py
│       ├── config/                # Configuration management
│       │   └── domain_config.py   # Centralized config
│       └── infrastructure/        # External adapters
│           └── persistence/       # Database integration
└── CLAUDE.md                      # This file - lessons and reality
```

## 🎯 CORE ARCHITECTURE ANALYSIS

### **HIERARQUIA DE DEPENDÊNCIAS (Mais Interna → Mais Externa)**

```
🏆 NÚCLEO ABSOLUTO: domain/pydantic_base.py
    ├── DomainBaseModel      # Foundation for all domain objects
    ├── DomainValueObject    # Immutable value objects
    ├── DomainEntity         # Entities with identity
    ├── DomainAggregateRoot  # Aggregates with event sourcing
    ├── DomainEvent          # Immutable domain events
    └── ServiceResult        # Result type for operations

↑ DEPENDS ON ↑

🎯 SISTEMA DE TIPOS: domain/base.py + advanced_types.py
    ├── DomainId             # Fundamental identifier type
    ├── UserId, TenantId     # Business-specific IDs
    ├── PipelineName         # Domain value types
    └── Python 3.13 aliases # Modern type system

↑ DEPENDS ON ↑

🏗️ MODELOS DE DOMÍNIO: entities.py + value_objects.py
    ├── Pipeline             # Core aggregate root
    ├── PipelineExecution    # Execution tracking entity
    ├── Plugin               # Plugin management entity
    ├── ExecutionStatus      # Value object enum
    └── Duration             # Time value object

↑ DEPENDS ON ↑

🔌 INTERFACES/BOUNDARIES: ports.py
    ├── PipelineManagementPort    # Primary port
    ├── PluginManagementPort      # Primary port
    ├── Repository interfaces     # Secondary ports
    └── EventBusPort              # Event integration

↑ DEPENDS ON ↑

⚡ SISTEMA DE EVENTOS: events/event_bus.py
    ├── EventBusProtocol     # Event bus interface
    ├── Lato integration     # DI container events
    └── Domain event routing # Event distribution

↑ DEPENDS ON ↑

⚙️ CONFIGURAÇÃO: config/domain_config.py
    ├── Environment config   # Pydantic Settings
    ├── Business constants   # Domain parameters
    └── Type-safe validation # Configuration validation

↑ DEPENDS ON ↑

🎯 CASOS DE USO: application/*
    ├── Command handlers     # Business operations
    ├── Domain services      # Complex business logic
    └── Application services # Use case orchestration

↑ DEPENDS ON ↑

🗃️ ADAPTADORES: infrastructure/*
    ├── Repository implementations  # Data persistence
    ├── Database models            # ORM mapping
    └── Unit of work pattern       # Transaction management
```

### **CARACTERÍSTICAS DO CORE**

#### **🏆 Pydantic-Centric Architecture**

- **100% baseado em Pydantic v2**: Validação e serialização nativas
- **Python 3.13 compatibility**: Sistema de tipos moderno
- **Enterprise configuration**: Configuração robusta para produção

#### **🎯 Domain-Driven Design Completo**

- **Aggregates**: Pipeline como aggregate root principal
- **Entities**: PipelineExecution, Plugin com identidade
- **Value Objects**: ExecutionStatus, Duration, PipelineName
- **Domain Events**: Event sourcing para auditoria
- **Specifications**: Business rules encapsuladas

#### **🔌 Clean Architecture (Hexagonal)**

- **Primary Ports**: PipelineManagementPort, PluginManagementPort
- **Secondary Ports**: Repository interfaces, EventBusPort
- **Clear boundaries**: Separation between domain and infrastructure
- **Dependency inversion**: Infrastructure depends on domain

#### **⚡ Event-Driven Foundation**

- **Lato integration**: DI container com eventos
- **Domain events**: Pipeline events, execution events
- **Async support**: Full async/await throughout
- **Event sourcing**: Complete audit trail

#### **📊 Type Safety & Validation**

- **Strict typing**: mypy --strict compliance
- **Runtime validation**: Pydantic model validation
- **Business constraints**: Domain rules in value objects
- **Error handling**: ServiceResult pattern

## 🔗 MODULE INTEGRATION MAP

### **Core as Central Hub**

FLX Core serves as the foundation for ALL other modules:

```mermaid
graph TB
    CORE[flx-core<br/>Foundation & Hub]

    subgraph "Security Layer"
        AUTH[flx-auth<br/>100% Complete<br/>Authentication]
        AUTH -->|provides auth| CORE
    end

    subgraph "Extension Layer"
        PLUGIN[flx-plugin<br/>40% Complete<br/>Plugin System]
        PLUGIN -->|extends| CORE
    end

    subgraph "Data Processing"
        MELTANO[flx-meltano<br/>100% Complete<br/>ETL Pipelines]
        MELTANO -->|uses domain| CORE
    end

    subgraph "Interface Layer"
        API[flx-api<br/>100% Complete<br/>REST Gateway]
        WEB[flx-web<br/>100% Complete<br/>Django UI]
        GRPC[flx-grpc<br/>100% Complete<br/>RPC Services]
        CLI[flx-cli<br/>95% Complete<br/>Terminal UI]

        API -->|commands| CORE
        WEB -->|queries| CORE
        GRPC -->|handlers| CORE
        CLI -->|execution| CORE
    end

    subgraph "Operations Layer"
        OBS[flx-observability<br/>100% Complete<br/>Monitoring]
        OBS -->|events| CORE
    end

    API -.->|authenticates| AUTH
    WEB -.->|sessions| AUTH
    GRPC -.->|tokens| AUTH
    CLI -.->|credentials| AUTH

    API -.->|metrics| OBS
    WEB -.->|telemetry| OBS
    GRPC -.->|traces| OBS

    MELTANO -.->|plugins| PLUGIN
    CLI -.->|extensions| PLUGIN
```

### **Dependency Relationships**

#### **🌐 [flx-api](../flx-api/) - Enterprise REST API Gateway**

- **Purpose**: RESTful API interface for all platform operations
- **Key Features**:
  - FastAPI with 100% async/await
  - Thread-safe pipeline storage
  - Rate limiting (100 req/min)
  - OpenAPI documentation
- **Dependencies from Core**:
  - `CommandBus` & `QueryBus` for CQRS pattern
  - Domain models (`Pipeline`, `Plugin`, `Execution`)
  - Service Result pattern for error handling
  - Infrastructure containers for DI
- **Integration Example**:

```python
from flx_core.application import CommandBus
from flx_core.domain.commands import CreatePipelineCommand
from flx_auth.jwt_service import JWTService

@router.post("/pipelines")
async def create_pipeline(cmd: CreatePipelineCommand, bus: CommandBus):
    result = await bus.execute(cmd)
    return result.unwrap_or_raise()
```

#### **🖥️ [flx-web](../flx-web/) - Enterprise Web Dashboard**

- **Purpose**: Django-based web interface for operations and monitoring
- **Key Features**:
  - Server-side rendering with Django templates
  - Real-time monitoring dashboards
  - Pipeline management UI
  - Admin interface for user management
- **Dependencies from Core**:
  - Django integration configuration
  - Persistence models via SQLAlchemy
  - Event subscriptions for real-time updates
  - gRPC client for backend communication
- **Integration Example**:

```python
from flx_core.config.django_integration import FlxDjangoConfig
from flx_core.infrastructure.persistence.models import PipelineModel
from flx_grpc.client import FlxGrpcClient

class PipelineView(View):
    def get(self, request):
        client = FlxGrpcClient()
        pipelines = client.list_pipelines()
        return render(request, 'pipelines.html', {'pipelines': pipelines})
```

#### **📡 [flx-grpc](../flx-grpc/) - High-Performance RPC Services**

- **Purpose**: gRPC services for internal microservice communication
- **Key Features**:
  - 50+ implemented RPC methods
  - Bidirectional streaming support
  - Health checking service
  - Interceptors for auth/metrics
- **Dependencies from Core**:
  - Command handlers for business logic
  - Domain-to-proto converters
  - Event bus for async notifications
  - Service implementations
- **Integration Example**:

```python
from flx_core.application.handlers import PipelineCommandHandler
from flx_core.domain.entities import Pipeline
from flx_grpc.converters import domain_to_proto

class FlxServiceImpl(FlxServiceServicer):
    async def CreatePipeline(self, request, context):
        command = proto_to_command(request)
        result = await self.handler.handle(command)
        return domain_to_proto(result)
```

#### **🖥️ [flx-cli](../flx-cli/) - Developer Command Line Interface**

- **Purpose**: Rich CLI for developers and operators
- **Key Features**:
  - Click-based command structure
  - Interactive pipeline builder
  - Real-time log streaming
  - Configuration management
- **Dependencies from Core**:
  - CLI application framework
  - Command execution engine
  - Configuration loading
  - Plugin discovery
- **Integration Example**:

```python
from flx_core.application.cli import CliApplication
from flx_core.domain.commands import ExecutePipelineCommand
import click

@click.command()
@click.option('--pipeline', required=True)
def run(pipeline: str):
    app = CliApplication()
    cmd = ExecutePipelineCommand(pipeline_id=pipeline)
    result = app.execute(cmd)
    click.echo(result)
```

#### **📊 [flx-meltano](../flx-meltano/) - Data Pipeline Engine**

- **Purpose**: Meltano integration for ETL/ELT pipelines
- **Key Features**:
  - 241KB of enterprise Meltano integration
  - State management with backup/restore
  - Plugin orchestration
  - Singer protocol support
- **Dependencies from Core**:
  - Domain entities for pipeline modeling
  - Execution engine for async operations
  - Event bus for progress tracking
  - Configuration management
- **Integration Example**:

```python
from flx_core.domain.entities import Pipeline
from flx_core.execution.engine import ExecutionEngine
from flx_meltano.orchestrator import MeltanoOrchestrator

orchestrator = MeltanoOrchestrator(engine)
result = await orchestrator.run_pipeline(pipeline)
```

#### **📈 [flx-observability](../flx-observability/) - Monitoring & Telemetry**

- **Purpose**: Comprehensive monitoring, metrics, and tracing
- **Key Features**:
  - Prometheus metrics collection
  - OpenTelemetry tracing
  - Business KPI tracking
  - Health check framework
- **Dependencies from Core**:
  - Event bus for metric events
  - Domain events for business metrics
  - Configuration for thresholds
  - Service health interfaces
- **Integration Example**:

```python
from flx_core.events import EventBus, PipelineExecutedEvent
from flx_observability.business_metrics import BusinessMetrics

@event_bus.subscribe(PipelineExecutedEvent)
async def track_pipeline_metrics(event: PipelineExecutedEvent):
    metrics.record_execution(
        pipeline_id=event.pipeline_id,
        duration=event.duration,
        success=event.success
    )
```

#### **🔐 [flx-auth](../flx-auth/) - Security & Authentication**

- **Purpose**: Enterprise authentication and authorization
- **Key Features**:
  - JWT with RS256 algorithm
  - Token blacklisting
  - Session management
  - Role-based access control
- **Bidirectional Relationship**:
  - **From Core**: Domain models (User, Role, Permission)
  - **To Core**: Authentication services, security context
- **Integration Example**:

```python
from flx_core.domain.entities import User
from flx_auth.user_service import UserService
from flx_auth.tokens import TokenManager

user = User(email="user@example.com")
token = await auth_service.create_token(user)
```

#### **🔌 [flx-plugin](../flx-plugin/) - Plugin System**

- **Purpose**: Dynamic plugin loading and management
- **Key Features**:
  - Plugin discovery
  - Dependency resolution
  - Hot reload (planned)
  - Version management
- **Embedded in Core**:
  - Currently in `flx_core.plugins`
  - To be extracted as separate module
- **Integration Example**:

```python
from flx_plugin.discovery import PluginDiscovery
from flx_plugin.loader import PluginLoader

discovery = PluginDiscovery("/opt/flx/plugins")
plugins = await discovery.scan()
loader = PluginLoader()
await loader.load_plugin(plugins[0])
```

### **Cross-Module Integration Points**

## 🚨 PROJECT-SPECIFIC ISSUES

### **Import Strategy for Extraction**

When extracting from flx-meltano-enterprise:

- ✅ Domain layer can be extracted almost as-is (95% complete)
- ✅ Authentication completed (100% - token storage implemented)
- ✅ gRPC services are fully functional, just need extraction
- ⏳ Plugin system needs hot reload implementation (40% → 100%)
- ⏳ CLI commands need to be consolidated from scattered locations

### **Module Initialization Order**

For proper startup, modules must be initialized in this order:

1. **flx-core** - Foundation services
2. **flx-auth** - Security context
3. **flx-observability** - Monitoring setup
4. **flx-plugin** - Extension loading
5. **flx-grpc** - RPC server
6. **flx-api** - REST gateway
7. **flx-web** - Web interface
8. **flx-cli** - Command interface
9. **flx-meltano** - Pipeline engine

### **Shared Infrastructure Requirements**

All modules share these infrastructure services:

```yaml
# docker-compose.yml for development
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: flx_core
      POSTGRES_USER: flx
      POSTGRES_PASSWORD: flx
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### **Module Communication Patterns**

#### **1. Command/Query Pattern (CQRS)**

```python
# Used by: flx-api, flx-cli, flx-grpc
from flx_core.application import CommandBus, QueryBus

# Commands modify state
command_bus.execute(CreatePipelineCommand(...))

# Queries read state
result = query_bus.execute(GetPipelineQuery(id=123))
```

#### **2. Event-Driven Communication**

```python
# Used by: flx-meltano, flx-observability
from flx_core.events import EventBus, DomainEvent

@dataclass
class PipelineStartedEvent(DomainEvent):
    pipeline_id: str
    started_at: datetime

event_bus.publish(PipelineStartedEvent(...))
```

#### **3. Service Result Pattern**

```python
# Used by: ALL modules for error handling
from flx_core.domain.advanced_types import ServiceResult

def process() -> ServiceResult[Pipeline]:
    if error:
        return ServiceResult.failure("Error message")
    return ServiceResult.success(pipeline)
```

### **Dependency Management**

```toml
# pyproject.toml dependencies based on REAL usage
[tool.poetry.dependencies]
python = "^3.13"
pydantic = "^2.5.0"       # Domain validation
sqlalchemy = "^2.0.0"     # Models already use this
grpcio = "^1.60.0"        # gRPC fully implemented
redis = "^5.0.0"          # For completing token storage
lato = "^0.3.0"           # DI already in use
bcrypt = "^4.1.0"         # Password hashing working
pyjwt = "^2.8.0"          # JWT implementation complete
```

## 🎯 IMPLEMENTATION PRIORITIES

### **Week 1: Core Infrastructure** ✅ COMPLETED

1. ✅ Extract domain layer (3,721 lines) - preserved all functionality
2. ✅ Extract authentication (70KB) - working code maintained
3. ✅ Complete token storage - 6 methods implemented
4. ✅ Extract all modules - 8 modules created

### **Week 2: Module Enhancement** 🚧 IN PROGRESS

1. ⏳ Extract flx-plugin as separate module (40% → 100%)
2. ⏳ Extract flx-cli as separate module (95% → 100%)
3. ⏳ Add hot reload to plugin system
4. ⏳ Complete remaining .env.example files

### **Week 3: Integration Testing** 📅 PLANNED

1. Create integration test suite
2. Docker Compose for local development
3. CI/CD pipeline setup
4. Performance benchmarking

### **Week 4: Production Readiness** 📅 PLANNED

1. Kubernetes manifests for each module
2. Helm charts for deployment
3. Monitoring dashboards (Grafana)
4. Documentation site (MkDocs)

## 📊 SUCCESS METRICS

- Domain extraction preserves 95%+ functionality
- Authentication reaches 100% completion (6 methods)
- Plugin hot reload implemented and tested
- All 289 NotImplementedError resolved
- Zero assumptions in documentation

## 🔒 PROJECT .ENV SECURITY REQUIREMENTS

### MANDATORY .env Variables

```bash
# WORKSPACE (required for all PyAuto projects)
WORKSPACE_ROOT=/home/marlonsc/pyauto
PYTHON_VENV=/home/marlonsc/pyauto/.venv
DEBUG_MODE=true

# FLX-CORE SPECIFIC
JWT_SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=RS256
DATABASE_URL=postgresql://user:pass@localhost/flx_core
REDIS_URL=redis://localhost:6379/0
GRPC_PORT=50051
PLUGIN_WATCH_DIRECTORY=/path/to/plugins
```

### MANDATORY CLI Usage

```bash
# ALWAYS source workspace venv + project .env + debug CLI
source /home/marlonsc/pyauto/.venv/bin/activate
source .env
python -m flx_core.cli command --debug --verbose
```

### SECURITY WARNINGS

- 🚨 NEVER modify .env without explicit user authorization
- ❌ NEVER use CLI without --debug flag
- ✅ .env is SINGLE SOURCE OF TRUTH for this project

## 📝 DOCUMENTATION STANDARDS

### **Truth-Based Documentation**

1. **INVESTIGATE FIRST**: Read actual code, check file sizes, count real issues
2. **VERIFY CLAIMS**: grep for actual NotImplementedError, don't assume
3. **BE SPECIFIC**: Exact line numbers and file paths
4. **ADMIT UNCERTAINTY**: Mark unknowns as "NEEDS VERIFICATION"

### **Anti-Patterns to Avoid**

- ❌ Assuming empty files based on names
- ❌ Inventing metrics without verification
- ❌ Claiming 0% implementation without checking
- ❌ Trusting patterns over actual code

## 🎯 NEXT ACTIONS

1. Complete extraction of working components from flx-meltano-enterprise
2. Implement missing token storage backends (6 methods)
3. Add hot reload to plugin system
4. Update all documentation with verified metrics
5. Create comprehensive test suite

---

**MANTRA FOR THIS PROJECT**: **INVESTIGATE DEEP, VERIFY EVERYTHING, DOCUMENT TRUTH**

**Remember**: The flx-meltano-enterprise codebase is EXCELLENT with minor gaps. Extract and complete, don't rewrite.
