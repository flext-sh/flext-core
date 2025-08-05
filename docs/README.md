# FLEXT Core Documentation

**Documentação baseada na implementação atual**

## 🎯 Visão Geral

FLEXT Core é uma biblioteca Python para padrões de arquitetura limpa, railway-oriented programming e dependency injection. Esta documentação reflete a implementação REAL em src/flext_core/.

## 📖 Estrutura da Documentação

### 🚀 **Getting Started**

- [**Installation Guide**](getting-started/installation.md) - Setup e configuração
- [**Quick Start Guide**](getting-started/quickstart.md) - Uso básico

### 🏗️ **Architecture**

- [**Architecture Overview**](architecture/overview.md) - Visão geral da arquitetura

### 📚 **API Reference**

- [**Core API**](api/core.md) - FlextResult, FlextContainer, FlextBaseSettings
- [**Patterns API**](api/patterns.md) - Commands, Handlers, Validation

### ⚙️ **Configuration**

- [**Configuration Overview**](configuration/overview.md) - Sistema de configuração

### 🛠️ **Development**

- [**Best Practices**](development/best-practices.md) - Práticas recomendadas

### 💡 **Examples**

- [**Examples Overview**](examples/overview.md) - Exemplos práticos validados

## 🔧 Core Patterns

### FlextResult[T] - Railway Pattern

```python
from flext_core import FlextResult

# Type-safe error handling
def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult.fail("Division by zero")
    return FlextResult.ok(a / b)

result = divide(10, 2)
if result.success:
    print(f"Result: {result.data}")  # 5.0
else:
    print(f"Error: {result.error}")
```

### FlextContainer - Dependency Injection

```python
from flext_core import FlextContainer

container = FlextContainer()

# Register service
database = DatabaseService("sqlite:///app.db")
reg_result = container.register("database", database)

# Retrieve service
service_result = container.get("database")
if service_result.success:
    db = service_result.data
```

### FlextBaseSettings - Configuration

```python
from flext_core import FlextBaseSettings

class AppSettings(FlextBaseSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str = "sqlite:///app.db"

    class Config:
        env_prefix = "APP_"

settings = AppSettings()  # Loads from env vars
```

## 🧪 Quick Start

### 1. Install

```bash
pip install flext-core
# or
poetry add flext-core
```

### 2. Basic Usage

```python
from flext_core import FlextResult, FlextContainer

# Railway pattern example
def process_user(user_data: dict) -> FlextResult[dict]:
    if not user_data.get("email"):
        return FlextResult.fail("Email required")

    processed = {
        "email": user_data["email"].lower(),
        "processed": True
    }
    return FlextResult.ok(processed)

# DI example
container = FlextContainer()
container.register("config", {"db_url": "sqlite:///app.db"})

config_result = container.get("config")
if config_result.success:
    config = config_result.data
    print(f"Database: {config['db_url']}")
```

## 📊 Implementation Status

### ✅ **Functional & Tested:**

- FlextResult[T] railway pattern
- FlextContainer dependency injection
- FlextBaseSettings configuration
- Basic logging support

### 🔧 **Available API (In Development):**

- Domain patterns (FlextEntity, FlextValueObject)
- Command/Handler patterns (FlextCommands, FlextHandlers)
- Validation patterns (FlextValidation)

### 📋 **Planned:**

- Complete CQRS implementation
- Event sourcing system
- Advanced domain patterns

## 🎯 For Different Users

### **New to FLEXT Core?**

1. [**Installation Guide**](getting-started/installation.md) - Setup
2. [**Examples**](examples/overview.md) - Working code samples
3. [**Core API**](api/core.md) - Main patterns

### **Building Applications?**

1. [**Best Practices**](development/best-practices.md) - Development patterns
2. [**Configuration**](configuration/overview.md) - Settings management
3. [**Architecture**](architecture/overview.md) - Design principles

### **Contributing?**

1. Check src/flext_core/ for current implementation
2. Review tests/ for expected behavior
3. Follow existing patterns for consistency

## ⚠️ Documentation Philosophy

**Esta documentação segue a filosofia "REALITY FIRST":**

### ✅ **We Document:**

- Actual working code from src/flext_core/
- Tested examples that compile and run
- Current implementation status
- Real API exports from **init**.py

### ❌ **We Don't Document:**

- Planned features without implementation
- Untested code examples
- Inflated status claims
- Theoretical architectures

## 🔗 Navigation

- **Beginners**: Installation → Examples → Core API
- **Developers**: Best Practices → Patterns API → Architecture
- **Contributors**: Core API → Architecture → Current codebase

---

**All documentation is validated against the current implementation in src/flext_core/**
