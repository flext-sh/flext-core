# Exemplos Práticos - FLEXT Core

**Exemplos baseados no código real de src/flext_core**

## 🎯 Visão Geral

Esta seção apresenta exemplos práticos usando componentes REAIS do FLEXT Core. Todos os exemplos foram validados contra o código atual em src/flext_core/**init**.py e funcionam com a implementação atual.

## 📦 Importações Disponíveis

**Baseado em src/flext_core/**init**.py:**

```python
# Core patterns (FUNCTIONAL)
from flext_core import FlextResult, FlextContainer

# Domain patterns (DISPONÍVEL)
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

# Configuration (FUNCTIONAL)
from flext_core import FlextBaseSettings

# Other exports - check __init__.py for current status
```

## 🔄 Exemplo 1: FlextResult Railway Pattern

**VALIDADO** - Baseado na implementação real:

```python
"""
Exemplo real usando FlextResult - o padrão central do FLEXT Core.
Este exemplo funciona com a implementação atual.
"""

from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if not email:
        return FlextResult.fail("Email é obrigatório")

    if "@" not in email:
        return FlextResult.fail("Email deve conter @")

    return FlextResult.ok(email.lower())

def create_user_id(email: str) -> FlextResult[str]:
    """Create user ID from validated email."""
    user_id = f"user_{hash(email) % 10000:04d}"
    return FlextResult.ok(user_id)

def save_user_data(user_id: str, email: str) -> FlextResult[dict]:
    """Simulate saving user data."""
    user_data = {
        "id": user_id,
        "email": email,
        "created": True
    }
    return FlextResult.ok(user_data)

# Railway-oriented programming pattern
def create_user(email: str) -> FlextResult[dict]:
    """Complete user creation with railway pattern."""
    return (
        validate_email(email)
        .flat_map(lambda validated_email: create_user_id(validated_email)
                  .map(lambda user_id: (user_id, validated_email)))
        .flat_map(lambda data: save_user_data(data[0], data[1]))
    )

# Usage examples
if __name__ == "__main__":
    # Success case
    result = create_user("user@example.com")
    if result.success:
        print(f"✅ User created: {result.data}")
    else:
        print(f"❌ Error: {result.error}")

    # Error case
    error_result = create_user("invalid-email")
    print(f"❌ Expected error: {error_result.error}")
```

## 🏗️ Exemplo 2: FlextContainer Dependency Injection

**VALIDADO** - Baseado na implementação real:

```python
"""
Exemplo real usando FlextContainer - sistema de DI do FLEXT Core.
Este exemplo funciona com a implementação atual.
"""

from flext_core import FlextContainer, FlextResult

# Simple services for DI example
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, data: dict) -> FlextResult[str]:
        """Simulate database save."""
        return FlextResult.ok(f"Saved {data} to {self.connection_string}")

class UserService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create user using injected database service."""
        user_data = {"name": name, "email": email}

        save_result = self.db_service.save(user_data)
        if save_result.is_failure:
            return FlextResult.fail(f"Save failed: {save_result.error}")

        return FlextResult.ok(user_data)

# Container setup and usage
def setup_container() -> FlextContainer:
    """Setup dependency injection container."""
    container = FlextContainer()

    # Register database service
    db_service = DatabaseService("sqlite:///app.db")
    reg_result = container.register("database", db_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register database: {reg_result.error}")

    # Register user service with dependency
    user_service = UserService(db_service)
    reg_result = container.register("user_service", user_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register user service: {reg_result.error}")

    return container

# Usage example
if __name__ == "__main__":
    # Setup container
    container = setup_container()

    # Get service from container
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.data

        # Use service
        create_result = user_service.create_user("João", "joao@test.com")
        if create_result.success:
            print(f"✅ User created: {create_result.data}")
        else:
            print(f"❌ Create failed: {create_result.error}")
    else:
        print(f"❌ Service not found: {service_result.error}")
```

## 🏛️ Exemplo 3: FlextEntity Domain Pattern

**VALIDADO** - Domain entities usando API atual:

```python
"""
Exemplo usando FlextEntity - padrão de domínio do FLEXT Core.
CORRETO - Usando API atual de models.py.
"""

from flext_core.models import FlextEntity
from flext_core import FlextResult
from typing import Optional
from datetime import datetime
from pydantic import Field

class User(FlextEntity):
    """Simple user entity example."""

    # Entity attributes (not in __init__)
    id: str
    name: str
    email: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

    def validate_business_rules(self) -> FlextResult[None]:
        """Required abstract method implementation."""
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Valid email is required")
        if not self.name or len(self.name.strip()) == 0:
            return FlextResult.fail("Name cannot be empty")
        return FlextResult.ok(None)

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult.fail("User is already active")

        self.is_active = True
        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user account."""
        if not self.is_active:
            return FlextResult.fail("User is already inactive")

        self.is_active = False
        return FlextResult.ok(None)

    def login(self) -> FlextResult[None]:
        """Record user login."""
        if not self.is_active:
            return FlextResult.fail("Cannot login - user is inactive")

        self.last_login = datetime.now()
        return FlextResult.ok(None)

# Usage example
if __name__ == "__main__":
    # Create user entity with proper field-based initialization
    user = User(
        id="user_123",
        name="Maria Silva",
        email="maria@test.com"
    )
    print(f"✅ User created: {user.name} (ID: {user.id})")

    # Business operations
    login_result = user.login()
    if login_result.success:
        print(f"✅ Login successful at {user.last_login}")

    # Test business rules
    deactivate_result = user.deactivate()
    if deactivate_result.success:
        print("✅ User deactivated")

    # This should fail
    login_after_deactivate = user.login()
    print(f"❌ Expected failure: {login_after_deactivate.error}")
```

## ⚙️ Exemplo 4: FlextBaseSettings Configuration

**VALIDADO** - Sistema de configuração funcional:

```python
"""
Exemplo usando FlextBaseSettings - sistema de configuração do FLEXT Core.
Baseado na implementação atual disponível.
"""

from flext_core import FlextBaseSettings
from typing import Optional

class AppSettings(FlextBaseSettings):
    """Application configuration using FLEXT Core settings."""

    # Basic settings with defaults
    app_name: str = "FLEXT Demo App"
    debug: bool = False
    port: int = 8000

    # Database settings
    database_url: str = "sqlite:///app.db"
    max_connections: int = 10

    # Optional settings
    redis_url: Optional[str] = None

    class Config:
        env_prefix = "APP_"

# Usage example
if __name__ == "__main__":
    # Load configuration (from env vars or defaults)
    settings = AppSettings()

    print(f"✅ App: {settings.app_name}")
    print(f"✅ Debug: {settings.debug}")
    print(f"✅ Port: {settings.port}")
    print(f"✅ Database: {settings.database_url}")
    print(f"✅ Redis: {settings.redis_url or 'Not configured'}")

    # Environment-aware settings
    if settings.debug:
        print("🔧 Running in debug mode")
    else:
        print("🚀 Running in production mode")
```

## 🧪 Como Executar os Exemplos

### 1. Verificar Dependências

```bash
# Verificar se FLEXT Core está instalado
python -c "from flext_core import FlextResult, FlextContainer; print('✅ Imports working')"
```

### 2. Executar Exemplos

```bash
# Salvar qualquer exemplo como arquivo .py e executar
python exemplo_railway.py
python exemplo_container.py
python exemplo_entity.py
python exemplo_config.py
```

### 3. Testar Modificações

```bash
# Modificar exemplos para suas necessidades
# Todos os exemplos usam apenas a API pública documentada
```

## 🎯 Próximos Passos

1. **[Quickstart](../getting-started/quickstart.md)** - Começar com FLEXT Core
2. **[API Core](../api/core.md)** - Referência completa da API
3. **[Arquitetura](../architecture/overview.md)** - Entender os padrões

## ⚠️ Nota Importante

Estes exemplos são baseados na implementação ATUAL em src/flext_core/. Para exemplos mais elaborados, consulte o código nos testes (tests/) e o diretório examples/ do projeto.

**Status dos Componentes** (baseado no código atual):

- ✅ **FlextResult**: Totalmente funcional
- ✅ **FlextContainer**: Implementado e testado
- 🔧 **FlextEntity**: API disponível, funcionalidade pode estar em desenvolvimento
- 🔧 **FlextBaseSettings**: Baseado em Pydantic, funcional
- 📋 **Patterns avançados**: Consultar código atual para status

---

**Todos os exemplos aqui foram validados contra o código em src/flext_core/**init**.py**
