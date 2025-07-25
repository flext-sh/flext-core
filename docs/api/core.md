# API Core - FLEXT Core

**Referência completa das APIs fundamentais do FLEXT Core**

## 🎯 Visão Geral

A API Core do FLEXT fornece os componentes fundamentais necessários para construir aplicações empresariais robustas. Todos os componentes são type-safe, testáveis e seguem padrões de arquitetura limpa.

## 📦 Imports Principais

```python
# Imports modernos recomendados
from flext_core import (
    FlextResult,           # Error handling type-safe
    FlextContainer,        # Dependency injection
    FlextCoreSettings,     # Configuration management
    FlextEntity,           # Domain entities
    FlextValueObject,      # Immutable value objects
    FlextAggregateRoot,    # Domain aggregates
)

# Imports de compatibilidade (legado)
from flext_core import (
    FlextResult,         # ⚠️ Deprecated: use FlextResult
    DIContainer,          # ⚠️ Deprecated: use FlextContainer
)
```

## 🎭 FlextResult[T] - Error Handling Type-Safe

**Substitui exceções por resultados explícitos e type-safe.**

### Criação de Resultados

```python
from flext_core import FlextResult

# Resultado de sucesso
def fetch_user(user_id: str) -> FlextResult[dict]:
    user_data = {"id": user_id, "name": "João"}
    return FlextResult.ok(user_data)

# Resultado de erro
def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult.fail("Email deve conter @")
    return FlextResult.ok(email)

# Factory methods
success_result = FlextResult.ok("success data")
error_result = FlextResult.fail("error message")
```

### Verificação de Status

```python
result = fetch_user("123")

# Verificação de sucesso/erro
if result.is_success:
    print(f"Dados: {result.data}")
else:
    print(f"Erro: {result.error}")

# Propriedades booleanas
assert result.is_success == True
assert result.is_failure == False
```

### Tratamento de Dados

```python
# Acesso seguro aos dados
result = fetch_user("123")

if result.is_success:
    # result.data é garantidamente não-None quando is_success=True
    user_data = result.datadict
    print(f"User: {user_data['name']}")

if result.is_failure:
    # result.error é garantidamente não-None quando is_failure=True
    error_msg = result.errorstr
    print(f"Error: {error_msg}")
```

### Composição e Encadeamento

```python
def create_and_save_user(name: str, email: str) -> FlextResult[str]:
    # Validação
    email_result = validate_email(email)
    if email_result.is_failure:
        return email_result  # Propaga o erro
    
    # Criação
    user_result = create_user(name, email_result.data)
    if user_result.is_failure:
        return FlextResult.fail(f"Falha na criação: {user_result.error}")
    
    # Salvamento
    save_result = save_user(user_result.data)
    if save_result.is_failure:
        return FlextResult.fail(f"Falha no salvamento: {save_result.error}")
    
    return FlextResult.ok("Usuário criado com sucesso")
```

### API Completa

```python
class FlextResult[T]:
    """Type-safe result container."""
    
    # Factory methods
    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create success result."""
    
    @classmethod  
    def fail(cls, error: str) -> FlextResult[T]:
        """Create failure result."""
    
    # Properties
    @property
    def is_success(self) -> bool:
        """True if result represents success."""
    
    @property
    def is_failure(self) -> bool:
        """True if result represents failure."""
    
    @property
    def data(self) -> T | None:
        """Success data (None if failure)."""
    
    @property
    def error(self) -> str | None:
        """Error message (None if success)."""
    
    # Methods
    def get_data_or(self, default: T) -> T:
        """Get data or return default if failure."""
    
    def get_error_or(self, default: str) -> str:
        """Get error or return default if success."""
```

## 🏗️ FlextContainer - Dependency Injection

**Container de injeção de dependência type-safe e enterprise-grade.**

### Registro de Serviços

```python
from flext_core import FlextContainer

# Criação do container
container = FlextContainer()

# Registro básico
result = container.register("database", DatabaseService())
if result.is_success:
    print("Serviço registrado com sucesso")

# Registro com factory
def create_email_service() -> EmailService:
    return EmailService(smtp_host="localhost", port=587)

container.register_factory("email", create_email_service)

# Registro com singleton (padrão)
container.register("cache", RedisCache(), singleton=True)
```

### Resolução de Dependências

```python
# Obter serviço registrado
db_result = container.get("database")
if db_result.is_success:
    database = db_result.dataDatabaseService
    users = database.fetch_users()

# Resolução com type hint
email_service = container.get_typed("email", EmailService)
if email_service.is_success:
    service = email_service.dataEmailService
    service.send_email("test@example.com", "Hello")
```

### Injeção Automática

```python
# Classe com dependências
class UserService:
    def __init__(self, database: DatabaseService, email: EmailService):
        self.database = database
        self.email = email
    
    def create_user(self, name: str, email_addr: str) -> FlextResult[str]:
        # Lógica usando as dependências
        user = {"name": name, "email": email_addr}
        save_result = self.database.save_user(user)
        
        if save_result.is_success:
            self.email.send_welcome_email(email_addr)
            return FlextResult.ok("Usuário criado")
        else:
            return FlextResult.fail("Falha ao salvar usuário")

# Registro com auto-wiring
container.register("user_service", UserService)
# Container automaticamente resolve DatabaseService e EmailService
```

### Configuração e Lifecycle

```python
# Configuração do container
container.configure(
    auto_wire=True,           # Auto-resolve dependencies
    strict_mode=True,         # Fail on missing dependencies  
    lazy_loading=False,       # Eager instantiation
    cache_instances=True      # Cache resolved instances
)

# Lifecycle management
result = container.start()   # Initialize all services
if result.is_success:
    print("Container iniciado")

# Cleanup
container.stop()            # Cleanup resources
container.clear()           # Remove all registrations
```

### API Completa

```python
class FlextContainer:
    """Dependency injection container."""
    
    def register(
        self, 
        key: str, 
        instance: Any, 
        singleton: bool = True
    ) -> FlextResult[None]:
        """Register service instance."""
    
    def register_factory(
        self,
        key: str,
        factory: Callable[[], Any]
    ) -> FlextResult[None]:
        """Register service factory."""
    
    def get(self, key: str) -> FlextResult[Any]:
        """Resolve service by key."""
    
    def get_typed(self, key: str, expected_type: type[T]) -> FlextResult[T]:
        """Resolve service with type checking."""
    
    def has(self, key: str) -> bool:
        """Check if service is registered."""
    
    def remove(self, key: str) -> FlextResult[None]:
        """Remove service registration."""
    
    def get_all_keys(self) -> list[str]:
        """Get all registered service keys."""
    
    def configure(self, **options) -> None:
        """Configure container behavior."""
    
    def start(self) -> FlextResult[None]:
        """Initialize container and services."""
    
    def stop(self) -> FlextResult[None]:
        """Stop container and cleanup."""
    
    def clear(self) -> None:
        """Clear all registrations."""
```

## ⚙️ FlextCoreSettings - Configuration Management

**Gerenciamento centralizado de configurações com Pydantic.**

### Configuração Básica

```python
from flext_core import FlextCoreSettings

# Configuração padrão
settings = FlextCoreSettings()

# Configuração customizada
settings = FlextCoreSettings(
    debug=True,
    environment="development",
    log_level="DEBUG",
    max_connections=100,
    cache_ttl=3600,
    database_url="postgresql://localhost/flext"
)
```

### Carregamento de Variáveis de Ambiente

```python
# Carrega automaticamente de variáveis de ambiente
# FLEXT_DEBUG=true
# FLEXT_LOG_LEVEL=INFO
# FLEXT_MAX_CONNECTIONS=50

settings = FlextCoreSettings()  # Auto-load from env
print(settings.debug)           # True
print(settings.log_level)       # "INFO"
print(settings.max_connections) # 50
```

### Validação de Configuração

```python
# Validação automática via Pydantic
try:
    settings = FlextCoreSettings(
        max_connections=-1,  # Invalid: must be positive
        log_level="INVALID"  # Invalid: not in allowed values
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Configurações Disponíveis

```python
class FlextCoreSettings(BaseSettings):
    """Core configuration settings."""
    
    # Environment
    debug: bool = False
    environment: str = "production"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Performance
    max_connections: int = 10
    cache_ttl: int = 3600
    max_retry_attempts: int = 3
    
    # Database
    database_url: str | None = None
    connection_timeout: int = 30
    
    # Security  
    secret_key: str | None = None
    token_expiry: int = 86400
    
    class Config:
        env_prefix = "FLEXT_"
        case_sensitive = False
```

## 🏛️ Domain Layer - Entidades e Value Objects

### FlextEntity[TId] - Domain Entities

```python
from flext_core import FlextEntity

# ID types
UserId = NewType('UserId', str)

class User(FlextEntity[UserId]):
    """Domain entity with identity."""
    
    def __init__(self, user_id: UserId, name: str, email: str):
        super().__init__(user_id)
        self._name = name
        self._email = email
        self._created_at = datetime.now()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property  
    def email(self) -> str:
        return self._email
    
    def change_name(self, new_name: str) -> FlextResult[None]:
        """Business logic for name change."""
        if not new_name.strip():
            return FlextResult.fail("Nome não pode ser vazio")
        
        self._name = new_name
        return FlextResult.ok(None)
    
    def __str__(self) -> str:
        return f"User(id={self.id}, name={self.name})"

# Usage
user_id = UserId("user_123")
user = User(user_id, "João Silva", "joao@example.com")

# Identity comparison
other_user = User(user_id, "João Santos", "joao2@example.com")
assert user == other_user  # Same ID = same entity
```

### FlextValueObject - Immutable Values

```python
from flext_core import FlextValueObject

class Email(FlextValueObject):
    """Immutable email value object."""
    
    def __init__(self, value: str):
        if not self._is_valid_email(value):
            raise ValueError(f"Invalid email: {value}")
        self._value = value.lower()
    
    @property
    def value(self) -> str:
        return self._value
    
    @property
    def domain(self) -> str:
        return self._value.split("@")[1]
    
    def _is_valid_email(self, email: str) -> bool:
        return "@" in email and "." in email.split("@")[1]
    
    def __str__(self) -> str:
        return self._value

class Money(FlextValueObject):
    """Money value object with currency."""
    
    def __init__(self, amount: float, currency: str = "BRL"):
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        self._amount = amount
        self._currency = currency
    
    @property
    def amount(self) -> float:
        return self._amount
    
    @property
    def currency(self) -> str:
        return self._currency
    
    def add(self, other: 'Money') -> 'Money':
        if self._currency != other._currency:
            raise ValueError("Cannot add different currencies")
        return Money(self._amount + other._amount, self._currency)
    
    def __str__(self) -> str:
        return f"{self._amount:.2f} {self._currency}"

# Usage
email = Email("joao@EXAMPLE.com")  # Normalized to lowercase
print(email.value)     # "joao@example.com"
print(email.domain)    # "example.com"

money1 = Money(100.0, "BRL")
money2 = Money(50.0, "BRL")
total = money1.add(money2)  # Money(150.0, "BRL")
```

### FlextAggregateRoot - Domain Aggregates

```python
from flext_core import FlextAggregateRoot

class Order(FlextAggregateRoot[OrderId]):
    """Order aggregate root."""
    
    def __init__(self, order_id: OrderId, customer_id: CustomerId):
        super().__init__(order_id)
        self._customer_id = customer_id
        self._items: list[OrderItem] = []
        self._status = OrderStatus.PENDING
        self._total = Money(0.0)
    
    def add_item(self, product_id: ProductId, quantity: int, price: Money) -> FlextResult[None]:
        """Add item to order with business rules."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")
        
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        item = OrderItem(product_id, quantity, price)
        self._items.append(item)
        self._recalculate_total()
        
        # Domain event
        self._add_domain_event(ItemAddedEvent(self.id, product_id, quantity))
        
        return FlextResult.ok(None)
    
    def confirm(self) -> FlextResult[None]:
        """Confirm order with business invariants."""
        if not self._items:
            return FlextResult.fail("Cannot confirm empty order")
        
        if self._total.amount == 0:
            return FlextResult.fail("Order total cannot be zero")
        
        self._status = OrderStatus.CONFIRMED
        
        # Domain event
        self._add_domain_event(OrderConfirmedEvent(self.id, self._total))
        
        return FlextResult.ok(None)
    
    def _recalculate_total(self) -> None:
        """Private method to maintain invariants."""
        total_amount = sum(item.total.amount for item in self._items)
        self._total = Money(total_amount)

# Usage
order = Order(OrderId("order_123"), CustomerId("customer_456"))

# Business operations
result = order.add_item(ProductId("prod_1"), 2, Money(50.0))
if result.is_success:
    print("Item added successfully")

confirm_result = order.confirm()
if confirm_result.is_success:
    print("Order confirmed")
```

## 📝 Best Practices

### 1. Error Handling

```python
# ✅ Always use FlextResult for operations that can fail
def save_user(user: User) -> FlextResult[None]:
    try:
        # Save operation
        return FlextResult.ok(None)
    except DatabaseError as e:
        return FlextResult.fail(f"Database error: {e}")

# ✅ Check results before using data
result = fetch_user("123")
if result.is_success:
    process_user(result.data)  # Safe to use
else:
    log_error(result.error)
```

### 2. Dependency Injection

```python
# ✅ Register dependencies at startup
def setup_container() -> FlextContainer:
    container = FlextContainer()
    
    # Infrastructure
    container.register("database", DatabaseService())
    container.register("cache", RedisCache())
    
    # Application services
    container.register("user_service", UserService)
    
    return container

# ✅ Resolve dependencies when needed
container = setup_container()
user_service = container.get("user_service").data
```

### 3. Configuration

```python
# ✅ Use environment-specific settings
settings = FlextCoreSettings()

if settings.debug:
    logging.getLogger().setLevel(logging.DEBUG)

# ✅ Validate configuration early
try:
    settings = FlextCoreSettings()
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    exit(1)
```

### 4. Domain Modeling

```python
# ✅ Use value objects for complex values
class UserEmail(FlextValueObject):
    def __init__(self, value: str):
        # Validation in constructor
        pass

# ✅ Keep business logic in entities
class User(FlextEntity[UserId]):
    def change_email(self, new_email: UserEmail) -> FlextResult[None]:
        # Business rules here
        pass

# ✅ Use aggregates for consistency boundaries
class Order(FlextAggregateRoot[OrderId]):
    def add_item(self, item: OrderItem) -> FlextResult[None]:
        # Maintain order invariants
        pass
```

## 🔗 Compatibilidade e Migração

### Migração de FlextResult para FlextResult

```python
# Old (deprecated)
from flext_core import FlextResult

def old_function() -> FlextResult[str]:
    return FlextResult.success("data")

# New (recommended) 
from flext_core import FlextResult

def new_function() -> FlextResult[str]:
    return FlextResult.ok("data")
```

### Migração de DIContainer para FlextContainer

```python
# Old (deprecated)
from flext_core import DIContainer

container = DIContainer()
container.set("service", service)
service = container.get("service")

# New (recommended)
from flext_core import FlextContainer

container = FlextContainer()
container.register("service", service)
service_result = container.get("service")
if service_result.is_success:
    service = service_result.data
```

---

Esta API Core fornece todos os componentes fundamentais necessários para construir aplicações empresariais robustas e type-safe com FLEXT Core.
