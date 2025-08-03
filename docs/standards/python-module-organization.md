# Python Module Organization & Semantic Patterns

**The FLEXT Core Module Architecture & Best Practices for the 32-Project Ecosystem**

---

## 🏗️ **Module Architecture Overview**

FLEXT Core implements a **layered module architecture** that supports Clean Architecture, Domain-Driven Design, and the railway-oriented programming paradigm. This structure serves as the template for all 32 projects in the FLEXT ecosystem.

### **Core Design Principles**

1. **Single Source of Truth**: Each pattern has one canonical implementation
2. **Explicit Dependencies**: Clear import paths with minimal coupling
3. **Type-Safe Everything**: Comprehensive type hints and MyPy compliance
4. **Railway-Oriented**: FlextResult[T] threading through all operations
5. **Ecosystem Consistency**: Patterns work identically across 32 projects

---

## 📁 **Module Structure & Responsibilities**

### **Foundation Layer**

```python
# Core foundation - used by everything
src/flext_core/
├── __init__.py              # 🎯 Public API gateway
├── flext_types.py           # 🎯 Type system foundation
├── constants.py             # 🎯 Ecosystem constants
└── version.py               # 🎯 Version management
```

**Responsibility**: Establish the foundational contracts that all other modules depend on.

**Import Pattern**:

```python
# All ecosystem projects start here
from flext_core import FlextResult, FlextContainer
```

### **Core Pattern Layer**

```python
# Railway-oriented programming core
├── result.py                # 🚀 FlextResult[T] - Railway pattern
├── container.py             # 🚀 FlextContainer - DI system
├── exceptions.py            # 🚀 Exception hierarchy
└── utilities.py             # 🚀 Pure utility functions
```

**Responsibility**: Provide the railway-oriented programming foundation and dependency injection.

**Usage Pattern**:

```python
from flext_core.result import FlextResult
from flext_core.container import FlextContainer

def process_data(data: str) -> FlextResult[ProcessedData]:
    return FlextResult.ok(ProcessedData(data))
```

### **Configuration & Infrastructure Layer**

```python
# Configuration and system integration
├── config.py                # ⚙️ FlextBaseSettings
├── loggings.py              # ⚙️ Structured logging
├── payload.py               # ⚙️ Message/Event/Payload
└── interfaces.py            # ⚙️ Protocol definitions
```

**Responsibility**: Handle system configuration, logging, and external integration contracts.

**Configuration Pattern**:

```python
from flext_core.config import FlextBaseSettings

class AppSettings(FlextBaseSettings):
    database_url: str = "sqlite:///app.db"
    debug_mode: bool = False

    class Config:
        env_prefix = "APP_"
```

### **Domain-Driven Design Layer**

```python
# DDD implementation patterns
├── entities.py              # 🏛️ FlextEntity - Domain entities
├── value_objects.py         # 🏛️ FlextValueObject - Value objects
├── aggregate_root.py        # 🏛️ FlextAggregateRoot - Aggregates
└── domain_services.py       # 🏛️ FlextDomainService - Domain services
```

**Responsibility**: Provide rich domain modeling patterns following DDD principles.

**Domain Modeling Pattern**:

```python
from flext_core.entities import FlextEntity
from flext_core.value_objects import FlextValueObject

class User(FlextEntity):
    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        # Business logic with domain events
        return FlextResult.ok(None)

class Email(FlextValueObject):
    address: str

    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError("Invalid email")
```

### **CQRS & Command Layer**

```python
# Command Query Responsibility Segregation
├── commands.py              # 📤 Command pattern implementation
├── handlers.py              # 📤 Handler pattern implementation
└── validation.py            # 📤 Input validation system
```

**Responsibility**: Implement CQRS patterns for enterprise scalability.

**CQRS Pattern**:

```python
from flext_core.commands import FlextCommand, FlextCommandBus
from flext_core.handlers import FlextCommandHandler

class CreateUserCommand(FlextCommand):
    name: str
    email: str

class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    async def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Implementation
        return FlextResult.ok(user)
```

### **Extension & Utility Layer**

```python
# Reusable behaviors and extensions
├── mixins.py                # 🔧 Reusable behavior mixins
├── decorators.py            # 🔧 Enterprise decorator patterns
├── fields.py                # 🔧 Field metadata system
├── guards.py                # 🔧 Validation guards
└── core.py                  # 🔧 FlextCore main class
```

**Responsibility**: Provide reusable patterns and cross-cutting concerns.

**Mixin Pattern**:

```python
from flext_core.mixins import TimestampMixin, SoftDeleteMixin

class User(FlextEntity, TimestampMixin, SoftDeleteMixin):
    name: str
    email: str
    # Automatically gets: created_at, updated_at, deleted_at, is_deleted
```

### **Internal Implementation Layer**

```python
# Private implementation details
└── _*_base.py               # 🔒 Base implementation modules (internal)
```

**Responsibility**: Internal implementation details not exposed in public API.

**Usage**: Never imported directly by ecosystem projects.

---

## 🎯 **Semantic Naming Conventions**

### **Public API Naming (FlextXxx)**

All public exports use the `Flext` prefix to avoid namespace conflicts:

```python
# Core patterns
FlextResult[T]               # Railway-oriented result type
FlextContainer              # Dependency injection container
FlextEntity                 # Domain entity base class
FlextValueObject            # Domain value object base class
FlextAggregateRoot          # DDD aggregate root
FlextCommand                # CQRS command base class
FlextCommandHandler         # CQRS command handler base class
FlextBaseSettings           # Configuration base class

# Utility patterns
FlextPayload                # Message payload wrapper
FlextEvent                  # Domain event base class
FlextMessage                # Messaging pattern base class
FlextDomainService          # Domain service base class
```

**Rationale**: Clear namespace separation prevents conflicts across 32 projects.

### **Module-Level Naming**

```python
# Module names are descriptive and focused
result.py                   # Contains FlextResult and related utilities
container.py                # Contains FlextContainer and DI patterns
entities.py                 # Contains FlextEntity and entity patterns
value_objects.py            # Contains FlextValueObject patterns
commands.py                 # Contains FlextCommand and command patterns
```

**Pattern**: One primary concern per module with related utilities.

### **Internal Naming (\_xxx)**

```python
# Internal modules use underscore prefix
_result_base.py             # Internal result implementation
_container_base.py          # Internal container implementation
_entity_base.py             # Internal entity implementation

# Internal functions and classes
def _validate_type(obj: Any) -> bool:
    """Internal validation function"""

class _InternalHandler:
    """Internal implementation detail"""
```

**Rule**: Anything with `_` prefix is internal and not part of public API.

---

## 📦 **Import Patterns & Best Practices**

### **Recommended Import Styles**

#### **1. Primary Pattern (Recommended for Ecosystem)**

```python
# Import from main package - gets everything needed
from flext_core import FlextResult, FlextContainer, FlextEntity

# Use patterns directly
def process_user(data: dict) -> FlextResult[User]:
    return FlextResult.ok(User(**data))
```

#### **2. Specific Module Pattern (For Advanced Usage)**

```python
# Import from specific modules for clarity
from flext_core.result import FlextResult
from flext_core.container import FlextContainer
from flext_core.entities import FlextEntity

# More explicit but verbose
```

#### **3. Type Annotation Pattern**

```python
# Import types for annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import FlextResult, FlextContainer

# Use in function signatures
def process_data(container: 'FlextContainer') -> 'FlextResult[str]':
    pass
```

### **Anti-Patterns (Forbidden)**

```python
# ❌ Don't import everything
from flext_core import *

# ❌ Don't import internal modules
from flext_core._result_base import _InternalResult

# ❌ Don't use deep imports unnecessarily
from flext_core.result import FlextResult, _private_function

# ❌ Don't alias core types
from flext_core import FlextResult as Result  # Confusing across ecosystem
```

---

## 🏛️ **Architectural Patterns**

### **Layer Separation**

```python
# Each layer has clear boundaries
┌─────────────────────────────────────┐
│        Application Layer            │  # commands.py, handlers.py
│  (CQRS, Commands, Application Logic)│
├─────────────────────────────────────┤
│          Domain Layer               │  # entities.py, value_objects.py
│    (Business Logic, Domain Rules)   │  # aggregate_root.py, domain_services.py
├─────────────────────────────────────┤
│       Infrastructure Layer          │  # config.py, loggings.py, payload.py
│  (External Concerns, Configuration) │
├─────────────────────────────────────┤
│         Foundation Layer            │  # result.py, container.py, types.py
│   (Core Patterns, Type System)      │  # constants.py, exceptions.py
└─────────────────────────────────────┘
```

### **Dependency Direction**

```python
# Dependencies flow inward (Clean Architecture)
Application Layer  →  Domain Layer  →  Foundation Layer
     ↓                     ↓                ↓
Infrastructure Layer → Foundation Layer   (OK)
```

**Rule**: Higher layers can depend on lower layers, never the reverse.

### **Cross-Cutting Concerns**

```python
# Handled via mixins and decorators
from flext_core.mixins import LoggingMixin, ValidationMixin
from flext_core.decorators import with_correlation_id, with_metrics

class UserService(LoggingMixin, ValidationMixin):
    @with_correlation_id
    @with_metrics("user_service.create_user")
    def create_user(self, data: dict) -> FlextResult[User]:
        return self.validate(data).flat_map(self._create_user_impl)
```

---

## 🔄 **Railway-Oriented Programming Patterns**

### **FlextResult Chain Patterns**

```python
# Basic chaining
def process_pipeline(data: dict) -> FlextResult[ProcessedData]:
    return (
        validate_input(data)
        .map(transform_data)
        .flat_map(save_to_database)
        .map(format_response)
    )

# Error aggregation
def validate_user_data(data: dict) -> FlextResult[dict]:
    errors = []

    if not data.get('name'):
        errors.append("Name is required")
    if not data.get('email'):
        errors.append("Email is required")

    return FlextResult.fail(errors) if errors else FlextResult.ok(data)

# Resource management with context
async def process_with_transaction(data: dict) -> FlextResult[User]:
    async with database_transaction() as tx:
        return (
            await validate_user_data(data)
            .flat_map_async(lambda d: save_user(tx, d))
            .map_async(lambda u: send_welcome_email(u))
        )
```

### **Container Integration Patterns**

```python
# Service location with error handling
def get_user_service(container: FlextContainer) -> FlextResult[UserService]:
    return container.get("user_service")

# Service composition
def process_user_workflow(container: FlextContainer, user_data: dict) -> FlextResult[User]:
    return (
        get_user_service(container)
        .flat_map(lambda service: service.validate_user(user_data))
        .flat_map(lambda service: service.create_user(user_data))
        .flat_map(lambda user: notify_user_created(container, user))
    )
```

---

## 🎯 **Domain-Driven Design Patterns**

### **Entity Patterns**

```python
from flext_core import FlextEntity, FlextResult

class User(FlextEntity):
    """Rich domain entity with business logic"""
    name: str
    email: str
    is_active: bool = False
    _domain_events: List[dict] = field(default_factory=list, init=False)

    def activate(self) -> FlextResult[None]:
        """Business operation with domain event"""
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        self.add_domain_event({
            "type": "UserActivated",
            "user_id": self.id,
            "timestamp": datetime.utcnow()
        })
        return FlextResult.ok(None)

    def change_email(self, new_email: str) -> FlextResult[None]:
        """Email change with validation"""
        if not "@" in new_email:
            return FlextResult.fail("Invalid email format")

        old_email = self.email
        self.email = new_email
        self.add_domain_event({
            "type": "EmailChanged",
            "user_id": self.id,
            "old_email": old_email,
            "new_email": new_email
        })
        return FlextResult.ok(None)
```

### **Value Object Patterns**

```python
from flext_core import FlextValueObject

class Email(FlextValueObject):
    """Immutable value object with validation"""
    address: str

    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError("Invalid email format")
        if len(self.address) > 254:
            raise ValueError("Email too long")

    @property
    def domain(self) -> str:
        return self.address.split("@")[1]

    @property
    def local_part(self) -> str:
        return self.address.split("@")[0]

class Money(FlextValueObject):
    """Money with currency and precision"""
    amount: Decimal
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if self.currency not in ["USD", "EUR", "BRL"]:
            raise ValueError("Unsupported currency")

    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)
```

### **Aggregate Root Patterns**

```python
from flext_core import FlextAggregateRoot, FlextResult

class Order(FlextAggregateRoot):
    """Aggregate root managing order lifecycle"""
    customer_id: str
    items: List[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.DRAFT
    total: Money = Money(Decimal('0'), 'USD')

    def add_item(self, product_id: str, quantity: int, price: Money) -> FlextResult[None]:
        """Add item with business rules"""
        if self.status != OrderStatus.DRAFT:
            return FlextResult.fail("Cannot modify confirmed order")

        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        item = OrderItem(product_id, quantity, price)
        self.items.append(item)
        self.recalculate_total()

        self.add_domain_event({
            "type": "ItemAdded",
            "order_id": self.id,
            "product_id": product_id,
            "quantity": quantity
        })
        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirm order with validation"""
        if not self.items:
            return FlextResult.fail("Cannot confirm empty order")

        if self.status != OrderStatus.DRAFT:
            return FlextResult.fail("Order already confirmed")

        self.status = OrderStatus.CONFIRMED
        self.add_domain_event({
            "type": "OrderConfirmed",
            "order_id": self.id,
            "total": str(self.total.amount),
            "currency": self.total.currency
        })
        return FlextResult.ok(None)
```

---

## 🔧 **Configuration Patterns**

### **Environment-Aware Configuration**

```python
from flext_core import FlextBaseSettings

class DatabaseSettings(FlextBaseSettings):
    """Database configuration with environment variables"""
    host: str = "localhost"
    port: int = 5432
    username: str = "app_user"
    password: str = field(default="", repr=False)  # Hidden in logs
    database: str = "app_db"

    class Config:
        env_prefix = "DB_"
        env_file = ".env"

    @property
    def connection_url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class AppSettings(FlextBaseSettings):
    """Application configuration composition"""
    app_name: str = "FLEXT Application"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseSettings = field(default_factory=DatabaseSettings)

    class Config:
        env_prefix = "APP_"
        env_nested_delimiter = "__"  # APP_DATABASE__HOST

# Usage in ecosystem projects
settings = AppSettings()
logger.info("Starting application", app_name=settings.app_name)
```

### **Hierarchical Configuration**

```python
from flext_core import FlextBaseSettings

class FeatureFlags(FlextBaseSettings):
    """Feature flag configuration"""
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False

    class Config:
        env_prefix = "FEATURE_"

class SecuritySettings(FlextBaseSettings):
    """Security configuration"""
    secret_key: str = field(repr=False)
    token_expiry: int = 3600
    allowed_hosts: List[str] = field(default_factory=list)

    class Config:
        env_prefix = "SECURITY_"

class CompleteSettings(FlextBaseSettings):
    """Complete application settings"""
    features: FeatureFlags = field(default_factory=FeatureFlags)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)

    class Config:
        env_nested_delimiter = "__"

# Environment variables:
# FEATURE_ENABLE_CACHING=true
# SECURITY_SECRET_KEY=supersecret
# DATABASE_HOST=postgres.example.com
```

---

## 🚀 **Performance & Optimization Patterns**

### **Lazy Loading Patterns**

```python
from functools import cached_property

class ExpensiveService(FlextService):
    """Service with expensive initialization"""

    @cached_property
    def database_connection(self) -> FlextResult[DatabaseConnection]:
        """Lazy database connection"""
        return self._connect_to_database()

    @cached_property
    def cache_client(self) -> FlextResult[CacheClient]:
        """Lazy cache client"""
        return self._connect_to_cache()

    def process_data(self, data: dict) -> FlextResult[ProcessedData]:
        """Use lazy-loaded resources"""
        return (
            self.database_connection
            .flat_map(lambda db: self.cache_client.map(lambda cache: (db, cache)))
            .flat_map(lambda resources: self._process_with_resources(data, *resources))
        )
```

### **Batch Processing Patterns**

```python
from typing import List, Iterator

def process_batch(items: List[dict], batch_size: int = 100) -> FlextResult[List[ProcessedItem]]:
    """Process items in batches for memory efficiency"""
    results = []

    for batch in _chunk_items(items, batch_size):
        batch_result = _process_batch_items(batch)
        if batch_result.is_failure:
            return batch_result
        results.extend(batch_result.data)

    return FlextResult.ok(results)

def _chunk_items(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Split items into chunks"""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]
```

### **Caching Patterns**

```python
from functools import wraps
from typing import Dict, Optional

class FlextCache:
    """Simple caching service"""
    _cache: Dict[str, Any] = {}

    def get(self, key: str) -> FlextResult[Optional[Any]]:
        value = self._cache.get(key)
        return FlextResult.ok(value)

    def set(self, key: str, value: Any, ttl: int = 3600) -> FlextResult[None]:
        self._cache[key] = value
        return FlextResult.ok(None)

def with_cache(cache_key_func: Callable[[Any], str], ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> FlextResult[Any]:
            cache_key = cache_key_func(*args, **kwargs)

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result.is_success and cached_result.data is not None:
                return FlextResult.ok(cached_result.data)

            # Execute function
            result = func(*args, **kwargs)
            if result.is_success:
                cache.set(cache_key, result.data, ttl)

            return result
        return wrapper
    return decorator

# Usage
@with_cache(lambda user_id: f"user:{user_id}", ttl=1800)
def get_user(user_id: str) -> FlextResult[User]:
    return database.find_user(user_id)
```

---

## 🧪 **Testing Patterns**

### **Test Organization**

```python
# Test structure mirrors source structure
tests/
├── unit/                    # Unit tests (isolated)
│   ├── test_result.py       # Tests for result.py
│   ├── test_container.py    # Tests for container.py
│   ├── test_entities.py     # Tests for entities.py
│   └── domain/              # Domain-specific tests
├── integration/             # Integration tests
│   ├── test_database.py     # Database integration
│   ├── test_config.py       # Configuration integration
│   └── test_services.py     # Service integration
├── e2e/                     # End-to-end tests
│   └── test_workflows.py    # Complete workflows
├── conftest.py              # Test configuration
└── shared_test_domain.py    # Shared test domain models
```

### **FlextResult Testing Patterns**

```python
import pytest
from flext_core import FlextResult

def test_result_success_path():
    """Test successful operation"""
    result = divide(10, 2)

    assert result.is_success
    assert result.data == 5.0
    assert result.error is None

def test_result_failure_path():
    """Test failure operation"""
    result = divide(10, 0)

    assert result.is_failure
    assert result.data is None
    assert result.error == "Division by zero"

def test_result_chaining():
    """Test railway-oriented chaining"""
    result = (
        FlextResult.ok(10)
        .map(lambda x: x * 2)
        .flat_map(lambda x: divide(x, 4))
    )

    assert result.is_success
    assert result.data == 5.0

def test_result_failure_propagation():
    """Test failure propagation in chains"""
    result = (
        FlextResult.fail("Initial error")
        .map(lambda x: x * 2)  # Should not execute
        .flat_map(lambda x: divide(x, 4))  # Should not execute
    )

    assert result.is_failure
    assert result.error == "Initial error"
```

### **Domain Testing Patterns**

```python
import pytest
from flext_core import FlextEntity, FlextResult

class TestUser:
    """Test domain entity behavior"""

    def test_user_activation_success(self):
        """Test successful user activation"""
        user = User(name="John", email="john@example.com")

        result = user.activate()

        assert result.is_success
        assert user.is_active
        assert len(user.domain_events) == 1
        assert user.domain_events[0]["type"] == "UserActivated"

    def test_user_activation_already_active(self):
        """Test activation of already active user"""
        user = User(name="John", email="john@example.com", is_active=True)

        result = user.activate()

        assert result.is_failure
        assert result.error == "User already active"
        assert len(user.domain_events) == 0

    def test_email_change_validation(self):
        """Test email change with validation"""
        user = User(name="John", email="john@example.com")

        # Valid email
        result = user.change_email("newemail@example.com")
        assert result.is_success
        assert user.email == "newemail@example.com"

        # Invalid email
        result = user.change_email("invalid-email")
        assert result.is_failure
        assert result.error == "Invalid email format"
```

### **Container Testing Patterns**

```python
import pytest
from flext_core import FlextContainer

@pytest.fixture
def clean_container():
    """Provide clean container for each test"""
    return FlextContainer()

def test_service_registration(clean_container):
    """Test service registration"""
    service = UserService()

    result = clean_container.register("user_service", service)

    assert result.is_success

    retrieved = clean_container.get("user_service")
    assert retrieved.is_success
    assert retrieved.data is service

def test_service_not_found(clean_container):
    """Test missing service retrieval"""
    result = clean_container.get("nonexistent_service")

    assert result.is_failure
    assert "not found" in result.error.lower()
```

---

## 📏 **Code Quality Standards**

### **Type Annotation Requirements**

```python
# ✅ Complete type annotations
def process_user(data: dict[str, Any]) -> FlextResult[User]:
    """Process user data with complete type safety"""
    return validate_user_data(data).flat_map(create_user)

# ✅ Generic type usage
T = TypeVar('T')
U = TypeVar('U')

def map_result(result: FlextResult[T], func: Callable[[T], U]) -> FlextResult[U]:
    """Generic result mapping with type safety"""
    if result.is_success:
        return FlextResult.ok(func(result.data))
    return FlextResult.fail(result.error)

# ❌ Missing type annotations
def process_user(data):  # Missing types
    return validate_user_data(data)
```

### **Error Handling Standards**

```python
# ✅ Always use FlextResult for error handling
def divide(a: int, b: int) -> FlextResult[float]:
    if b == 0:
        return FlextResult.fail("Division by zero")
    return FlextResult.ok(a / b)

# ✅ Chain operations safely
def complex_calculation(x: int, y: int) -> FlextResult[str]:
    return (
        divide(x, y)
        .map(lambda result: result * 2)
        .flat_map(lambda result: format_number(result))
    )

# ❌ Never raise exceptions in business logic
def divide_bad(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Division by zero")  # Breaks railway pattern
    return a / b
```

### **Documentation Standards**

```python
def process_user_data(
    data: dict[str, Any],
    validator: UserValidator,
    repository: UserRepository
) -> FlextResult[User]:
    """
    Process user data through validation and persistence.

    This function implements the complete user creation workflow including
    validation, domain object creation, and persistence. It follows the
    railway-oriented programming pattern for consistent error handling.

    Args:
        data: Raw user data from external source
        validator: User validation service for business rule checking
        repository: User persistence service for database operations

    Returns:
        FlextResult[User]: Success contains created user, failure contains
        detailed error message explaining validation or persistence failure

    Example:
        >>> data = {"name": "John", "email": "john@example.com"}
        >>> result = process_user_data(data, validator, repository)
        >>> if result.is_success:
        ...     print(f"Created user: {result.data.name}")
        ... else:
        ...     print(f"Error: {result.error}")
    """
    return (
        validator.validate(data)
        .flat_map(lambda valid_data: User.create(valid_data))
        .flat_map(lambda user: repository.save(user))
    )
```

---

## 🌐 **Ecosystem Integration Guidelines**

### **Cross-Project Import Standards**

```python
# ✅ Standard ecosystem imports
from flext_core import FlextResult, FlextContainer, FlextEntity
from flext_db_oracle import OracleConnection, OracleRepository
from flext_ldap import LdapConnection, LdapUser

# ✅ Consistent error handling across projects
def sync_user_from_ldap(ldap_conn: LdapConnection, user_id: str) -> FlextResult[User]:
    return (
        ldap_conn.find_user(user_id)  # Returns FlextResult[LdapUser]
        .map(lambda ldap_user: User.from_ldap(ldap_user))
        .flat_map(lambda user: save_user_to_oracle(user))
    )

# ❌ Don't create custom result types per project
class OracleResult[T]:  # Creates ecosystem fragmentation
    pass
```

### **Configuration Integration**

```python
# ✅ Extend FlextBaseSettings in all projects
class OracleSettings(FlextBaseSettings):
    """Oracle-specific configuration extending core patterns"""
    host: str = "localhost"
    port: int = 1521
    service_name: str = "XEPDB1"

    class Config:
        env_prefix = "ORACLE_"

class ProjectSettings(FlextBaseSettings):
    """Project configuration composing ecosystem settings"""
    oracle: OracleSettings = field(default_factory=OracleSettings)
    ldap: LdapSettings = field(default_factory=LdapSettings)
    core: FlextCoreSettings = field(default_factory=FlextCoreSettings)
```

### **Domain Model Consistency**

```python
# ✅ Use FlextEntity consistently across ecosystem
class OracleUser(FlextEntity):
    """Oracle user entity following core patterns"""
    oracle_id: str
    username: str
    email: str

    def sync_to_ldap(self, ldap_service: LdapService) -> FlextResult[None]:
        """Cross-system synchronization using core patterns"""
        return ldap_service.update_user(self.to_ldap_format())

class LdapUser(FlextEntity):
    """LDAP user entity following core patterns"""
    dn: str
    uid: str
    mail: str

    def to_oracle_format(self) -> dict[str, Any]:
        """Transform to Oracle-compatible format"""
        return {
            "username": self.uid,
            "email": self.mail,
            "external_id": self.dn
        }
```

---

## 🔄 **Migration & Versioning Patterns**

### **Semantic Versioning for Ecosystem**

```python
# Version format: MAJOR.MINOR.PATCH
# - MAJOR: Breaking changes affecting all 32 projects
# - MINOR: New features maintaining backward compatibility
# - PATCH: Bug fixes and improvements

# Current: 0.9.0 (Beta)
# Target: 1.0.0 (Production Ready, December 2025)

# Version compatibility matrix
COMPATIBLE_VERSIONS = {
    "1.0.0": ["1.0.x", "1.1.x", "1.2.x"],  # Major version compatibility
    "2.0.0": ["2.0.x", "2.1.x", "2.2.x"],  # Breaking changes in 2.0
}
```

### **Backward Compatibility Patterns**

```python
# ✅ Deprecation pattern for ecosystem changes
from warnings import warn
from typing import Optional

def old_function(data: dict) -> FlextResult[User]:
    """
    DEPRECATED: Use new_function instead.

    This function will be removed in version 2.0.0.
    Use new_function for better performance and type safety.
    """
    warn(
        "old_function is deprecated and will be removed in version 2.0.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(data, legacy_mode=True)

def new_function(data: dict, legacy_mode: bool = False) -> FlextResult[User]:
    """New implementation with backward compatibility"""
    if legacy_mode:
        # Handle old behavior
        pass
    # New behavior
    pass
```

### **Migration Utilities**

```python
class FlextMigration:
    """Base class for ecosystem migrations"""

    version_from: str
    version_to: str

    def migrate(self, project_config: dict) -> FlextResult[dict]:
        """Migrate project configuration between versions"""
        pass

    def rollback(self, project_config: dict) -> FlextResult[dict]:
        """Rollback migration if needed"""
        pass

class Migration_0_9_to_1_0(FlextMigration):
    """Migration from 0.9.x to 1.0.0"""

    version_from = "0.9.x"
    version_to = "1.0.0"

    def migrate(self, config: dict) -> FlextResult[dict]:
        """Update configuration for 1.0.0 patterns"""
        # Handle specific migration logic
        return FlextResult.ok(config)
```

---

## 📋 **Checklist for New Modules**

### **Module Creation Checklist**

- [ ] **Naming**: Uses clear, descriptive name following conventions
- [ ] **Location**: Placed in appropriate architectural layer
- [ ] **Imports**: Only imports from same or lower layers
- [ ] **Types**: Complete type annotations with MyPy compliance
- [ ] **Error Handling**: Uses FlextResult for all error conditions
- [ ] **Documentation**: Comprehensive docstrings with examples
- [ ] **Tests**: 95% coverage with unit and integration tests
- [ ] **Exports**: Added to `__init__.py` if public API
- [ ] **Examples**: Working examples in appropriate example files
- [ ] **Ecosystem Impact**: Validated across dependent projects

### **Quality Gate Checklist**

- [ ] **Linting**: `make lint` passes (Ruff with all rules)
- [ ] **Type Check**: `make type-check` passes (strict MyPy)
- [ ] **Tests**: `make test` passes (95% coverage minimum)
- [ ] **Security**: `make security` passes (Bandit + pip-audit)
- [ ] **Format**: `make format` passes (79 character line limit)
- [ ] **Integration**: Works with existing ecosystem projects
- [ ] **Documentation**: Updated relevant documentation files
- [ ] **Examples**: Added or updated working examples

---

**Last Updated**: August 2, 2025  
**Target Audience**: FLEXT ecosystem developers and contributors  
**Scope**: Python module organization for 32-project ecosystem  
**Version**: 0.9.0 → 1.0.0 development guidelines
