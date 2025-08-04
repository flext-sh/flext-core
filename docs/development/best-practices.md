# Boas Práticas - FLEXT Core

**Guia abrangente de boas práticas para desenvolvimento empresarial com FLEXT Core**

## 🎯 Princípios Fundamentais

### 1. Type Safety First

**SEMPRE use type hints em toda função e método.**

```python
# ✅ Excelente - Type hints completos
def process_user_data(
    user_id: str,
    data: dict[str, object],
    validate: bool = True
) -> FlextResult[User]:
    """Process user data with type safety."""
    pass

# ❌ Evite - Sem type hints
def process_user_data(user_id, data, validate=True):
    pass
```

### 2. Explicit Error Handling

**Use FlextResult em vez de exceções para fluxo de controle.**

```python
# ✅ Excelente - Error handling explícito
def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Divide numbers with explicit error handling."""
    if b == 0:
        return FlextResult.fail("Division by zero not allowed")

    return FlextResult.ok(a / b)

# Usage
result = divide_numbers(10, 2)
if result.success:
    print(f"Result: {result.data}")
else:
    print(f"Error: {result.error}")


# ❌ Evite - Exceções para fluxo de controle
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")  # Não use para controle
    return a / b
```

### 3. Immutability Where Possible

**Prefira objetos imutáveis para value objects.**

```python
# ✅ Excelente - Value object imutável
from flext_core import FlextValueObject

class Money(FlextValueObject):
    def __init__(self, amount: float, currency: str):
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
        """Return new instance instead of modifying."""
        if other.currency != self.currency:
            raise ValueError("Cannot add different currencies")

        return Money(self.amount + other.amount, self.currency)


# ❌ Evite - Mutabilidade desnecessária
class Money:
    def __init__(self, amount: float, currency: str):
        self.amount = amount  # Mutável
        self.currency = currency  # Mutável

    def add(self, other):
        self.amount += other.amount  # Modifica estado
```

## 🏗️ Arquitetura e Design

### 1. Separação de Responsabilidades

**Organize código seguindo Clean Architecture.**

```python
# ✅ Excelente - Separação clara de camadas

# DOMAIN LAYER - Regras de negócio puras
class User(FlextEntity[str]):
    def __init__(self, user_id: str, email: str, name: str):
        super().__init__(user_id)
        self._email = email
        self._name = name
        self._is_active = True

    def deactivate(self) -> FlextResult[None]:
        """Business rule: only active users can be deactivated."""
        if not self._is_active:
            return FlextResult.fail("User is already inactive")

        self._is_active = False
        return FlextResult.ok(None)

# APPLICATION LAYER - Orquestração
class DeactivateUserCommand(FlextCommand):
    def __init__(self, user_id: str, reason: str):
        super().__init__()
        self.user_id = user_id
        self.reason = reason

    def validate(self) -> FlextResult[None]:
        if not self.user_id:
            return FlextResult.fail("User ID is required")

        if not self.reason:
            return FlextResult.fail("Reason is required")

        return FlextResult.ok(None)

class DeactivateUserHandler(FlextCommandHandler[DeactivateUserCommand, None]):
    def __init__(self, user_repository: UserRepository):
        super().__init__()
        self._user_repository = user_repository

    def handle(self, command: DeactivateUserCommand) -> FlextResult[None]:
        # Get user
        user_result = self._user_repository.find_by_id(command.user_id)
        if user_result.is_failure:
            return FlextResult.fail(f"User not found: {command.user_id}")

        user = user_result.data

        # Apply business rule
        deactivate_result = user.deactivate()
        if deactivate_result.is_failure:
            return deactivate_result

        # Persist
        save_result = self._user_repository.save(user)
        return save_result

# INFRASTRUCTURE LAYER - Implementações técnicas
class PostgreSQLUserRepository(UserRepository):
    def __init__(self, connection: Connection):
        self._connection = connection

    def find_by_id(self, user_id: str) -> FlextResult[User]:
        # Database implementation
        pass

    def save(self, user: User) -> FlextResult[None]:
        # Database implementation
        pass
```

### 2. Dependency Injection

**Use FlextContainer para gerenciar dependências.**

```python
# ✅ Excelente - DI bem estruturado
from flext_core import FlextContainer

def setup_container() -> FlextContainer:
    """Configure application dependencies."""
    container = FlextContainer()

    # Infrastructure
    db_connection = create_database_connection()
    container.register("db_connection", db_connection)

    # Repositories
    user_repo = PostgreSQLUserRepository(db_connection)
    container.register("user_repository", user_repo)

    # Handlers
    deactivate_handler = DeactivateUserHandler(user_repo)
    container.register("deactivate_user_handler", deactivate_handler)

    return container

# Usage
container = setup_container()
handler_result = container.get("deactivate_user_handler")
if handler_result.success:
    handler = handler_result.data
    result = handler.process_command(command)


# ❌ Evite - Hard-coded dependencies
class DeactivateUserHandler:
    def __init__(self):
        # Hard-coded dependency
        self._user_repository = PostgreSQLUserRepository()  # ❌
```

### 3. Validation Strategy

**Implemente validação em múltiplas camadas.**

```python
# ✅ Excelente - Validação em camadas

# 1. INPUT VALIDATION - Na entrada
class CreateUserCommand(FlextCommand):
    def validate(self) -> FlextResult[None]:
        """Input validation - format and presence."""
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Valid email is required")

        if not self.name or len(self.name.strip()) < 2:
            return FlextResult.fail("Name must have at least 2 characters")

        return FlextResult.ok(None)

# 2. BUSINESS VALIDATION - No domínio
class User(FlextEntity[str]):
    def change_email(self, new_email: str) -> FlextResult[None]:
        """Business validation - domain rules."""
        if new_email == self._email:
            return FlextResult.fail("New email must be different")

        if self._is_suspended:
            return FlextResult.fail("Suspended users cannot change email")

        self._email = new_email
        return FlextResult.ok(None)

# 3. SYSTEM VALIDATION - Na aplicação
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Check if email already exists
        exists_result = self._user_repository.exists_by_email(command.email)
        if exists_result.success and exists_result.data:
            return FlextResult.fail("Email already registered")

        # Create user
        user = User.create(command.name, command.email)
        return self._user_repository.save(user)
```

## 📊 Error Handling Patterns

### 1. Result Chaining

**Combine multiple operations safely.**

```python
# ✅ Excelente - Result chaining
def transfer_money(
    from_account: str,
    to_account: str,
    amount: float
) -> FlextResult[TransferResult]:
    """Transfer money with comprehensive error handling."""

    # Chain of operations
    from_account_result = account_service.get_account(from_account)
    if from_account_result.is_failure:
        return FlextResult.fail(f"Source account error: {from_account_result.error}")

    to_account_result = account_service.get_account(to_account)
    if to_account_result.is_failure:
        return FlextResult.fail(f"Target account error: {to_account_result.error}")

    from_acc = from_account_result.data
    to_acc = to_account_result.data

    # Business validation
    withdraw_result = from_acc.withdraw(amount)
    if withdraw_result.is_failure:
        return FlextResult.fail(f"Withdrawal failed: {withdraw_result.error}")

    deposit_result = to_acc.deposit(amount)
    if deposit_result.is_failure:
        # Rollback withdrawal
        from_acc.deposit(amount)
        return FlextResult.fail(f"Deposit failed: {deposit_result.error}")

    # Success
    return FlextResult.ok(TransferResult(from_account, to_account, amount))
```

### 2. Error Classification

**Categorize errors for better handling.**

```python
# ✅ Excelente - Error classification
from enum import Enum

class ErrorType(str, Enum):
    VALIDATION = "validation"
    BUSINESS_RULE = "business_rule"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"

class FlextError:
    def __init__(self, error_type: ErrorType, message: str, details: dict = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

def create_user(data: dict) -> FlextResult[User]:
    # Validation error
    if not data.get("email"):
        error = FlextError(
            ErrorType.VALIDATION,
            "Email is required",
            {"field": "email", "value": data.get("email")}
        )
        return FlextResult.fail(error)

    # Business rule error
    if user_exists(data["email"]):
        error = FlextError(
            ErrorType.BUSINESS_RULE,
            "User already exists",
            {"email": data["email"]}
        )
        return FlextResult.fail(error)

    # Infrastructure error
    try:
        user = save_user(data)
        return FlextResult.ok(user)
    except DatabaseException as e:
        error = FlextError(
            ErrorType.INFRASTRUCTURE,
            "Database save failed",
            {"original_error": str(e)}
        )
        return FlextResult.fail(error)
```

## 🧪 Testing Best Practices

### 1. Test Structure

**Organize tests following AAA pattern.**

```python
# ✅ Excelente - AAA pattern (Arrange, Act, Assert)
def test_user_deactivation_success():
    """Test successful user deactivation."""
    # ARRANGE
    user_id = "user_123"
    user = User(user_id, "john@test.com", "John Doe")
    mock_repository = Mock(spec=UserRepository)
    mock_repository.find_by_id.return_value = FlextResult.ok(user)
    mock_repository.save.return_value = FlextResult.ok(None)

    handler = DeactivateUserHandler(mock_repository)
    command = DeactivateUserCommand(user_id, "Account cleanup")

    # ACT
    result = handler.process_command(command)

    # ASSERT
    assert result.success
    assert not user.is_active
    mock_repository.find_by_id.assert_called_once_with(user_id)
    mock_repository.save.assert_called_once_with(user)

def test_user_deactivation_user_not_found():
    """Test deactivation when user doesn't exist."""
    # ARRANGE
    user_id = "nonexistent"
    mock_repository = Mock(spec=UserRepository)
    mock_repository.find_by_id.return_value = FlextResult.fail("User not found")

    handler = DeactivateUserHandler(mock_repository)
    command = DeactivateUserCommand(user_id, "Test")

    # ACT
    result = handler.process_command(command)

    # ASSERT
    assert result.is_failure
    assert "User not found" in result.error
    mock_repository.save.assert_not_called()
```

### 2. Test Data Builders

**Use builders for complex test data.**

```python
# ✅ Excelente - Test data builders
class UserBuilder:
    def __init__(self):
        self._user_id = "default_id"
        self._name = "Default Name"
        self._email = "default@test.com"
        self._is_active = True

    def with_id(self, user_id: str) -> 'UserBuilder':
        self._user_id = user_id
        return self

    def with_name(self, name: str) -> 'UserBuilder':
        self._name = name
        return self

    def with_email(self, email: str) -> 'UserBuilder':
        self._email = email
        return self

    def inactive(self) -> 'UserBuilder':
        self._is_active = False
        return self

    def build(self) -> User:
        user = User(self._user_id, self._email, self._name)
        if not self._is_active:
            user.deactivate()
        return user

# Usage in tests
def test_inactive_user_cannot_change_email():
    # Readable test data creation
    user = (UserBuilder()
            .with_id("test_123")
            .with_email("old@test.com")
            .inactive()
            .build())

    result = user.change_email("new@test.com")
    assert result.is_failure
    assert "inactive" in result.error.lower()
```

### 3. Integration Test Patterns

**Test full workflows with real dependencies.**

```python
# ✅ Excelente - Integration tests
@pytest.mark.integration
def test_complete_user_registration_workflow():
    """Test complete user registration from command to persistence."""
    # Setup real dependencies
    container = create_test_container()  # Real container with test DB
    handler = container.get("register_user_handler").data

    # Test data
    command = RegisterUserCommand(
        name="Integration Test User",
        email="integration@test.com",
        password="SecurePass123!"
    )

    # Execute full workflow
    result = handler.process_command(command)

    # Verify success:
    assert result.success
    user = result.data
    assert user.name == command.name
    assert user.email == command.email
    assert user.is_active

    # Verify persistence
    user_repo = container.get("user_repository").data
    found_user_result = user_repo.find_by_email(command.email)
    assert found_user_result.success
    assert found_user_result.data.id == user.id
```

## 🔧 Performance Optimization

### 1. Lazy Loading

**Load data only when needed.**

```python
# ✅ Excelente - Lazy loading pattern
class Order(FlextEntity[str]):
    def __init__(self, order_id: str, customer_id: str):
        super().__init__(order_id)
        self._customer_id = customer_id
        self._customer: Optional[Customer] = None
        self._items: Optional[List[OrderItem]] = None

    @property
    def customer(self) -> Customer:
        """Lazy load customer when accessed."""
        if self._customer is None:
            # Load only when needed
            customer_result = self._customer_repository.find_by_id(self._customer_id)
            if customer_result.success:
                self._customer = customer_result.data
            else:
                raise ValueError(f"Customer not found: {self._customer_id}")

        return self._customer

    @property
    def items(self) -> List[OrderItem]:
        """Lazy load items when accessed."""
        if self._items is None:
            items_result = self._order_item_repository.find_by_order_id(self.id)
            self._items = items_result.data if items_result.success else []

        return self._items
```

### 2. Caching Strategy

**Cache expensive operations appropriately.**

```python
# ✅ Excelente - Strategic caching
from functools import lru_cache
from typing import Optional

class ProductCatalogService:
    def __init__(self, repository: ProductRepository):
        self._repository = repository
        self._cache: dict[str, Product] = {}

    def get_product(self, product_id: str) -> FlextResult[Product]:
        """Get product with intelligent caching."""
        # Check cache first
        if product_id in self._cache:
            return FlextResult.ok(self._cache[product_id])

        # Load from repository
        result = self._repository.find_by_id(product_id)
        if result.success:
            # Cache successful result
            self._cache[product_id] = result.data

        return result

    @lru_cache(maxsize=100)
    def calculate_discount(self, product_id: str, customer_tier: str) -> float:
        """Cache discount calculations."""
        # Expensive calculation cached automatically
        discount_rules = self._get_discount_rules(customer_tier)
        return self._apply_rules(product_id, discount_rules)

    def invalidate_product_cache(self, product_id: str) -> None:
        """Clear cache when product changes."""
        self._cache.pop(product_id, None)
        # Clear related caches
        self.calculate_discount.cache_clear()
```

## 🔐 Security Best Practices

### 1. Input Sanitization

**Always validate and sanitize inputs.**

```python
# ✅ Excelente - Input sanitization
import re
from html import escape

class UserInputValidator:
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z\s\'-]{1,100}$')

    @classmethod
    def sanitize_email(cls, email: str) -> FlextResult[str]:
        """Sanitize and validate email input."""
        if not email:
            return FlextResult.fail("Email is required")

        # Remove whitespace
        cleaned_email = email.strip().lower()

        # Validate format
        if not cls.EMAIL_PATTERN.match(cleaned_email):
            return FlextResult.fail("Invalid email format")

        # Check length
        if len(cleaned_email) > 254:  # RFC 5321 limit
            return FlextResult.fail("Email too long")

        return FlextResult.ok(cleaned_email)

    @classmethod
    def sanitize_name(cls, name: str) -> FlextResult[str]:
        """Sanitize and validate name input."""
        if not name:
            return FlextResult.fail("Name is required")

        # Remove extra whitespace
        cleaned_name = ' '.join(name.strip().split())

        # Validate pattern
        if not cls.SAFE_NAME_PATTERN.match(cleaned_name):
            return FlextResult.fail("Name contains invalid characters")

        # HTML escape for safety
        escaped_name = escape(cleaned_name)

        return FlextResult.ok(escaped_name)

# Usage
class CreateUserCommand(FlextCommand):
    def __init__(self, name: str, email: str):
        super().__init__()
        self.name = name
        self.email = email

    def validate(self) -> FlextResult[None]:
        # Sanitize inputs
        name_result = UserInputValidator.sanitize_name(self.name)
        if name_result.is_failure:
            return FlextResult.fail(f"Name validation: {name_result.error}")

        email_result = UserInputValidator.sanitize_email(self.email)
        if email_result.is_failure:
            return FlextResult.fail(f"Email validation: {email_result.error}")

        # Update with sanitized values
        self.name = name_result.data
        self.email = email_result.data

        return FlextResult.ok(None)
```

### 2. Access Control

**Implement proper authorization patterns.**

```python
# ✅ Excelente - Authorization pattern
from enum import Enum

class Permission(str, Enum):
    READ_USER = "user:read"
    WRITE_USER = "user:write"
    DELETE_USER = "user:delete"
    ADMIN_ACCESS = "admin:access"

class AuthorizationService:
    def __init__(self, user_permissions: dict[str, list[Permission]]):
        self._user_permissions = user_permissions

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_perms = self._user_permissions.get(user_id, [])
        return permission in user_perms or Permission.ADMIN_ACCESS in user_perms

class SecureUserHandler(FlextCommandHandler[DeleteUserCommand, None]):
    def __init__(self, auth_service: AuthorizationService, user_repo: UserRepository):
        super().__init__()
        self._auth_service = auth_service
        self._user_repo = user_repo

    def handle(self, command: DeleteUserCommand) -> FlextResult[None]:
        # Authorization check
        if not self._auth_service.check_permission(
            command.requester_id,
            Permission.DELETE_USER
        ):
            return FlextResult.fail("Insufficient permissions to delete user")

        # Business logic
        return self._user_repo.delete(command.user_id)
```

## 📈 Monitoring and Observability

### 1. Structured Logging

**Log important events with context.**

```python
# ✅ Excelente - Structured logging
import logging
import json
from datetime import datetime

class FlextLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def log_command_execution(
        self,
        command_type: str,
        command_id: str,
        user_id: str,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None
    ) -> None:
        """Log command execution with structured data."""
        log_data = {
            "event_type": "command_execution",
            "timestamp": datetime.utcnow().isoformat(),
            "command_type": command_type,
            "command_id": command_id,
            "user_id": user_id,
            "success": success,
            "duration_ms": duration_ms,
            "error": error
        }

        level = logging.INFO if success else logging.ERROR
        self._logger.log(level, json.dumps(log_data))

# Usage in handlers
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def __init__(self, user_repo: UserRepository):
        super().__init__()
        self._user_repo = user_repo
        self._logger = FlextLogger(__name__)

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        start_time = time.time()

        try:
            result = self._create_user(command)

            # Log successful execution
            duration = int((time.time() - start_time) * 1000)
            self._logger.log_command_execution(
                command_type="CreateUser",
                command_id=command.command_id,
                user_id=command.requester_id,
                success=result.success,
                duration_ms=duration,
                error=result.error if result.is_failure else None
            )

            return result

        except Exception as e:
            # Log unexpected errors
            duration = int((time.time() - start_time) * 1000)
            self._logger.log_command_execution(
                command_type="CreateUser",
                command_id=command.command_id,
                user_id=command.requester_id,
                success=False,
                duration_ms=duration,
                error=f"Unexpected error: {str(e)}"
            )
            return FlextResult.fail(f"Internal error: {str(e)}")
```

### 2. Metrics Collection

**Collect relevant business metrics.**

```python
# ✅ Excelente - Business metrics
from collections import defaultdict
from typing import Dict, Counter

class BusinessMetrics:
    def __init__(self):
        self._command_counts: Counter[str] = Counter()
        self._error_counts: Counter[str] = Counter()
        self._response_times: Dict[str, list[float]] = defaultdict(list)

    def record_command_execution(
        self,
        command_type: str,
        success: bool,
        duration_ms: float
    ) -> None:
        """Record command execution metrics."""
        self._command_counts[command_type] += 1
        self._response_times[command_type].append(duration_ms)

        if not success:
            self._error_counts[command_type] += 1

    def get_command_stats(self, command_type: str) -> dict:
        """Get statistics for a command type."""
        executions = self._command_counts[command_type]
        errors = self._error_counts[command_type]
        response_times = self._response_times[command_type]

        return {
            "total_executions": executions,
            "error_count": errors,
            "success_rate": (executions - errors) / executions if executions > 0 else 0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        }

# Integration with handlers
metrics = BusinessMetrics()

class MetricsDecorator:
    @staticmethod
    def with_metrics(handler_class):
        """Decorator to add metrics to handlers."""
        original_handle = handler_class.handle

        def handle_with_metrics(self, command):
            start_time = time.time()
            result = original_handle(self, command)
            duration = (time.time() - start_time) * 1000

            metrics.record_command_execution(
                command.__class__.__name__,
                result.success,
                duration
            )

            return result

        handler_class.handle = handle_with_metrics
        return handler_class

@MetricsDecorator.with_metrics
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    # Handler implementation automatically gets metrics
    pass
```

## 📚 Documentation Standards

### 1. Code Documentation

**Document complex business logic and design decisions.**

```python
# ✅ Excelente - Comprehensive documentation
class OrderAggregateRoot(FlextEntity[OrderId]):
    """Order aggregate root implementing business rules for e-commerce orders.

    This aggregate encapsulates all business rules related to order management,
    including item addition, status transitions, and payment processing.

    Business Rules:
        - Orders can only be modified in PENDING status
        - Total amount must be positive
        - Customer must exist and be active
        - Items must be available in inventory

    Domain Events:
        - OrderCreated: When order is first created
        - OrderConfirmed: When order moves to CONFIRMED status
        - OrderCancelled: When order is cancelled

    Args:
        order_id: Unique identifier for the order
        customer_id: ID of the customer placing the order

    Raises:
        ValueError: When customer_id is empty or invalid

    Example:
        >>> order = OrderAggregateRoot("order_123", "customer_456")
        >>> result = order.add_item(product, quantity=2)
        >>> if result.success:
        ...     confirm_result = order.confirm()
    """

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item to order with business validation.

        This method implements several business rules:
        1. Order must be in PENDING status
        2. Product must be available
        3. Quantity must be positive
        4. Total order value cannot exceed customer credit limit

        Args:
            product: Product to add to the order
            quantity: Number of items to add (must be > 0)

        Returns:
            FlextResult[None]: Success if item added, failure with reason otherwise

        Business Rules Applied:
            - Inventory check: Ensures product availability
            - Credit limit check: Validates customer can afford the addition
            - Order state validation: Only pending orders can be modified

        Example:
            >>> product = Product("laptop", 1500.00, stock=5)
            >>> result = order.add_item(product, quantity=1)
            >>> if result.is_failure:
            ...     print(f"Cannot add item: {result.error}")
        """
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail(
                f"Cannot modify order in {self._status} status. "
                f"Only PENDING orders can be modified."
            )

        if quantity <= 0:
            return FlextResult.fail(
                f"Quantity must be positive, got {quantity}"
            )

        # Check inventory availability
        if product.available_stock < quantity:
            return FlextResult.fail(
                f"Insufficient stock. Available: {product.available_stock}, "
                f"Requested: {quantity}"
            )

        # Calculate new total
        item_cost = product.price * quantity
        new_total = self.total_amount + item_cost

        # Check customer credit limit
        credit_check = self._validate_customer_credit_limit(new_total)
        if credit_check.is_failure:
            return credit_check

        # Add item
        order_item = OrderItem(product, quantity)
        self._items.append(order_item)

        # Reserve inventory
        product.reserve_stock(quantity)

        # Raise domain event
        self._raise_domain_event(ItemAddedToOrder(self.id, product.id, quantity))

        return FlextResult.ok(None)
```

### 2. API Documentation

**Provide clear examples and use cases.**

```python
# ✅ Excelente - API documentation with examples
class UserService:
    """Service for managing user lifecycle and business operations.

    This service provides high-level operations for user management,
    coordinating between domain entities and infrastructure concerns.

    Thread Safety:
        This service is thread-safe when used with proper dependency injection.

    Dependencies:
        - UserRepository: For data persistence
        - EmailService: For user notifications
        - AuthorizationService: For permission checks

    Examples:
        Basic Usage:
            >>> container = FlextContainer()
            >>> user_service = container.get('user_service').data
            >>> result = user_service.create_user("John Doe", "john@example.com")

        Error Handling:
            >>> result = user_service.create_user("", "invalid-email")
            >>> if result.is_failure:
            ...     print(f"User creation failed: {result.error}")

        Advanced Usage with Validation:
            >>> create_data = UserCreationData(
            ...     name="Jane Smith",
            ...     email="jane@company.com",
            ...     department="Engineering"
            ... )
            >>> result = user_service.create_user_with_validation(create_data)
    """

    def create_user(self, name: str, email: str) -> FlextResult[User]:
        """Create a new user with basic validation.

        This method creates a user with minimal validation. For more comprehensive
        validation, use create_user_with_validation().

        Args:
            name: Full name of the user (2-100 characters)
            email: Valid email address (will be normalized to lowercase)

        Returns:
            FlextResult[User]: Created user on success, error message on failure

        Possible Errors:
            - "Name is required": When name is empty or whitespace
            - "Invalid email format": When email format is invalid
            - "Email already exists": When email is already registered
            - "Database error": When persistence fails

        Business Rules:
            - Email addresses are automatically normalized to lowercase
            - Duplicate emails are not allowed (case-insensitive)
            - New users are created in ACTIVE status by default
            - Welcome email is sent automatically on successful creation

        Example:
            >>> # Basic user creation
            >>> result = user_service.create_user("Alice Johnson", "alice@test.com")
            >>> if result.success:
            ...     user = result.data
            ...     print(f"Created user {user.id} with email {user.email}")
            ... else:
            ...     print(f"Failed to create user: {result.error}")

            >>> # Handle specific error cases
            >>> result = user_service.create_user("", "invalid")
            >>> match result.error:
            ...     case error if "name" in error.lower():
            ...         print("Please provide a valid name")
            ...     case error if "email" in error.lower():
            ...         print("Please provide a valid email address")
            ...     case _:
            ...         print(f"Unexpected error: {result.error}")
        """
        pass
```

## ⚡ Performance Considerations

### 1. Memory Management

**Be mindful of memory usage in long-running processes.**

```python
# ✅ Excelente - Efficient memory usage
class LargeDataProcessor:
    def process_users_batch(self, user_ids: list[str]) -> FlextResult[ProcessingReport]:
        """Process large batches of users efficiently."""
        report = ProcessingReport()

        # Process in chunks to avoid memory issues
        chunk_size = 100
        for i in range(0, len(user_ids), chunk_size):
            chunk = user_ids[i:i + chunk_size]

            # Process chunk
            chunk_result = self._process_user_chunk(chunk)
            if chunk_result.is_failure:
                return chunk_result

            # Update report
            report.merge(chunk_result.data)

            # Optional: Force garbage collection for large datasets
            if i % 1000 == 0:
                import gc
                gc.collect()

        return FlextResult.ok(report)

    def _process_user_chunk(self, user_ids: list[str]) -> FlextResult[ProcessingReport]:
        """Process a chunk of users with proper resource cleanup."""
        processed_users = []

        try:
            for user_id in user_ids:
                user_result = self._user_repository.find_by_id(user_id)
                if user_result.success:
                    processed_users.append(user_result.data)

            # Process users
            report = self._generate_report(processed_users)

            # Clear references to help GC
            processed_users.clear()

            return FlextResult.ok(report)

        except Exception as e:
            # Cleanup on error
            processed_users.clear()
            return FlextResult.fail(f"Chunk processing failed: {str(e)}")
```

### 2. Database Optimization

**Optimize database operations.**

```python
# ✅ Excelente - Optimized database operations
class OptimizedUserRepository:
    def find_users_by_criteria(
        self,
        criteria: UserSearchCriteria,
        page: int = 0,
        page_size: int = 50
    ) -> FlextResult[list[User]]:
        """Find users with optimized query and pagination."""

        # Validate pagination parameters
        if page < 0 or page_size <= 0 or page_size > 1000:
            return FlextResult.fail("Invalid pagination parameters")

        try:
            # Build optimized query with proper indexing
            query = self._build_optimized_query(criteria)

            # Add pagination
            offset = page * page_size
            query = query.offset(offset).limit(page_size)

            # Execute with connection management
            with self._connection_pool.get_connection() as conn:
                users = query.all()

                # Convert to domain objects efficiently
                user_entities = [
                    self._to_domain_user(user_row)
                    for user_row in users
                ]

                return FlextResult.ok(user_entities)

        except DatabaseException as e:
            return FlextResult.fail(f"Database query failed: {str(e)}")

    def bulk_update_users(self, updates: list[UserUpdate]) -> FlextResult[int]:
        """Perform bulk updates efficiently."""
        if not updates:
            return FlextResult.ok(0)

        try:
            with self._connection_pool.get_connection() as conn:
                # Use bulk operations instead of individual updates
                update_data = [
                    {
                        'id': update.user_id,
                        'name': update.name,
                        'email': update.email,
                        'updated_at': datetime.utcnow()
                    }
                    for update in updates
                ]

                # Bulk update in single query
                result = conn.execute(
                    update(User.__table__)
                    .where(User.id == bindparam('id'))
                    .values(
                        name=bindparam('name'),
                        email=bindparam('email'),
                        updated_at=bindparam('updated_at')
                    ),
                    update_data
                )

                return FlextResult.ok(result.rowcount)

        except DatabaseException as e:
            return FlextResult.fail(f"Bulk update failed: {str(e)}")
```

---

## 🎯 Resumo das Boas Práticas

### ✅ FAÇA

- Use type hints em todas as funções
- Prefira FlextResult sobre exceções
- Implemente Clean Architecture
- Valide dados em múltiplas camadas
- Use dependency injection
- Escreva testes abrangentes
- Documente decisões de design
- Monitore performance e erros
- Sanitize inputs de usuário
- Implemente logging estruturado

### ❌ EVITE

- Exceções para fluxo de controle
- Hard-coded dependencies
- Mutabilidade desnecessária
- Código sem type hints
- Operações de banco não otimizadas
- Validação apenas na entrada
- Código sem testes
- Logs não estruturados
- Inputs não sanitizados
- Memory leaks em processamento batch

---

**Seguindo essas práticas, você construirá sistemas robustos, mantíveis e escaláveis com FLEXT Core!**
