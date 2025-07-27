# Exemplos Práticos - FLEXT Core

**Exemplos reais de implementação usando FLEXT Core em sistemas empresariais**

## 🎯 Visão Geral dos Exemplos

Esta seção apresenta exemplos práticos e casos de uso reais demonstrando como usar FLEXT Core em cenários empresariais. Cada exemplo é completo, testável e segue as melhores práticas da biblioteca.

### 📋 Índice de Exemplos

1. **[Sistema de Usuários](#sistema-de-usuários)** - CRUD básico com validação
2. **[E-commerce com Pedidos](#e-commerce-com-pedidos)** - Agregados e regras complexas
3. **[Sistema de Notificações](#sistema-de-notificações)** - Event-driven architecture
4. **[API REST](#api-rest)** - Integração com frameworks web
5. **[Processamento em Lote](#processamento-em-lote)** - Operações assíncronas
6. **[Sistema de Auditoria](#sistema-de-auditoria)** - Logging e compliance

---

## 🔐 Sistema de Usuários

**Exemplo completo de gerenciamento de usuários com autenticação e autorização.**

```python
"""
Sistema completo de usuários demonstrando:
- Domain entities com regras de negócio
- Command/Handler pattern
- Validation pattern  
- Repository pattern
- Dependency injection
"""

from flext_core import FlextEntity, FlextResult, FlextContainer
from flext_core.patterns import FlextCommand, FlextCommandHandler, FlextValidator
from typing import NewType, Optional
from datetime import datetime
from enum import Enum
import hashlib
import secrets

# ========================
# DOMAIN LAYER
# ========================

UserId = NewType("UserId", str)
Email = NewType("Email", str)
HashedPassword = NewType("HashedPassword", str)

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class User(FlextEntity[UserId]):
    """Domain entity representing a system user."""
    
    def __init__(
        self,
        user_id: UserId,
        email: Email,
        name: str,
        password_hash: HashedPassword,
        role: UserRole = UserRole.USER
    ):
        super().__init__(user_id)
        self._email = email
        self._name = name
        self._password_hash = password_hash
        self._role = role
        self._status = UserStatus.ACTIVE
        self._created_at = datetime.now()
        self._last_login: Optional[datetime] = None
        self.failed_login_attempts = 0
    
    @property
    def email(self) -> Email:
        return self._email
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def role(self) -> UserRole:
        return self._role
    
    @property
    def status(self) -> UserStatus:
        return self._status
    
    @property
    def is_active(self) -> bool:
        return self._status == UserStatus.ACTIVE
    
    @property
    def is_locked(self) -> bool:
        return self.failed_login_attempts >= 5
    
    def change_email(self, new_email: Email) -> FlextResult[None]:
        """Change user email with business validation."""
        if new_email == self._email:
            return FlextResult.fail("New email must be different from current")
        
        if self._status == UserStatus.SUSPENDED:
            return FlextResult.fail("Suspended users cannot change email")
        
        self._email = new_email
        return FlextResult.ok(None)
    
    def change_password(self, new_password_hash: HashedPassword) -> FlextResult[None]:
        """Change user password."""
        if new_password_hash == self._password_hash:
            return FlextResult.fail("New password must be different")
        
        self._password_hash = new_password_hash
        self.failed_login_attempts = 0  # Reset on password change
        return FlextResult.ok(None)
    
    def authenticate(self, password_hash: HashedPassword) -> FlextResult[None]:
        """Authenticate user with password."""
        if self._status != UserStatus.ACTIVE:
            return FlextResult.fail(f"User is {self._status.value}")
        
        if self.is_locked:
            return FlextResult.fail("Account is locked due to failed login attempts")
        
        if self._password_hash != password_hash:
            self.failed_login_attempts += 1
            return FlextResult.fail("Invalid credentials")
        
        # Success
        self._last_login = datetime.now()
        self.failed_login_attempts = 0
        return FlextResult.ok(None)
    
    def suspend(self, reason: str) -> FlextResult[None]:
        """Suspend user account."""
        if self._status == UserStatus.SUSPENDED:
            return FlextResult.fail("User is already suspended")
        
        self._status = UserStatus.SUSPENDED
        return FlextResult.ok(None)
    
    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self._status == UserStatus.ACTIVE:
            return FlextResult.fail("User is already active")
        
        self._status = UserStatus.ACTIVE
        self.failed_login_attempts = 0
        return FlextResult.ok(None)
    
    @classmethod
    def create(
        cls,
        email: Email,
        name: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> FlextResult['User']:
        """Factory method to create new user."""
        # Generate ID
        user_id = UserId(f"user_{secrets.token_hex(8)}")
        
        # Hash password
        password_hash = HashedPassword(
            hashlib.sha256(password.encode()).hexdigest()
        )
        
        # Create user
        user = cls(user_id, email, name, password_hash, role)
        return FlextResult.ok(user)

# ========================
# APPLICATION LAYER
# ========================

class CreateUserCommand(FlextCommand):
    """Command to create a new user."""
    
    def __init__(self, name: str, email: str, password: str, role: str = "user"):
        super().__init__()
        self.name = name
        self.email = email
        self.password = password
        self.role = role
    
    def validate(self) -> FlextResult[None]:
        """Validate command inputs."""
        if not self.name or len(self.name.strip()) < 2:
            return FlextResult.fail("Name must have at least 2 characters")
        
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Valid email is required")
        
        if not self.password or len(self.password) < 8:
            return FlextResult.fail("Password must have at least 8 characters")
        
        try:
            UserRole(self.role)
        except ValueError:
            return FlextResult.fail(f"Invalid role: {self.role}")
        
        return FlextResult.ok(None)

class AuthenticateUserCommand(FlextCommand):
    """Command to authenticate a user."""
    
    def __init__(self, email: str, password: str):
        super().__init__()
        self.email = email
        self.password = password
    
    def validate(self) -> FlextResult[None]:
        if not self.email:
            return FlextResult.fail("Email is required")
        
        if not self.password:
            return FlextResult.fail("Password is required")
        
        return FlextResult.ok(None)

# User Repository Interface
from abc import ABC, abstractmethod

class UserRepository(ABC):
    """Repository interface for user persistence."""
    
    @abstractmethod
    def save(self, user: User) -> FlextResult[None]:
        """Save user to storage."""
        pass
    
    @abstractmethod
    def find_by_id(self, user_id: UserId) -> FlextResult[User]:
        """Find user by ID."""
        pass
    
    @abstractmethod
    def find_by_email(self, email: Email) -> FlextResult[User]:
        """Find user by email."""
        pass
    
    @abstractmethod
    def exists_by_email(self, email: Email) -> FlextResult[bool]:
        """Check if user exists by email."""
        pass

# Command Handlers
class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    """Handler for user creation."""
    
    def __init__(self, user_repository: UserRepository):
        super().__init__()
        self._user_repository = user_repository
    
    def can_handle(self, command) -> bool:
        return isinstance(command, CreateUserCommand)
    
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Handle user creation."""
        # Check if email already exists
        email = Email(command.email.lower())
        exists_result = self._user_repository.exists_by_email(email)
        if exists_result.is_success and exists_result.data:
            return FlextResult.fail("Email already registered")
        
        # Create user
        role = UserRole(command.role)
        user_result = User.create(email, command.name, command.password, role)
        if user_result.is_failure:
            return user_result
        
        user = user_result.data
        
        # Save user
        save_result = self._user_repository.save(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save user: {save_result.error}")
        
        return FlextResult.ok(user)

class AuthenticateUserHandler(FlextCommandHandler[AuthenticateUserCommand, User]):
    """Handler for user authentication."""
    
    def __init__(self, user_repository: UserRepository):
        super().__init__()
        self._user_repository = user_repository
    
    def can_handle(self, command) -> bool:
        return isinstance(command, AuthenticateUserCommand)
    
    def handle(self, command: AuthenticateUserCommand) -> FlextResult[User]:
        """Handle user authentication."""
        # Find user by email
        email = Email(command.email.lower())
        user_result = self._user_repository.find_by_email(email)
        if user_result.is_failure:
            return FlextResult.fail("Invalid credentials")
        
        user = user_result.data
        
        # Authenticate
        password_hash = HashedPassword(
            hashlib.sha256(command.password.encode()).hexdigest()
        )
        auth_result = user.authenticate(password_hash)
        if auth_result.is_failure:
            # Save failed attempt
            self._user_repository.save(user)
            return FlextResult.fail(auth_result.error)
        
        # Save successful login
        save_result = self._user_repository.save(user)
        if save_result.is_failure:
            return FlextResult.fail("Authentication succeeded but failed to update login time")
        
        return FlextResult.ok(user)

# ========================
# INFRASTRUCTURE LAYER
# ========================

class InMemoryUserRepository(UserRepository):
    """In-memory implementation for testing."""
    
    def __init__(self):
        self._users: dict[UserId, User] = {}
        self._email_index: dict[Email, UserId] = {}
    
    def save(self, user: User) -> FlextResult[None]:
        """Save user to memory."""
        self._users[user.id] = user
        self._email_index[user.email] = user.id
        return FlextResult.ok(None)
    
    def find_by_id(self, user_id: UserId) -> FlextResult[User]:
        """Find user by ID."""
        if user_id not in self._users:
            return FlextResult.fail(f"User not found: {user_id}")
        
        return FlextResult.ok(self._users[user_id])
    
    def find_by_email(self, email: Email) -> FlextResult[User]:
        """Find user by email."""
        if email not in self._email_index:
            return FlextResult.fail(f"User not found: {email}")
        
        user_id = self._email_index[email]
        return FlextResult.ok(self._users[user_id])
    
    def exists_by_email(self, email: Email) -> FlextResult[bool]:
        """Check if user exists by email."""
        return FlextResult.ok(email in self._email_index)

# ========================
# APPLICATION SETUP
# ========================

def setup_user_system() -> FlextContainer:
    """Setup complete user system with dependencies."""
    container = FlextContainer()
    
    # Infrastructure
    user_repository = InMemoryUserRepository()
    container.register("user_repository", user_repository)
    
    # Handlers
    create_handler = CreateUserHandler(user_repository)
    auth_handler = AuthenticateUserHandler(user_repository)
    
    container.register("create_user_handler", create_handler)
    container.register("authenticate_user_handler", auth_handler)
    
    return container

# ========================
# EXAMPLE USAGE
# ========================

def demo_user_system():
    """Demonstrate complete user system."""
    print("🔐 User Management System Demo")
    print("=" * 50)
    
    # Setup system
    container = setup_user_system()
    create_handler = container.get("create_user_handler").data
    auth_handler = container.get("authenticate_user_handler").data
    
    # Create user
    print("\n1. Creating user...")
    create_command = CreateUserCommand(
        name="Alice Johnson",
        email="alice@company.com",
        password="SecurePass123!",
        role="user"
    )
    
    create_result = create_handler.process_command(create_command)
    if create_result.is_success:
        user = create_result.data
        print(f"✅ User created: {user.name} ({user.id})")
        print(f"   Email: {user.email}")
        print(f"   Role: {user.role}")
        print(f"   Status: {user.status}")
    else:
        print(f"❌ Create failed: {create_result.error}")
        return
    
    # Authenticate user
    print("\n2. Authenticating user...")
    auth_command = AuthenticateUserCommand(
        email="alice@company.com",
        password="SecurePass123!"
    )
    
    auth_result = auth_handler.process_command(auth_command)
    if auth_result.is_success:
        authenticated_user = auth_result.data
        print(f"✅ Authentication successful")
        print(f"   Last login: {authenticated_user._last_login}")
    else:
        print(f"❌ Authentication failed: {auth_result.error}")
    
    # Test wrong password
    print("\n3. Testing wrong password...")
    wrong_auth_command = AuthenticateUserCommand(
        email="alice@company.com",
        password="WrongPassword"
    )
    
    wrong_auth_result = auth_handler.process_command(wrong_auth_command)
    print(f"❌ Expected failure: {wrong_auth_result.error}")
    
    # Test duplicate email
    print("\n4. Testing duplicate email...")
    duplicate_command = CreateUserCommand(
        name="Bob Smith",
        email="alice@company.com",  # Same email
        password="AnotherPass456!"
    )
    
    duplicate_result = create_handler.process_command(duplicate_command)
    print(f"❌ Expected failure: {duplicate_result.error}")
    
    print("\n" + "=" * 50)
    print("🎉 User system demo completed successfully!")

if __name__ == "__main__":
    demo_user_system()
```

---

## 🛒 E-commerce com Pedidos

**Sistema completo de e-commerce com agregados complexos e regras de negócio.**

```python
"""
Sistema de e-commerce demonstrando:
- Aggregate roots complexos
- Domain events
- Business rules enforcement
- Money value objects
- Inventory management
"""

from flext_core import FlextEntity, FlextValueObject, FlextResult
from flext_core.patterns import FlextCommand, FlextCommandHandler
from typing import NewType, List, Optional
from datetime import datetime
from enum import Enum
from decimal import Decimal

# ========================
# VALUE OBJECTS
# ========================

class Money(FlextValueObject):
    """Immutable money value object."""
    
    def __init__(self, amount: Decimal, currency: str = "USD"):
        if amount < 0:
            raise ValueError("Money amount cannot be negative")
        
        if not currency or len(currency) != 3:
            raise ValueError("Currency must be 3-letter code")
        
        self._amount = amount
        self._currency = currency.upper()
    
    @property
    def amount(self) -> Decimal:
        return self._amount
    
    @property
    def currency(self) -> str:
        return self._currency
    
    def add(self, other: 'Money') -> 'Money':
        """Add two money amounts."""
        if other.currency != self.currency:
            raise ValueError(f"Cannot add {other.currency} to {self.currency}")
        
        return Money(self.amount + other.amount, self.currency)
    
    def multiply(self, factor: int | float) -> 'Money':
        """Multiply money by a factor."""
        if factor < 0:
            raise ValueError("Cannot multiply by negative factor")
        
        return Money(self.amount * Decimal(str(factor)), self.currency)
    
    def __str__(self) -> str:
        return f"{self.amount:.2f} {self.currency}"
    
    @classmethod
    def zero(cls, currency: str = "USD") -> 'Money':
        """Create zero money amount."""
        return cls(Decimal("0"), currency)

# ========================
# DOMAIN ENTITIES
# ========================

ProductId = NewType("ProductId", str)
OrderId = NewType("OrderId", str)
CustomerId = NewType("CustomerId", str)

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed" 
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Product(FlextEntity[ProductId]):
    """Product entity with inventory management."""
    
    def __init__(
        self,
        product_id: ProductId,
        name: str,
        price: Money,
        stock_quantity: int,
        description: str = ""
    ):
        super().__init__(product_id)
        self._name = name
        self._price = price
        self._stock_quantity = stock_quantity
        self._description = description
        self._reserved_stock = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def price(self) -> Money:
        return self._price
    
    @property
    def stock_quantity(self) -> int:
        return self._stock_quantity
    
    @property
    def available_stock(self) -> int:
        """Available stock considering reservations."""
        return self._stock_quantity - self._reserved_stock
    
    def reserve_stock(self, quantity: int) -> FlextResult[None]:
        """Reserve stock for an order."""
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        if self.available_stock < quantity:
            return FlextResult.fail(
                f"Insufficient stock. Available: {self.available_stock}, "
                f"Requested: {quantity}"
            )
        
        self._reserved_stock += quantity
        return FlextResult.ok(None)
    
    def release_stock(self, quantity: int) -> FlextResult[None]:
        """Release reserved stock."""
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        if self._reserved_stock < quantity:
            return FlextResult.fail(
                f"Cannot release more than reserved. Reserved: {self._reserved_stock}"
            )
        
        self._reserved_stock -= quantity
        return FlextResult.ok(None)
    
    def confirm_stock_usage(self, quantity: int) -> FlextResult[None]:
        """Confirm stock usage (remove from both stock and reserved)."""
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        if self._reserved_stock < quantity:
            return FlextResult.fail("Not enough reserved stock")
        
        self._stock_quantity -= quantity
        self._reserved_stock -= quantity
        return FlextResult.ok(None)

class OrderItem:
    """Order item with product and quantity."""
    
    def __init__(self, product: Product, quantity: int):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        self.product = product
        self.quantity = quantity
    
    def total_price(self) -> Money:
        """Calculate total price for this item."""
        return self.product.price.multiply(self.quantity)

class Order(FlextEntity[OrderId]):
    """Order aggregate root with complex business rules."""
    
    def __init__(self, order_id: OrderId, customer_id: CustomerId):
        super().__init__(order_id)
        self._customer_id = customer_id
        self._items: List[OrderItem] = []
        self._status = OrderStatus.PENDING
        self._created_at = datetime.now()
        self._confirmed_at: Optional[datetime] = None
        self._shipping_address: Optional[str] = None
    
    @property
    def customer_id(self) -> CustomerId:
        return self._customer_id
    
    @property
    def status(self) -> OrderStatus:
        return self._status
    
    @property
    def items(self) -> List[OrderItem]:
        return self._items.copy()
    
    @property
    def total_amount(self) -> Money:
        """Calculate total order amount."""
        if not self._items:
            return Money.zero()
        
        total = self._items[0].total_price()
        for item in self._items[1:]:
            total = total.add(item.total_price())
        
        return total
    
    @property
    def item_count(self) -> int:
        """Total number of items in order."""
        return sum(item.quantity for item in self._items)
    
    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item to order with business validation."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail(
                f"Cannot modify order in {self._status} status"
            )
        
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        # Check if product already in order
        for existing_item in self._items:
            if existing_item.product.id == product.id:
                return FlextResult.fail(
                    f"Product {product.name} already in order. "
                    f"Use update_item_quantity instead."
                )
        
        # Reserve stock
        reserve_result = product.reserve_stock(quantity)
        if reserve_result.is_failure:
            return reserve_result
        
        # Add item
        item = OrderItem(product, quantity)
        self._items.append(item)
        
        return FlextResult.ok(None)
    
    def remove_item(self, product_id: ProductId) -> FlextResult[None]:
        """Remove item from order."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail(
                f"Cannot modify order in {self._status} status"
            )
        
        # Find item
        item_to_remove = None
        for item in self._items:
            if item.product.id == product_id:
                item_to_remove = item
                break
        
        if not item_to_remove:
            return FlextResult.fail(f"Product {product_id} not found in order")
        
        # Release stock
        release_result = item_to_remove.product.release_stock(
            item_to_remove.quantity
        )
        if release_result.is_failure:
            return release_result
        
        # Remove item
        self._items.remove(item_to_remove)
        
        return FlextResult.ok(None)
    
    def set_shipping_address(self, address: str) -> FlextResult[None]:
        """Set shipping address."""
        if not address.strip():
            return FlextResult.fail("Shipping address cannot be empty")
        
        if self._status not in [OrderStatus.PENDING, OrderStatus.CONFIRMED]:
            return FlextResult.fail("Cannot change address for shipped orders")
        
        self._shipping_address = address.strip()
        return FlextResult.ok(None)
    
    def confirm(self) -> FlextResult[None]:
        """Confirm order with business validation."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail(
                f"Cannot confirm order in {self._status} status"
            )
        
        if not self._items:
            return FlextResult.fail("Order must have at least one item")
        
        if not self._shipping_address:
            return FlextResult.fail("Shipping address is required")
        
        # Confirm stock usage for all items
        for item in self._items:
            confirm_result = item.product.confirm_stock_usage(item.quantity)
            if confirm_result.is_failure:
                return FlextResult.fail(
                    f"Stock confirmation failed for {item.product.name}: "
                    f"{confirm_result.error}"
                )
        
        # Update status
        self._status = OrderStatus.CONFIRMED
        self._confirmed_at = datetime.now()
        
        return FlextResult.ok(None)
    
    def cancel(self) -> FlextResult[None]:
        """Cancel order and release stock."""
        if self._status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
            return FlextResult.fail(
                f"Cannot cancel order in {self._status} status"
            )
        
        if self._status == OrderStatus.CANCELLED:
            return FlextResult.fail("Order is already cancelled")
        
        # Release all reserved stock
        for item in self._items:
            if self._status == OrderStatus.PENDING:
                # Release reserved stock
                item.product.release_stock(item.quantity)
            # Note: For confirmed orders, stock is already consumed
        
        self._status = OrderStatus.CANCELLED
        return FlextResult.ok(None)

# ========================
# APPLICATION COMMANDS
# ========================

class CreateOrderCommand(FlextCommand):
    """Command to create a new order."""
    
    def __init__(self, customer_id: str, shipping_address: str):
        super().__init__()
        self.customer_id = customer_id
        self.shipping_address = shipping_address
    
    def validate(self) -> FlextResult[None]:
        if not self.customer_id:
            return FlextResult.fail("Customer ID is required")
        
        if not self.shipping_address.strip():
            return FlextResult.fail("Shipping address is required")
        
        return FlextResult.ok(None)

class AddItemToOrderCommand(FlextCommand):
    """Command to add item to order."""
    
    def __init__(self, order_id: str, product_id: str, quantity: int):
        super().__init__()
        self.order_id = order_id
        self.product_id = product_id
        self.quantity = quantity
    
    def validate(self) -> FlextResult[None]:
        if not self.order_id:
            return FlextResult.fail("Order ID is required")
        
        if not self.product_id:
            return FlextResult.fail("Product ID is required")
        
        if self.quantity <= 0:
            return FlextResult.fail("Quantity must be positive")
        
        return FlextResult.ok(None)

class ConfirmOrderCommand(FlextCommand):
    """Command to confirm an order."""
    
    def __init__(self, order_id: str):
        super().__init__()
        self.order_id = order_id
    
    def validate(self) -> FlextResult[None]:
        if not self.order_id:
            return FlextResult.fail("Order ID is required")
        
        return FlextResult.ok(None)

# ========================
# EXAMPLE USAGE
# ========================

def demo_ecommerce_system():
    """Demonstrate complete e-commerce system."""
    print("🛒 E-commerce Order System Demo")
    print("=" * 50)
    
    # Create products
    laptop = Product(
        ProductId("laptop_001"),
        "Gaming Laptop",
        Money(Decimal("1500.00")),
        5,
        "High-performance gaming laptop"
    )
    
    mouse = Product(
        ProductId("mouse_001"),
        "Gaming Mouse",
        Money(Decimal("75.00")),
        20,
        "Precision gaming mouse"
    )
    
    print("📦 Products created:")
    print(f"   {laptop.name}: {laptop.price} (Stock: {laptop.stock_quantity})")
    print(f"   {mouse.name}: {mouse.price} (Stock: {mouse.stock_quantity})")
    
    # Create order
    print("\n🛒 Creating order...")
    order = Order(OrderId("order_001"), CustomerId("customer_123"))
    
    # Set shipping address
    address_result = order.set_shipping_address("123 Main St, City, State 12345")
    if address_result.is_success:
        print("✅ Shipping address set")
    else:
        print(f"❌ Address error: {address_result.error}")
    
    # Add items to order
    print("\n📋 Adding items to order...")
    
    # Add laptop
    laptop_result = order.add_item(laptop, 1)
    if laptop_result.is_success:
        print(f"✅ Added: 1x {laptop.name}")
        print(f"   Available stock now: {laptop.available_stock}")
    else:
        print(f"❌ Failed to add laptop: {laptop_result.error}")
    
    # Add mice
    mouse_result = order.add_item(mouse, 2)
    if mouse_result.is_success:
        print(f"✅ Added: 2x {mouse.name}")
        print(f"   Available stock now: {mouse.available_stock}")
    else:
        print(f"❌ Failed to add mouse: {mouse_result.error}")
    
    # Display order summary
    print(f"\n📊 Order Summary:")
    print(f"   Order ID: {order.id}")
    print(f"   Customer: {order.customer_id}")
    print(f"   Status: {order.status}")
    print(f"   Items: {order.item_count}")
    print(f"   Total: {order.total_amount}")
    
    # List items
    print(f"\n📝 Order Items:")
    for item in order.items:
        print(f"   - {item.quantity}x {item.product.name} @ {item.product.price} = {item.total_price()}")
    
    # Confirm order
    print(f"\n✅ Confirming order...")
    confirm_result = order.confirm()
    if confirm_result.is_success:
        print(f"✅ Order confirmed successfully")
        print(f"   Status: {order.status}")
        print(f"   Laptop stock after confirmation: {laptop.stock_quantity} (Reserved: {laptop._reserved_stock})")
        print(f"   Mouse stock after confirmation: {mouse.stock_quantity} (Reserved: {mouse._reserved_stock})")
    else:
        print(f"❌ Confirmation failed: {confirm_result.error}")
    
    # Test insufficient stock
    print(f"\n🚫 Testing insufficient stock...")
    order2 = Order(OrderId("order_002"), CustomerId("customer_456"))
    order2.set_shipping_address("456 Oak Ave, City, State 67890")
    
    # Try to add more laptops than available
    insufficient_result = order2.add_item(laptop, 10)  # Only 4 left after first order
    print(f"❌ Expected error: {insufficient_result.error}")
    
    print("\n" + "=" * 50)
    print("🎉 E-commerce system demo completed!")

if __name__ == "__main__":
    demo_ecommerce_system()
```

---

## 📬 Sistema de Notificações

**Sistema event-driven com handlers assíncronos e diferentes canais de notificação.**

```python
"""
Sistema de notificações demonstrando:
- Event-driven architecture
- Multiple notification channels
- Event handlers with retries
- Template system
- Async processing patterns
"""

from flext_core import FlextResult
from flext_core.patterns import FlextEventHandler, FlextHandlerRegistry
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json

# ========================
# DOMAIN EVENTS
# ========================

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class DomainEvent:
    """Base domain event class."""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_id = f"evt_{hash(str(datetime.now()))}"
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()
        self.metadata: Dict[str, Any] = {}

class UserRegisteredEvent(DomainEvent):
    """Event fired when a user registers."""
    
    def __init__(self, user_id: str, email: str, name: str):
        super().__init__("user.registered", {
            "user_id": user_id,
            "email": email,
            "name": name
        })

class OrderConfirmedEvent(DomainEvent):
    """Event fired when an order is confirmed."""
    
    def __init__(self, order_id: str, customer_id: str, total_amount: float):
        super().__init__("order.confirmed", {
            "order_id": order_id,
            "customer_id": customer_id,
            "total_amount": total_amount
        })

class PaymentFailedEvent(DomainEvent):
    """Event fired when payment fails."""
    
    def __init__(self, payment_id: str, user_id: str, amount: float, reason: str):
        super().__init__("payment.failed", {
            "payment_id": payment_id,
            "user_id": user_id,
            "amount": amount,
            "reason": reason
        })

# ========================
# NOTIFICATION SYSTEM
# ========================

class NotificationTemplate:
    """Template for notifications."""
    
    def __init__(
        self,
        template_id: str,
        subject_template: str,
        body_template: str,
        channel: NotificationChannel
    ):
        self.template_id = template_id
        self.subject_template = subject_template
        self.body_template = body_template
        self.channel = channel
    
    def render(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Render template with data."""
        try:
            subject = self.subject_template.format(**data)
            body = self.body_template.format(**data)
            
            return {
                "subject": subject,
                "body": body
            }
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

class Notification:
    """Notification to be sent."""
    
    def __init__(
        self,
        recipient: str,
        channel: NotificationChannel,
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.notification_id = f"notif_{hash(recipient + str(datetime.now()))}"
        self.recipient = recipient
        self.channel = channel
        self.subject = subject
        self.body = body
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.sent_at: Optional[datetime] = None
        self.failed_at: Optional[datetime] = None
        self.retry_count = 0
        self.max_retries = 3

# ========================
# NOTIFICATION PROVIDERS
# ========================

class NotificationProvider(ABC):
    """Abstract notification provider."""
    
    @abstractmethod
    def send(self, notification: Notification) -> FlextResult[None]:
        """Send notification through this provider."""
        pass
    
    @abstractmethod
    def get_channel(self) -> NotificationChannel:
        """Get the channel this provider handles."""
        pass

class EmailProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    def get_channel(self) -> NotificationChannel:
        return NotificationChannel.EMAIL
    
    def send(self, notification: Notification) -> FlextResult[None]:
        """Send email notification."""
        try:
            # Simulate email sending
            print(f"📧 EMAIL SENT to {notification.recipient}")
            print(f"   Subject: {notification.subject}")
            print(f"   Body: {notification.body[:100]}...")
            
            notification.sent_at = datetime.now()
            return FlextResult.ok(None)
            
        except Exception as e:
            notification.failed_at = datetime.now()
            return FlextResult.fail(f"Email send failed: {str(e)}")

class SMSProvider(NotificationProvider):
    """SMS notification provider."""
    
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
    
    def get_channel(self) -> NotificationChannel:
        return NotificationChannel.SMS
    
    def send(self, notification: Notification) -> FlextResult[None]:
        """Send SMS notification."""
        try:
            # Simulate SMS sending
            print(f"📱 SMS SENT to {notification.recipient}")
            print(f"   Message: {notification.body}")
            
            notification.sent_at = datetime.now()
            return FlextResult.ok(None)
            
        except Exception as e:
            notification.failed_at = datetime.now()
            return FlextResult.fail(f"SMS send failed: {str(e)}")

class PushProvider(NotificationProvider):
    """Push notification provider."""
    
    def __init__(self, push_service_config: Dict[str, Any]):
        self.push_service_config = push_service_config
    
    def get_channel(self) -> NotificationChannel:
        return NotificationChannel.PUSH
    
    def send(self, notification: Notification) -> FlextResult[None]:
        """Send push notification."""
        try:
            # Simulate push notification
            print(f"🔔 PUSH SENT to {notification.recipient}")
            print(f"   Title: {notification.subject}")
            print(f"   Body: {notification.body}")
            
            notification.sent_at = datetime.now()
            return FlextResult.ok(None)
            
        except Exception as e:
            notification.failed_at = datetime.now()
            return FlextResult.fail(f"Push send failed: {str(e)}")

# ========================
# NOTIFICATION SERVICE
# ========================

class NotificationService:
    """Central notification service."""
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.pending_notifications: List[Notification] = []
    
    def register_provider(self, provider: NotificationProvider) -> None:
        """Register a notification provider."""
        self.providers[provider.get_channel()] = provider
    
    def register_template(self, template: NotificationTemplate) -> None:
        """Register a notification template."""
        self.templates[template.template_id] = template
    
    def send_notification(
        self,
        recipient: str,
        template_id: str,
        data: Dict[str, Any],
        channel: NotificationChannel,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> FlextResult[Notification]:
        """Send notification using template."""
        # Get template
        if template_id not in self.templates:
            return FlextResult.fail(f"Template not found: {template_id}")
        
        template = self.templates[template_id]
        
        # Render template
        try:
            rendered = template.render(data)
        except ValueError as e:
            return FlextResult.fail(f"Template render failed: {str(e)}")
        
        # Create notification
        notification = Notification(
            recipient=recipient,
            channel=channel,
            subject=rendered["subject"],
            body=rendered["body"],
            priority=priority,
            metadata={"template_id": template_id, "data": data}
        )
        
        # Send notification
        return self._send_notification(notification)
    
    def _send_notification(self, notification: Notification) -> FlextResult[Notification]:
        """Internal method to send notification."""
        # Get provider
        if notification.channel not in self.providers:
            return FlextResult.fail(f"No provider for channel: {notification.channel}")
        
        provider = self.providers[notification.channel]
        
        # Attempt to send
        send_result = provider.send(notification)
        
        if send_result.is_success:
            return FlextResult.ok(notification)
        
        # Handle failure
        notification.retry_count += 1
        
        if notification.retry_count <= notification.max_retries:
            # Queue for retry
            self.pending_notifications.append(notification)
            return FlextResult.fail(f"Send failed, queued for retry: {send_result.error}")
        
        return FlextResult.fail(f"Send failed after {notification.max_retries} retries: {send_result.error}")
    
    def process_pending_notifications(self) -> int:
        """Process pending notifications (retry failed ones)."""
        processed = 0
        remaining = []
        
        for notification in self.pending_notifications:
            if notification.retry_count <= notification.max_retries:
                result = self._send_notification(notification)
                if result.is_success:
                    processed += 1
                else:
                    remaining.append(notification)
            
        self.pending_notifications = remaining
        return processed

# ========================
# EVENT HANDLERS
# ========================

class UserRegistrationNotificationHandler(FlextEventHandler[UserRegisteredEvent]):
    """Handle user registration notifications."""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__()
        self.notification_service = notification_service
    
    def get_event_type(self):
        return "user.registered"
    
    def handle_event(self, event: UserRegisteredEvent) -> FlextResult[None]:
        """Send welcome email to new user."""
        print(f"📧 Processing user registration notification for user {event.data['user_id']}")
        
        # Send welcome email
        email_result = self.notification_service.send_notification(
            recipient=event.data["email"],
            template_id="welcome_email",
            data={
                "name": event.data["name"],
                "user_id": event.data["user_id"]
            },
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.NORMAL
        )
        
        if email_result.is_failure:
            return FlextResult.fail(f"Welcome email failed: {email_result.error}")
        
        # Send SMS notification if phone available
        # (In real system, would check user preferences)
        
        return FlextResult.ok(None)

class OrderConfirmationNotificationHandler(FlextEventHandler[OrderConfirmedEvent]):
    """Handle order confirmation notifications."""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__()
        self.notification_service = notification_service
    
    def get_event_type(self):
        return "order.confirmed"
    
    def handle_event(self, event: OrderConfirmedEvent) -> FlextResult[None]:
        """Send order confirmation notifications."""
        print(f"📋 Processing order confirmation notification for order {event.data['order_id']}")
        
        # In real system, would get customer email from customer service
        customer_email = f"customer_{event.data['customer_id']}@test.com"
        
        # Send email confirmation
        email_result = self.notification_service.send_notification(
            recipient=customer_email,
            template_id="order_confirmation",
            data={
                "order_id": event.data["order_id"],
                "total_amount": event.data["total_amount"]
            },
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.HIGH
        )
        
        if email_result.is_failure:
            return FlextResult.fail(f"Order confirmation email failed: {email_result.error}")
        
        # Send push notification
        push_result = self.notification_service.send_notification(
            recipient=customer_email,  # Would be device token in real system
            template_id="order_confirmation_push",
            data={
                "order_id": event.data["order_id"]
            },
            channel=NotificationChannel.PUSH,
            priority=NotificationPriority.HIGH
        )
        
        if push_result.is_failure:
            print(f"⚠️ Push notification failed: {push_result.error}")
            # Don't fail the whole process for push failure
        
        return FlextResult.ok(None)

class PaymentFailedNotificationHandler(FlextEventHandler[PaymentFailedEvent]):
    """Handle payment failure notifications."""
    
    def __init__(self, notification_service: NotificationService):
        super().__init__()
        self.notification_service = notification_service
    
    def get_event_type(self):
        return "payment.failed"
    
    def handle_event(self, event: PaymentFailedEvent) -> FlextResult[None]:
        """Send urgent payment failure notifications."""
        print(f"💳 Processing payment failure notification for payment {event.data['payment_id']}")
        
        # In real system, would get user contact info
        user_email = f"user_{event.data['user_id']}@test.com"
        
        # Send urgent email
        email_result = self.notification_service.send_notification(
            recipient=user_email,
            template_id="payment_failed",
            data={
                "payment_id": event.data["payment_id"],
                "amount": event.data["amount"],
                "reason": event.data["reason"]
            },
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.URGENT
        )
        
        if email_result.is_failure:
            return FlextResult.fail(f"Payment failure email failed: {email_result.error}")
        
        return FlextResult.ok(None)

# ========================
# SETUP AND DEMO
# ========================

def setup_notification_system() -> tuple[NotificationService, FlextHandlerRegistry]:
    """Setup complete notification system."""
    # Create notification service
    notification_service = NotificationService()
    
    # Register providers
    email_provider = EmailProvider({"smtp_server": "localhost", "port": 587})
    sms_provider = SMSProvider("fake_api_key", "https://api.sms.com")
    push_provider = PushProvider({"service": "firebase", "key": "fake_key"})
    
    notification_service.register_provider(email_provider)
    notification_service.register_provider(sms_provider)
    notification_service.register_provider(push_provider)
    
    # Register templates
    welcome_template = NotificationTemplate(
        "welcome_email",
        "Welcome to our platform, {name}!",
        "Hi {name},\n\nWelcome to our platform! Your user ID is: {user_id}\n\nBest regards,\nThe Team",
        NotificationChannel.EMAIL
    )
    
    order_confirmation_template = NotificationTemplate(
        "order_confirmation",
        "Order Confirmed - {order_id}",
        "Your order {order_id} has been confirmed.\nTotal amount: ${total_amount}\n\nThank you for your purchase!",
        NotificationChannel.EMAIL
    )
    
    order_confirmation_push_template = NotificationTemplate(
        "order_confirmation_push",
        "Order Confirmed",
        "Your order {order_id} has been confirmed!",
        NotificationChannel.PUSH
    )
    
    payment_failed_template = NotificationTemplate(
        "payment_failed",
        "Payment Failed - Action Required",
        "Payment {payment_id} for ${amount} failed.\nReason: {reason}\n\nPlease update your payment method.",
        NotificationChannel.EMAIL
    )
    
    notification_service.register_template(welcome_template)
    notification_service.register_template(order_confirmation_template)
    notification_service.register_template(order_confirmation_push_template)
    notification_service.register_template(payment_failed_template)
    
    # Create event handlers
    handler_registry = FlextHandlerRegistry()
    
    user_handler = UserRegistrationNotificationHandler(notification_service)
    order_handler = OrderConfirmationNotificationHandler(notification_service)
    payment_handler = PaymentFailedNotificationHandler(notification_service)
    
    handler_registry.register(user_handler)
    handler_registry.register(order_handler)
    handler_registry.register(payment_handler)
    
    return notification_service, handler_registry

def demo_notification_system():
    """Demonstrate notification system with events."""
    print("📬 Notification System Demo")
    print("=" * 50)
    
    # Setup system
    notification_service, handler_registry = setup_notification_system()
    
    # Simulate events
    events = [
        UserRegisteredEvent("user_123", "alice@test.com", "Alice Johnson"),
        OrderConfirmedEvent("order_456", "customer_789", 299.99),
        PaymentFailedEvent("payment_101", "user_456", 150.00, "Insufficient funds")
    ]
    
    print(f"\n🎭 Processing {len(events)} domain events...")
    
    for event in events:
        print(f"\n📡 Event: {event.event_type}")
        print(f"   Event ID: {event.event_id}")
        print(f"   Timestamp: {event.timestamp}")
        
        # Find handlers for this event
        handlers = handler_registry.find_handlers(event)
        
        if not handlers:
            print(f"   ⚠️ No handlers found for event type: {event.event_type}")
            continue
        
        print(f"   📝 Found {len(handlers)} handler(s)")
        
        # Process with each handler
        for handler in handlers:
            if hasattr(handler, 'process_event'):
                result = handler.process_event(event)
                if result.is_success:
                    print(f"   ✅ Handler processed successfully")
                else:
                    print(f"   ❌ Handler failed: {result.error}")
    
    # Process any pending notifications
    print(f"\n🔄 Processing pending notifications...")
    pending_count = len(notification_service.pending_notifications)
    if pending_count > 0:
        processed = notification_service.process_pending_notifications()
        print(f"   📤 Processed {processed} out of {pending_count} pending notifications")
    else:
        print(f"   ✅ No pending notifications")
    
    print("\n" + "=" * 50)
    print("🎉 Notification system demo completed!")

if __name__ == "__main__":
    demo_notification_system()
```

---

## 🔗 Links para Mais Exemplos

### Exemplos Básicos

- **[CRUD Simples](basic/crud-example.md)** - Operações básicas com entidades
- **[Validation](basic/validation-example.md)** - Sistema de validação completo
- **[Container DI](basic/container-example.md)** - Dependency injection na prática

### Exemplos Avançados

- **[Microserviços](advanced/microservices-example.md)** - Arquitetura distribuída
- **[Event Sourcing](advanced/event-sourcing-example.md)** - Padrão de eventos
- **[CQRS](advanced/cqrs-example.md)** - Command Query Responsibility Segregation

### Integrações

- **[FastAPI](integrations/fastapi-example.md)** - API REST com FastAPI
- **[Django](integrations/django-example.md)** - Integração com Django
- **[Celery](integrations/celery-example.md)** - Tasks assíncronas

### Casos de Uso Reais

- **[Sistema Bancário](real-world/banking-system.md)** - Sistema financeiro completo
- **[ERP](real-world/erp-system.md)** - Sistema de gestão empresarial
- **[CRM](real-world/crm-system.md)** - Customer Relationship Management

---

## 🎯 Como Usar os Exemplos

### 1. Executar Localmente

```bash
# Clonar repositório
git clone https://github.com/flext/flext-core.git
cd flext-core

# Instalar dependências
poetry install

# Executar exemplo específico
poetry run python docs/examples/user_system_example.py
poetry run python docs/examples/ecommerce_example.py
poetry run python docs/examples/notification_example.py
```

### 2. Adaptar para Seu Projeto

```python
# 1. Copie as classes de domínio
# 2. Adapte para suas regras de negócio
# 3. Implemente repositories para sua base de dados
# 4. Configure seu container de DI
# 5. Adicione seus próprios handlers
```

### 3. Executar Testes

```bash
# Testar exemplos
poetry run pytest docs/examples/tests/

# Testar com coverage
poetry run pytest docs/examples/tests/ --cov=docs/examples
```

---

## 💡 Dicas para Implementação

### ✅ Práticas Recomendadas

1. **Comece Simples**: Use exemplos básicos como base
2. **Adapte Gradualmente**: Modifique para suas necessidades específicas
3. **Teste Primeiro**: Escreva testes antes de implementar
4. **Use Type Hints**: Mantenha type safety em todo código
5. **Documente Decisões**: Explique regras de negócio no código

### 🚫 Armadilhas Comuns

1. **Não copie cegamente**: Entenda antes de usar
2. **Não ignore validação**: Sempre valide inputs
3. **Não negligencie erros**: Use FlextResult consistentemente
4. **Não misture responsabilidades**: Mantenha separação de camadas
5. **Não esqueça testes**: Cobertura mínima de 90%

---

**Os exemplos são sua melhor ferramenta para aprender FLEXT Core na prática!** 🚀
