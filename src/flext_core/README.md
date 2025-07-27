# FLEXT Core Domain Layer

Domain-Driven Design components for building rich business models.

## Overview

This module provides the foundation for implementing Domain-Driven Design patterns including entities, value objects, aggregates, and domain services.

## Components

### Entity (`entity.py`)

- `FlextEntity` - Base class for domain entities with identity
- Identity-based equality and lifecycle management
- Domain event support for change tracking

### Value Object (`value_object.py`)

- `FlextValueObject` - Base class for immutable value objects
- Value-based equality and validation
- Type-safe immutable data structures

### Aggregate Root (`aggregate_root.py`)

- `FlextAggregateRoot` - Root entity for consistency boundaries
- Domain event collection and publishing
- Transaction boundary enforcement

### Domain Service (`domain_service.py`)

- `FlextDomainService` - Base class for domain services
- Stateless operations across multiple entities
- Business logic that doesn't belong to a single entity

## Usage Examples

### Entity Definition

```python
from flext_core.entities import FlextEntity
from pydantic import Field

class User(FlextEntity):
    name: str = Field(..., description="User full name")
    email: str = Field(..., description="User email address")
    is_active: bool = Field(True, description="User active status")
    
    def activate(self) -> FlextResult[None]:
        """Activate the user account."""
        if self.is_active:
            return FlextResult.fail("User is already active")
        
        self.is_active = True
        self.add_domain_event(UserActivatedEvent(user_id=self.id))
        return FlextResult.ok(None)
    
    def deactivate(self) -> FlextResult[None]:
        """Deactivate the user account."""
        if not self.is_active:
            return FlextResult.fail("User is already inactive")
        
        self.is_active = False
        self.add_domain_event(UserDeactivatedEvent(user_id=self.id))
        return FlextResult.ok(None)
```

### Value Object Definition

```python
from flext_core.value_objects import FlextValueObject
from pydantic import Field, field_validator

class Email(FlextValueObject):
    address: str = Field(..., description="Email address")
    
    @field_validator("address")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Validate email format."""
        if "@" not in v or "." not in v.split("@")[1]:
            raise ValueError("Invalid email format")
        return v.lower()

class Money(FlextValueObject):
    amount: int = Field(..., description="Amount in cents")
    currency: str = Field(..., description="Currency code")
    
    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        if len(v) != 3 or not v.isupper():
            raise ValueError("Currency must be 3-letter uppercase code")
        return v
    
    def add(self, other: "Money") -> "Money":
        """Add two money amounts."""
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)
```

### Aggregate Root Definition

```python
from flext_core.aggregate_root import FlextAggregateRoot
from datetime import datetime

class Order(FlextAggregateRoot):
    customer_id: str = Field(..., description="Customer identifier")
    items: list[OrderItem] = Field(default_factory=list, description="Order items")
    status: OrderStatus = Field(OrderStatus.PENDING, description="Order status")
    total: Money = Field(..., description="Order total")
    
    def add_item(self, product_id: str, quantity: int, price: Money) -> FlextResult[None]:
        """Add item to the order."""
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")
        
        item = OrderItem(product_id=product_id, quantity=quantity, price=price)
        self.items.append(item)
        self._recalculate_total()
        
        self.add_domain_event(OrderItemAddedEvent(
            order_id=self.id,
            product_id=product_id,
            quantity=quantity
        ))
        
        return FlextResult.ok(None)
    
    def confirm(self) -> FlextResult[None]:
        """Confirm the order."""
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Order is not in pending status")
        
        if not self.items:
            return FlextResult.fail("Cannot confirm empty order")
        
        self.status = OrderStatus.CONFIRMED
        self.add_domain_event(OrderConfirmedEvent(
            order_id=self.id,
            total=self.total,
            confirmed_at=datetime.utcnow()
        ))
        
        return FlextResult.ok(None)
```

### Domain Service Definition

```python
from flext_core.domain_services import FlextDomainService

class OrderPricingService(FlextDomainService):
    """Domain service for order pricing calculations."""
    
    def calculate_order_total(self, items: list[OrderItem], customer: Customer) -> Money:
        """Calculate total order amount with discounts."""
        subtotal = sum(item.total for item in items)
        discount = self._calculate_customer_discount(customer, subtotal)
        return subtotal - discount
    
    def _calculate_customer_discount(self, customer: Customer, subtotal: Money) -> Money:
        """Calculate customer-specific discount."""
        if customer.is_premium:
            discount_rate = 0.1  # 10% discount for premium customers
            discount_amount = int(subtotal.amount * discount_rate)
            return Money(amount=discount_amount, currency=subtotal.currency)
        return Money(amount=0, currency=subtotal.currency)
```

## Design Principles

### Entity Characteristics

- **Identity**: Entities have unique identifiers that persist through changes
- **Mutability**: Entities can change state while maintaining identity
- **Lifecycle**: Entities have creation, modification, and deletion phases
- **Equality**: Based on identity, not attribute values

### Value Object Characteristics

- **Immutability**: Value objects cannot be modified after creation
- **Value Equality**: Two value objects are equal if all attributes match
- **Side-Effect Free**: Operations return new instances rather than modifying existing ones
- **Validation**: Enforce business rules and invariants at creation time

### Aggregate Characteristics

- **Consistency Boundary**: Aggregates enforce business rules across related entities
- **Transaction Boundary**: Changes to an aggregate happen within a single transaction
- **Root Access**: External objects reference only the aggregate root
- **Event Publishing**: Domain events communicate changes to other bounded contexts

### Domain Service Characteristics

- **Stateless**: Domain services don't maintain state between operations
- **Business Logic**: Contain business logic that doesn't belong to entities or value objects
- **Coordination**: Orchestrate operations across multiple domain objects
- **Pure Functions**: Operations should be deterministic and side-effect free

## Integration with FLEXT Core

- **FlextResult**: All domain operations return FlextResult for error handling
- **FlextContainer**: Domain services can be registered as dependencies
- **Event System**: Domain events integrate with application event handlers
- **Validation**: Pydantic validation ensures data integrity

For detailed usage examples, see the [core API documentation](../../docs/api/core.md).

## FLEXT Core Patterns

Enterprise patterns for building robust, maintainable applications.

## Overview

This module provides implementation of common enterprise patterns including Command/Query Responsibility Segregation (CQRS), message handling, and validation patterns.

## Components

### Commands (`commands.py`)

- `FlextCommand` - Base class for command objects
- `FlextCommandHandler` - Command processing interface
- Command validation and execution patterns

### Handlers (`handlers.py`)

- `FlextHandler` - Base handler with metadata support
- `FlextMessageHandler` - Generic message processing
- `FlextEventHandler` - Domain event processing  
- `FlextRequestHandler` - Request/response handling
- `FlextHandlerRegistry` - Central handler management

### Validation (`validation.py`)

- `FlextValidator` - Data validation interface
- `FlextBusinessRule` - Business rule specifications
- `FlextValidationPipeline` - Orchestrated validation

### Logging (`logging.py`)

- Structured logging patterns
- Context-aware log handling
- Performance monitoring integration

### Type Definitions (`typedefs.py`)

- Type aliases for pattern components
- NewType definitions for type safety
- Generic type variables

### Fields (`fields.py`)

- Custom Pydantic field types
- Validation helpers
- Type-safe field definitions

## Usage Examples

### Command Pattern

```python
from flext_core.patterns import FlextCommand, FlextCommandHandler

class CreateUserCommand(FlextCommand):
    name: str
    email: str

class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Implementation
        return FlextResult.ok(user)
```

### Event Handling

```python
from flext_core.patterns import FlextEventHandler

class UserCreatedHandler(FlextEventHandler[UserCreatedEvent]):
    def handle_event(self, event: UserCreatedEvent) -> FlextResult[None]:
        # Handle the event
        return FlextResult.ok(None)
    
    def get_event_type(self) -> str:
        return "user_created"
```

### Validation

```python
from flext_core.patterns import FlextValidator, FlextBusinessRule

class EmailValidator(FlextValidator[str]):
    def validate(self, email: str) -> FlextResult[str]:
        if "@" not in email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(email)

class AgeRule(FlextBusinessRule[User]):
    def is_satisfied_by(self, user: User) -> bool:
        return user.age >= 18
```

## Design Principles

- **Type Safety**: All patterns use Python 3.13 generics for type safety
- **Error Handling**: Consistent FlextResult usage throughout
- **Composability**: Patterns can be combined and extended
- **Testability**: Clear interfaces enable easy unit testing
- **Performance**: Minimal overhead with efficient implementations

## Integration

These patterns integrate seamlessly with other FLEXT Core components:

- **FlextResult**: All operations return FlextResult for error handling
- **FlextContainer**: Handlers can be registered as services
- **Domain Layer**: Events and commands work with domain entities
- **Configuration**: Validation integrates with settings management

For detailed usage examples, see the [patterns documentation](../docs/api/patterns.md).
