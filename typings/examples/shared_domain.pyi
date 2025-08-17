from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from _typeshed import Incomplete

from flext_core import FlextEntity, FlextResult, FlextValueObject, TEntityId

__all__ = [
    "CURRENCY_CODE_LENGTH",
    "MAX_AGE",
    "MAX_EMAIL_LENGTH",
    "MAX_NAME_LENGTH",
    "MIN_AGE",
    "MIN_NAME_LENGTH",
    "SUPPORTED_CURRENCIES",
    "Address",
    "Age",
    "ComplexValueObject",
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "EmailAddress",
    "Money",
    "Order",
    "OrderItem",
    "OrderStatus",
    "PaymentMethod",
    "PhoneNumber",
    "Product",
    "SharedDomainFactory",
    "TestDomainFactory",
    "User",
    "UserStatus",
    "log_domain_operation",
]

MIN_AGE: int
MAX_AGE: int
MAX_EMAIL_LENGTH: int
MIN_NAME_LENGTH: int
MAX_NAME_LENGTH: int
CURRENCY_CODE_LENGTH: int
SUPPORTED_CURRENCIES: Incomplete

class UserStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class OrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class PaymentMethod(StrEnum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"

class EmailAddress(FlextValueObject):
    email: str
    def validate_business_rules(self) -> FlextResult[None]: ...

class Age(FlextValueObject):
    value: int
    def validate_business_rules(self) -> FlextResult[None]: ...

class Money(FlextValueObject):
    amount: Decimal
    currency: str
    def validate_business_rules(self) -> FlextResult[None]: ...
    def add(self, other: Money) -> FlextResult[Money]: ...
    def multiply(self, factor: Decimal) -> FlextResult[Money]: ...

class Address(FlextValueObject):
    street: str
    city: str
    postal_code: str
    country: str
    state: str | None
    def validate_business_rules(self) -> FlextResult[None]: ...
    def is_same_city(self, other: Address) -> bool: ...

class PhoneNumber(FlextValueObject):
    number: str
    country_code: str | None
    def validate_business_rules(self) -> FlextResult[None]: ...

class User(FlextEntity):
    name: str
    email_address: EmailAddress
    age: Age
    status: UserStatus
    phone: PhoneNumber | None
    address: Address | None
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def activate(self) -> FlextResult[User]: ...
    def suspend(self, reason: str) -> FlextResult[User]: ...
    def copy_with(self, **kwargs: object) -> FlextResult[User]: ...

class Product(FlextEntity):
    name: str
    description: str
    price: Money
    category: str
    in_stock: bool
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def update_price(self, new_price: Money) -> FlextResult[Product]: ...

class OrderItem(FlextValueObject):
    product_id: TEntityId
    product_name: str
    quantity: int
    unit_price: Money
    def validate_business_rules(self) -> FlextResult[None]: ...
    def total_price(self) -> FlextResult[Money]: ...

class Order(FlextEntity):
    customer_id: TEntityId
    items: list[OrderItem]
    status: OrderStatus
    payment_method: PaymentMethod | None
    shipping_address: Address | None
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def calculate_total(self) -> FlextResult[Money]: ...
    def confirm(self) -> FlextResult[Order]: ...

class SharedDomainFactory:
    @staticmethod
    def create_user(
        name: str, email: str, age: int, **kwargs: object
    ) -> FlextResult[User]: ...
    @staticmethod
    def create_product(
        name: str,
        description: str,
        price_amount: str | Decimal,
        currency: str = "USD",
        **kwargs: object,
    ) -> FlextResult[Product]: ...
    @staticmethod
    def create_order(
        customer_id: TEntityId, items: list[dict[str, object]], **kwargs: object
    ) -> FlextResult[Order]: ...

def log_domain_operation(
    operation: str, entity_type: str, entity_id: str, **context: object
) -> None: ...

class ConcreteFlextEntity(FlextEntity):
    name: str
    status: str
    created_at: datetime
    def validate_domain_rules(self) -> FlextResult[None]: ...

class ConcreteValueObject(FlextValueObject):
    amount: Decimal
    currency: str
    description: str
    def validate_business_rules(self) -> FlextResult[None]: ...

class ComplexValueObject(FlextValueObject):
    name: str
    tags: list[str]
    metadata: dict[str, object]
    def validate_business_rules(self) -> FlextResult[None]: ...

class TestDomainFactory:
    @staticmethod
    def create_concrete_entity(
        name: str, status: str = "active", **kwargs: object
    ) -> FlextResult[ConcreteFlextEntity]: ...
    @staticmethod
    def create_concrete_value_object(
        amount: Decimal, currency: str = "USD", **kwargs: object
    ) -> FlextResult[ConcreteValueObject]: ...
    @staticmethod
    def create_complex_value_object(
        name: str, tags: list[str], metadata: dict[str, object]
    ) -> FlextResult[ComplexValueObject]: ...

class SharedDemonstrationPattern:
    @staticmethod
    def run_demonstration(
        title: str, demonstration_functions: list[Callable[[], None]]
    ) -> None: ...
