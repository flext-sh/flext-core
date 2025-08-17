from decimal import Decimal
from enum import StrEnum

from flext_core import FlextEntity, FlextResult, FlextValueObject, TEntityId

__all__ = [
    "TestAddress",
    "TestComplexValueObject",
    "TestDomainFactory",
    "TestEmailAddress",
    "TestMoney",
    "TestOrder",
    "TestOrderStatus",
    "TestProduct",
    "TestUser",
    "TestUserStatus",
    "create_test_order_safe",
    "create_test_product_safe",
    "create_test_user_safe",
    "log_test_operation",
]

class TestUserStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class TestOrderStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class TestEmailAddress(FlextValueObject):
    email: str
    def validate_business_rules(self) -> FlextResult[None]: ...

class TestMoney(FlextValueObject):
    amount: Decimal
    currency: str
    description: str
    def validate_business_rules(self) -> FlextResult[None]: ...

class TestAddress(FlextValueObject):
    street: str
    city: str
    state: str
    zip_code: str
    country: str
    def validate_business_rules(self) -> FlextResult[None]: ...

class TestComplexValueObject(FlextValueObject):
    name: str
    tags: list[str]
    metadata: dict[str, object]
    def validate_business_rules(self) -> FlextResult[None]: ...

class TestUser(FlextEntity):
    name: str
    email: str
    age: int
    status: TestUserStatus
    balance: Decimal
    @classmethod
    def validate_name(cls, v: str) -> str: ...
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def activate(self) -> FlextResult[None]: ...
    def deactivate(self) -> FlextResult[None]: ...

class TestOrder(FlextEntity):
    user_id: TEntityId
    total_amount: Decimal
    status: TestOrderStatus
    items: list[str]
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def add_item(self, item: str) -> FlextResult[None]: ...
    def confirm(self) -> FlextResult[None]: ...

class TestProduct(FlextEntity):
    name: str
    price: Decimal
    description: str
    in_stock: bool
    category: str
    def validate_domain_rules(self) -> FlextResult[None]: ...
    def update_stock(self, *, _in_stock: bool) -> FlextResult[None]: ...

class TestDomainFactory:
    @classmethod
    def create_test_user(
        cls, name: str = "Test User", email: str = "test@example.com", **kwargs: object
    ) -> FlextResult[TestUser]: ...
    @classmethod
    def create_test_order(
        cls,
        user_id: TEntityId,
        total_amount: str | Decimal = "100.00",
        **kwargs: object,
    ) -> FlextResult[TestOrder]: ...
    @classmethod
    def create_test_product(
        cls,
        name: str = "Test Product",
        price: str | Decimal = "50.00",
        **kwargs: object,
    ) -> FlextResult[TestProduct]: ...
    @classmethod
    def create_test_email(
        cls, email: str = "test@example.com"
    ) -> FlextResult[TestEmailAddress]: ...
    @classmethod
    def create_test_money(
        cls,
        amount: str | Decimal = "100.00",
        currency: str = "USD",
        description: str = "test money",
    ) -> FlextResult[TestMoney]: ...
    @classmethod
    def create_concrete_entity(
        cls, name: str = "Test User", **kwargs: object
    ) -> FlextResult[TestUser]: ...
    @classmethod
    def create_concrete_value_object(
        cls, **kwargs: object
    ) -> FlextResult[TestMoney]: ...
    @classmethod
    def create_complex_value_object(
        cls, name: str, tags: list[str], metadata: dict[str, object]
    ) -> FlextResult[TestComplexValueObject]: ...

def create_test_user_safe(
    name: str = "Test User", email: str = "test@example.com", **kwargs: object
) -> TestUser: ...
def create_test_order_safe(
    user_id: TEntityId, total_amount: str | Decimal = "100.00", **kwargs: object
) -> TestOrder: ...
def create_test_product_safe(
    name: str = "Test Product", **kwargs: object
) -> TestProduct: ...
def log_test_operation(operation: str, entity_type: str, entity_id: str) -> None: ...
