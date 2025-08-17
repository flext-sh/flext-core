from enum import StrEnum

from _typeshed import Incomplete

from flext_core import FlextEntity, FlextResult, FlextSettings, FlextValueObject

from .shared_domain import Age as Age, Money as Money, User as SharedUser

MIN_PRODUCT_NAME_LENGTH: int
MAX_ORDER_ITEM_QUANTITY: int
MAX_ORDER_ITEMS: int

class AppConfig(FlextSettings):
    database_url: str
    redis_url: str
    payment_api_key: str
    payment_endpoint: str
    max_order_value: int
    min_order_value: int
    model_config: Incomplete

class OrderStatus(StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Customer(SharedUser):
    is_premium: bool
    def promote_to_premium(self) -> FlextResult[Customer]: ...

class Product(FlextEntity):
    name: str
    price: Money
    stock_quantity: int
    category: str
    def is_available(self, quantity: int = 1) -> bool: ...
    def reserve_stock(self, quantity: int) -> FlextResult[None]: ...
    def validate_business_rules(self) -> FlextResult[None]: ...

class OrderItem(FlextValueObject):
    product_id: str
    quantity: int
    unit_price: Money
    def validate_business_rules(self) -> FlextResult[None]: ...
    def total_price(self) -> Money: ...

class Order(FlextEntity):
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus
    total: Money
    def add_item(self, product: Product, quantity: int) -> FlextResult[Order]: ...
    def confirm(self) -> FlextResult[Order]: ...
    def validate_business_rules(self) -> FlextResult[None]: ...

class InventoryService:
    def reserve_items(self, items: list[OrderItem]) -> FlextResult[None]: ...

class PaymentService:
    def charge(self, customer_id: str, amount: Money) -> FlextResult[str]: ...

class NotificationService:
    def send_order_confirmation(
        self, customer: Customer, order: Order
    ) -> FlextResult[None]: ...

def create_customer(name: str, email_address: str) -> FlextResult[Customer]: ...
def create_product(
    name: str, price_cents: int, stock: int, category: str
) -> FlextResult[Product]: ...

class OrderProcessingService:
    inventory: Incomplete
    payment: Incomplete
    notifications: Incomplete
    def __init__(self) -> None: ...
    def process_order(
        self, customer_id: str, order_data: dict[str, object]
    ) -> FlextResult[Order]: ...

def demonstrate_modern_patterns() -> None: ...
def demonstrate_boilerplate_comparison() -> None: ...
