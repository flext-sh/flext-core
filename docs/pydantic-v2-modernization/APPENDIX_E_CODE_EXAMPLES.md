# Appendix E: Complete Code Examples

**Status**: WORKING CODE SAMPLES
**Purpose**: Copy-paste ready examples for common patterns
**Usage**: Reference when implementing migration

---

## Table of Contents

1. [Basic Model Migration](#basic-model-migration)
2. [Validators](#validators)
3. [Serialization](#serialization)
4. [Reusable Types](#reusable-types)
5. [Advanced Patterns](#advanced-patterns)

---

## Basic Model Migration

### Before (Pydantic v1)

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str
    age: int
    email: str

    class Config:
        frozen = True

    @validator("age")
    def validate_age(cls, value):
        if value < 0 or value > 150:
            raise ValueError("Invalid age")
        return value
```

### After (Pydantic v2)

```python
from pydantic import BaseModel, ConfigDict, field_validator

class User(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    age: int
    email: str

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 0 or value > 150:
            raise ValueError("Invalid age")
        return value
```

---

### Migration Checklist

```python
# ✅ Always do:
- [ ] Replace `class Config:` with `model_config = ConfigDict(...)`
- [ ] Add `@classmethod` to validators
- [ ] Add type hints to validator parameters and returns
- [ ] Replace `@validator` with `@field_validator`
- [ ] Replace `.dict()` with `.model_dump()`
- [ ] Replace `.json()` with `.model_dump_json()`
```

---

## Validators

### Field Validators

```python
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    name: str
    price: float
    quantity: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate single field."""
        if not value.strip():
            raise ValueError("Name cannot be empty")
        return value.strip()

    @field_validator("price", "quantity")
    @classmethod
    def validate_positive(cls, value: float | int) -> float | int:
        """Validate multiple fields with same rule."""
        if value <= 0:
            raise ValueError("Must be positive")
        return value
```

---

### Mode: Before vs After

```python
from pydantic import BaseModel, field_validator

class Config(BaseModel):
    port: int

    @field_validator("port", mode="before")
    @classmethod
    def parse_port(cls, value: str | int) -> int:
        """Run before type coercion - receives raw input."""
        return int(value)  # Convert string "8080" to int 8080

    @field_validator("port", mode="after")
    @classmethod
    def validate_port_range(cls, value: int) -> int:
        """Run after type coercion - receives validated int."""
        if not (0 <= value <= 65535):
            raise ValueError("Port out of range")
        return value
```

---

### Model Validator (Cross-Field)

```python
from pydantic import BaseModel, model_validator
from datetime import date

class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        """Validate relationship between fields."""
        if self.end < self.start:
            raise ValueError("End date must be after start date")
        return self
```

---

## Serialization

### Custom Serialization

```python
from pydantic import BaseModel, field_serializer
from decimal import Decimal
import json

class Invoice(BaseModel):
    amount: Decimal
    tax: Decimal
    created_at: str

    @field_serializer("amount", "tax")
    def serialize_currency(self, value: Decimal) -> str:
        """Serialize currency fields as formatted strings."""
        return f"${value:.2f}"

    @field_serializer("created_at")
    def serialize_date(self, value: str) -> str:
        """Serialize date in specific format."""
        return value.upper()  # Just example

# Usage
invoice = Invoice(amount=Decimal("100.50"), tax=Decimal("15.07"), created_at="2025-01-21")
print(invoice.model_dump())
# {'amount': '$100.50', 'tax': '$15.07', 'created_at': '2025-01-21'}
```

---

### Computed Fields

```python
from pydantic import BaseModel, computed_field

class Order(BaseModel):
    items: list[float]
    tax_rate: float = 0.1

    @computed_field
    @property
    def subtotal(self) -> float:
        """Automatically calculated subtotal."""
        return sum(self.items)

    @computed_field
    @property
    def tax(self) -> float:
        """Automatically calculated tax."""
        return self.subtotal * self.tax_rate

    @computed_field
    @property
    def total(self) -> float:
        """Automatically calculated total."""
        return self.subtotal + self.tax

# Usage
order = Order(items=[10.0, 20.0, 30.0])
print(order.model_dump())
# {'items': [10.0, 20.0, 30.0], 'tax_rate': 0.1, 'subtotal': 60.0, 'tax': 6.0, 'total': 66.0}
```

---

### Mode-Specific Serialization

```python
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    name: str
    created_at: datetime

# Different serialization modes
user = User(name="Alice", created_at=datetime(2025, 1, 21))

# Python types
print(user.model_dump(mode="python"))
# {'name': 'Alice', 'created_at': datetime.datetime(2025, 1, 21, ...)}

# JSON types (datetime as string)
print(user.model_dump(mode="json"))
# {'name': 'Alice', 'created_at': '2025-01-21T00:00:00'}

# Direct JSON string
print(user.model_dump_json())
# {"name":"Alice","created_at":"2025-01-21T00:00:00"}
```

---

## Reusable Types

### Creating Custom Types

```python
from typing import Annotated
from pydantic import BaseModel, Field, AfterValidator, BeforeValidator

# Simple Field constraints
PortNumber = Annotated[int, Field(ge=1, le=65535, description="Network port")]  # MIN_PORT=1
NonEmptyStr = Annotated[str, Field(min_length=1)]
Percentage = Annotated[float, Field(ge=0, le=100)]

# With custom validators
def validate_dns(value: str) -> str:
    """Validate hostname is resolvable."""
    import socket
    try:
        socket.gethostbyname(value)
        return value
    except socket.gaierror:
        raise ValueError(f"Cannot resolve: {value}")

HostName = Annotated[str, AfterValidator(validate_dns)]

# Complex type
def parse_csv(value: str | list) -> list[str]:
    """Parse CSV string to list."""
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",")]

CSVList = Annotated[list[str], BeforeValidator(parse_csv)]

# Usage
class Config(BaseModel):
    port: PortNumber
    host: HostName
    name: NonEmptyStr
    threshold: Percentage
    tags: CSVList

# Valid
config = Config(
    port=8080,
    host="localhost",
    name="MyApp",
    threshold=75.5,
    tags="tag1, tag2, tag3"
)
print(config.tags)
# ['tag1', 'tag2', 'tag3']
```

---

### Reusable Validator Functions

```python
from typing import Annotated
from pydantic import AfterValidator, BeforeValidator

def validate_email(value: str) -> str:
    """Validate email format."""
    if "@" not in value or "." not in value.split("@")[1]:
        raise ValueError("Invalid email")
    return value.lower()

def validate_phone(value: str) -> str:
    """Validate phone format."""
    digits = ''.join(c for c in value if c.isdigit())
    if len(digits) != 10:
        raise ValueError("Phone must have 10 digits")
    return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"

# Create reusable types
Email = Annotated[str, AfterValidator(validate_email)]
Phone = Annotated[str, AfterValidator(validate_phone)]

# Use across models
class Contact(BaseModel):
    email: Email
    phone: Phone

# Test
contact = Contact(email="Alice@EXAMPLE.COM", phone="555-123-4567")
print(contact.model_dump())
# {'email': 'alice@example.com', 'phone': '+1-555-123-4567'}
```

---

## Advanced Patterns

### Tagged Unions

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Discriminator

class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict

class ErrorResponse(BaseModel):
    status: Literal["error"]
    error: str
    code: int

# ✅ OPTIMIZED - Tagged union (O(1) lookup)
Response = Annotated[
    SuccessResponse | ErrorResponse,
    Discriminator("status")
]

class APIResult(BaseModel):
    response: Response

# Pydantic uses "status" field to directly instantiate correct type
result1 = APIResult(response={"status": "success", "data": {"id": 123}})
result2 = APIResult(response={"status": "error", "error": "Not found", "code": 404})
```

---

### Railway Pattern (FlextResult Equivalent)

```python
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")
E = TypeVar("E")

@dataclass
class Result(Generic[T]):
    """Functional error handling - railway pattern."""
    _value: T | None = None
    _error: str | None = None

    @property
    def is_success(self) -> bool:
        return self._error is None

    @property
    def value(self) -> T:
        if self._error:
            raise ValueError(self._error)
        return self._value

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """Create success."""
        return cls(_value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        """Create failure."""
        return cls(_error=error)

    def flat_map(self, func):
        """Chain operations."""
        if self._error:
            return self
        return func(self._value)

    def map(self, func):
        """Transform success."""
        if self._error:
            return self
        return Result.ok(func(self._value))

# Usage
def validate_user(name: str) -> Result[str]:
    if not name.strip():
        return Result.fail("Name cannot be empty")
    return Result.ok(name.upper())

def save_user(name: str) -> Result[dict]:
    return Result.ok({"name": name, "id": 123})

# Chain operations
result = (
    validate_user("alice")
    .flat_map(save_user)
    .map(lambda u: f"Saved: {u}")
)

if result.is_success:
    print(result.value)  # "Saved: {'name': 'ALICE', 'id': 123}"
```

---

### Pydantic with Railway Pattern

```python
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass

@dataclass
class Result:
    """Simplified Result type."""
    ok: bool
    value: object = None
    error: str = None

class User(BaseModel):
    name: str
    age: int

def validate_and_save(data: dict) -> Result:
    """Validate model and handle errors."""
    try:
        user = User(**data)
        # Save to database
        return Result(ok=True, value={"id": 123, **user.model_dump()})
    except ValidationError as e:
        errors = ", ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return Result(ok=False, error=errors)

# Usage
result = validate_and_save({"name": "Alice", "age": 30})
if result.ok:
    print(f"Saved: {result.value}")
else:
    print(f"Error: {result.error}")
```

---

### Deprecation Pattern

```python
import warnings
from pydantic import BaseModel, ConfigDict
from typing import Annotated
from pydantic import Field

# Old API (deprecated)
class UserOld(BaseModel):
    model_config = ConfigDict(extra="allow")

    def validate_age_old(self, value: int) -> int:
        """Deprecated: Use Pydantic validators instead."""
        warnings.warn(
            "validate_age_old() is deprecated, Pydantic v2 validates automatically",
            DeprecationWarning,
            stacklevel=2
        )
        return value

# New API (correct)
AgeConstraint = Annotated[int, Field(ge=0, le=150)]

class UserNew(BaseModel):
    name: str
    age: AgeConstraint  # Validation built-in

# Migration helper
class UserMigration(BaseModel):
    """Transitional version supporting both APIs."""
    name: str
    age: AgeConstraint

    # Support old validation method during transition
    def validate_age_old(self, value: int) -> int:
        warnings.warn("Use age field directly", DeprecationWarning, stacklevel=2)
        return value
```

---

### Performance Optimization

```python
from pydantic import BaseModel, TypeAdapter
import json

class User(BaseModel):
    name: str
    age: int

# ✅ CORRECT - TypeAdapter at module level
_USER_ADAPTER = TypeAdapter(User)
_USER_LIST_ADAPTER = TypeAdapter(list[User])

def process_single_json(json_str: str) -> User:
    """Optimized single object parsing."""
    # One-pass Rust parsing
    return User.model_validate_json(json_str)

def process_list_json(json_str: str) -> list[User]:
    """Optimized list parsing with module-level adapter."""
    return _USER_LIST_ADAPTER.validate_json(json_str)

def process_many(items: list[dict]) -> list[User]:
    """Optimized batch validation."""
    # Reuse adapter, don't create in loop
    return [_USER_ADAPTER.validate_python(item) for item in items]

# Benchmark comparison
import time

data_list = [{"name": f"User{i}", "age": 20+i} for i in range(1000)]
json_str = json.dumps(data_list)

# ❌ SLOW
start = time.time()
for _ in range(100):
    users = [User(**item) for item in data_list]
slow_time = time.time() - start

# ✅ FAST
start = time.time()
for _ in range(100):
    users = process_many(data_list)
fast_time = time.time() - start

print(f"Slow: {slow_time:.3f}s, Fast: {fast_time:.3f}s")
print(f"Improvement: {(slow_time/fast_time - 1)*100:.0f}% faster")
```

---

## Complete Application Example

```python
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Annotated
from pydantic import Field
from datetime import datetime

# Reusable types
EmailStr = Annotated[str, Field(pattern=r"^[^@]+@[^@]+\.[^@]+$")]
NonEmptyStr = Annotated[str, Field(min_length=1)]

# Models
class Address(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    street: NonEmptyStr
    city: NonEmptyStr
    zip_code: str

class User(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: NonEmptyStr
    email: EmailStr
    age: Annotated[int, Field(ge=0, le=150)]
    address: Address
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("age")
    @classmethod
    def validate_adult(cls, value: int) -> int:
        """Additional business logic."""
        if value < 18:
            raise ValueError("Must be adult")
        return value

# Usage
user_data = {
    "name": "Alice Smith",
    "email": "alice@example.com",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "NYC",
        "zip_code": "10001"
    }
}

user = User(**user_data)
print(user.model_dump_json(indent=2))
```

---

**Next**: [Appendix F: FAQ](./APPENDIX_F_FAQ.md)
