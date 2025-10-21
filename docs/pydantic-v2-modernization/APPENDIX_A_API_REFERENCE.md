# Appendix A: Pydantic v2 API Reference

**Status**: REFERENCE MATERIAL
**Purpose**: Quick reference for Pydantic v2 APIs used in FLEXT ecosystem
**Source**: `/home/marlonsc/flext/docs/references/pydantic2/`

---

## Table of Contents

1. [Model Methods](#model-methods)
2. [Validators](#validators)
3. [Serialization](#serialization)
4. [Field Configuration](#field-configuration)
5. [TypeAdapter](#typeadapter)
6. [ConfigDict Options](#configdict-options)

---

## Model Methods

### model_validate()

**Purpose**: Validate Python dict/object into model instance

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Pydantic v2
user = User.model_validate({"name": "Alice", "age": 30})

# ❌ OLD (v1): parse_obj()
user = User.parse_obj({"name": "Alice", "age": 30})
```

---

### model_validate_json()

**Purpose**: Parse JSON string directly into model (Rust-optimized, 2-3x faster)

```python
import json

# ✅ BEST - One-pass Rust parsing
user = User.model_validate_json('{"name": "Alice", "age": 30}')

# ❌ SLOW - Two-pass parsing (Python → Rust)
data = json.loads('{"name": "Alice", "age": 30}')
user = User.model_validate(data)
```

**Performance**: 50-70% faster than json.loads() + model_validate()

---

### model_dump()

**Purpose**: Export model to dict

```python
user = User(name="Alice", "age": 30)

# ✅ NEW (v2)
data = user.model_dump()
data = user.model_dump(mode="python")  # Python types
data = user.model_dump(mode="json")    # JSON-serializable types
data = user.model_dump(exclude={"age"})
data = user.model_dump(by_alias=True)

# ❌ OLD (v1): .dict()
data = user.dict()
```

---

### model_dump_json()

**Purpose**: Export model directly to JSON string (Rust-optimized)

```python
# ✅ BEST - One-pass Rust serialization
json_str = user.model_dump_json()
json_str = user.model_dump_json(indent=2)
json_str = user.model_dump_json(exclude={"age"})

# ❌ SLOW - Two-pass serialization
data = user.model_dump()
json_str = json.dumps(data)
```

---

### model_copy()

**Purpose**: Create a copy with optional updates

```python
# ✅ NEW (v2)
user2 = user.model_copy()
user2 = user.model_copy(update={"age": 31})

# ❌ OLD (v1): .copy()
user2 = user.copy()
```

---

## Validators

### @field_validator

**Purpose**: Validate single field

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Age must be non-negative")
        return value

# Multiple fields
    @field_validator("name", "email")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty")
        return value

# Mode: before vs after
    @field_validator("age", mode="before")
    @classmethod
    def parse_age(cls, value: str | int) -> int:
        """Run before Pydantic type coercion."""
        return int(value)

    @field_validator("age", mode="after")
    @classmethod
    def validate_age_range(cls, value: int) -> int:
        """Run after Pydantic type coercion."""
        if not 0 <= value <= 150:
            raise ValueError("Invalid age range")
        return value
```

**Modes**:
- `mode="before"`: Runs before Pydantic type coercion
- `mode="after"` (default): Runs after type coercion

---

### @model_validator

**Purpose**: Validate entire model (cross-field validation)

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_date_range(self) -> "DateRange":
        """Cross-field validation after model creation."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

# Mode: before (dict) vs after (model instance)
    @model_validator(mode="before")
    @classmethod
    def normalize_dates(cls, data: dict) -> dict:
        """Run before model instantiation (receives dict)."""
        if "start_date" in data:
            data["start_date"] = parse_date(data["start_date"])
        return data
```

---

### AfterValidator / BeforeValidator (Annotated)

**Purpose**: Reusable validators for Annotated types

```python
from pydantic import AfterValidator, BeforeValidator
from typing import Annotated

def validate_positive(value: int) -> int:
    if value <= 0:
        raise ValueError("Must be positive")
    return value

def parse_int_string(value: str | int) -> int:
    return int(value)

# Create reusable types
PositiveInt = Annotated[int, AfterValidator(validate_positive)]
ParsedInt = Annotated[int, BeforeValidator(parse_int_string)]

class Model(BaseModel):
    count: PositiveInt
    port: ParsedInt
```

---

## Serialization

### @field_serializer

**Purpose**: Custom serialization for specific fields

```python
from pydantic import BaseModel, field_serializer
from decimal import Decimal

class Product(BaseModel):
    name: str
    price: Decimal

    @field_serializer("price")
    def serialize_price(self, value: Decimal) -> str:
        """Convert Decimal to string for JSON."""
        return f"${value:.2f}"

# Result:
product = Product(name="Book", price=Decimal("19.99"))
assert product.model_dump()["price"] == "$19.99"
```

---

### @model_serializer

**Purpose**: Custom serialization for entire model

```python
from pydantic import BaseModel, model_serializer

class User(BaseModel):
    username: str
    password: str

    @model_serializer
    def serialize_model(self) -> dict:
        """Custom serialization excluding password."""
        return {"username": self.username, "is_authenticated": True}
```

---

### @computed_field

**Purpose**: Computed properties included in serialization

```python
from pydantic import BaseModel, computed_field

class User(BaseModel):
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        """Computed field included in model_dump()."""
        return f"{self.first_name} {self.last_name}"

# Result:
user = User(first_name="Alice", last_name="Smith")
assert user.model_dump() == {
    "first_name": "Alice",
    "last_name": "Smith",
    "full_name": "Alice Smith"  # Included automatically
}
```

---

## Field Configuration

### Field()

**Purpose**: Add validation constraints and metadata

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(exclude=True)  # Not in model_dump()
    api_key: str = Field(alias="apiKey")  # JSON uses apiKey
```

**Common Field Parameters**:
- `ge`, `le`: Greater/less than or equal
- `gt`, `lt`: Strictly greater/less than
- `min_length`, `max_length`: String/list length
- `pattern`: Regex validation
- `exclude`: Exclude from serialization
- `alias`: Alternative name for deserialization
- `default`, `default_factory`: Default values
- `description`: Documentation string

---

### Annotated Pattern (RECOMMENDED)

**Purpose**: Combine type with constraints for reusability

```python
from typing import Annotated
from pydantic import BaseModel, Field

# Define reusable types
PortNumber = Annotated[int, Field(ge=1, le=65535)]  # MIN_PORT=1 per constants.py
NonEmptyStr = Annotated[str, Field(min_length=1)]
Email = Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

class ServerConfig(BaseModel):
    port: PortNumber
    host: NonEmptyStr
    REDACTED_LDAP_BIND_PASSWORD_email: Email
```

**Benefits**:
- Type definition in one place (DRY)
- Reusable across multiple models
- Clear semantic meaning

---

## TypeAdapter

**Purpose**: Validate generic types not wrapped in BaseModel

```python
from pydantic import TypeAdapter

# Create adapter (ONCE at module level)
_LIST_ADAPTER = TypeAdapter(list[User])

# Validate data
users = _LIST_ADAPTER.validate_python([
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
])

# JSON parsing
users = _LIST_ADAPTER.validate_json('[{"name": "Alice", "age": 30}]')

# Serialization
json_str = _LIST_ADAPTER.dump_json(users)
```

**Performance Tip**: Create TypeAdapter at module level, NOT inside functions

```python
# ✅ CORRECT - Module level (created once)
_USER_LIST_ADAPTER = TypeAdapter(list[User])

def process_users(data: list[dict]) -> list[User]:
    return _USER_LIST_ADAPTER.validate_python(data)

# ❌ WRONG - Inside function (created every call)
def process_users(data: list[dict]) -> list[User]:
    adapter = TypeAdapter(list[User])  # Overhead!
    return adapter.validate_python(data)
```

---

## ConfigDict Options

**Purpose**: Configure model behavior

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        # Validation
        strict=True,              # No type coercion
        validate_assignment=True, # Validate on field assignment
        validate_default=True,    # Validate default values

        # Serialization
        use_enum_values=True,     # Serialize enums as values
        by_alias=True,            # Use aliases in dump
        exclude_none=True,        # Omit None fields

        # Immutability
        frozen=True,              # Make model immutable

        # Extra fields
        extra="forbid",           # Forbid extra fields
        # extra="allow",          # Allow extra fields
        # extra="ignore",         # Ignore extra fields

        # Performance
        arbitrary_types_allowed=True,  # Allow any types

        # Parsing
        str_strip_whitespace=True,     # Strip whitespace
        str_to_lower=True,             # Lowercase strings
        str_to_upper=True,             # Uppercase strings
    )
```

**Common Options**:
- `strict=True`: No automatic type coercion
- `frozen=True`: Immutable models (Value Objects)
- `extra="forbid"`: Reject unknown fields
- `validate_assignment=True`: Validate on attribute setting

---

## Tagged Unions (Discriminated Unions)

**Purpose**: Fast union validation using discriminator field

```python
from pydantic import BaseModel, Field, Discriminator
from typing import Annotated, Literal

class Cat(BaseModel):
    type: Literal["cat"]
    meow: str

class Dog(BaseModel):
    type: Literal["dog"]
    bark: str

# ✅ OPTIMIZED - Tagged union (O(1) dispatch)
Animal = Annotated[
    Cat | Dog,
    Discriminator("type")
]

class Zoo(BaseModel):
    animals: list[Animal]

# Pydantic uses "type" field to dispatch directly
# No trial-and-error validation
```

**Performance**: O(1) vs O(N) for plain unions

---

## Migration Cheat Sheet

| Pydantic v1 | Pydantic v2 |
|-------------|-------------|
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `.copy()` | `.model_copy()` |
| `.parse_obj()` | `.model_validate()` |
| `.parse_raw()` | `.model_validate_json()` |
| `@validator` | `@field_validator` |
| `@root_validator` | `@model_validator` |
| `class Config:` | `model_config = ConfigDict(...)` |

---

## Additional Resources

- **Official Docs**: `/home/marlonsc/flext/docs/references/pydantic2/`
- **Performance**: `concepts/performance.md`
- **Validators**: `concepts/validators.md`
- **Serialization**: `concepts/serialization.md`
- **Fields**: `concepts/fields.md`

---

**Next**: [Appendix B: Migration Checklist](./APPENDIX_B_MIGRATION_CHECKLIST.md)
