# Pydantic v2 Patterns for FLEXT Ecosystem

**Version**: 1.0.0
**Date**: 2025-10-21
**Status**: PRODUCTION-READY (All 33 projects 100% compliant)
**Audience**: All FLEXT developers

---

## Quick Reference

### Model Definition

```python
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from typing import Annotated

class User(BaseModel):
    """User model with Pydantic v2 patterns."""

    # Configuration (MANDATORY for Pydantic v2)
    model_config = ConfigDict(
        frozen=False,                # Mutable by default
        validate_assignment=True,    # Validate on assignment
        str_strip_whitespace=True,   # Strip strings
    )

    # Fields with constraints (NO custom validators needed)
    username: Annotated[str, Field(min_length=3, max_length=50)]
    email: EmailStr                  # Pydantic built-in type
    age: Annotated[int, Field(ge=0, le=150)]
    port: Annotated[int, Field(ge=1, le=65535)]  # Or use PortNumber from FlextTypes
```

### Validators (Business Logic Only)

```python
from pydantic import field_validator, model_validator

class User(BaseModel):
    username: str
    password: str
    password_confirm: str

    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """Field validator - runs AFTER Pydantic v2 validation.

        Use ONLY for business logic, not for constraints that Field() handles.
        """
        if not v.isalnum():
            raise ValueError('must be alphanumeric')
        return v.lower()

    @model_validator(mode='after')
    @classmethod
    def passwords_match(cls, model: 'User') -> 'User':
        """Model validator - validates entire model after all fields validated.

        Use ONLY for cross-field validation.
        """
        if model.password != model.password_confirm:
            raise ValueError('passwords do not match')
        return model
```

### Serialization

```python
user = User(username='john', email='john@example.com', age=30)

# Python mode (preserves Python types)
python_dict = user.model_dump()
# {'username': 'john', 'email': 'john@example.com', 'age': 30}

# JSON mode (JSON-compatible types)
json_dict = user.model_dump(mode='json')
# Same for most models, but tuple→list, datetime→str, etc.

# JSON string (FASTEST method for JSON serialization - uses Rust!)
json_str = user.model_dump_json()
# '{"username":"john","email":"john@example.com","age":30}'

# Exclude fields
user.model_dump(exclude={'password'})

# Include only specific fields
user.model_dump(include={'username', 'email'})

# Use aliases
class User(BaseModel):
    username: str = Field(serialization_alias='user_name')

user.model_dump(by_alias=True)
# {'user_name': 'john', ...}
```

### Validation

```python
# From dict
data = {'username': 'john', 'email': 'john@example.com'}
user = User.model_validate(data)

# From JSON string (FAST - Rust-based parsing)
json_str = '{"username":"john","email":"john@example.com"}'
user = User.model_validate_json(json_str)

# Strict mode (no coercion)
user = User.model_validate(data, strict=True)

# Context (pass extra data to validators)
user = User.model_validate(data, context={'admin': True})

@field_validator('username')
@classmethod
def validate_username(cls, v: str, info) -> str:
    """Access validation context."""
    if info.context and info.context.get('admin'):
        # Special rules for admin
        pass
    return v
```

---

## Common Migrations

### Migration 1: Class Config → ConfigDict

```python
# ❌ OLD (Pydantic v1)
class User(BaseModel):
    username: str

    class Config:
        frozen = True
        validate_assignment = True

# ✅ NEW (Pydantic v2)
from pydantic import ConfigDict

class User(BaseModel):
    username: str

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
    )
```

### Migration 2: @validator → @field_validator

```python
# ❌ OLD (Pydantic v1)
from pydantic import validator

class User(BaseModel):
    username: str

    @validator('username')
    def username_lower(cls, v):
        return v.lower()

# ✅ NEW (Pydantic v2)
from pydantic import field_validator

class User(BaseModel):
    username: str

    @field_validator('username')
    @classmethod
    def username_lower(cls, v: str) -> str:
        return v.lower()
```

### Migration 3: Serialization Methods

```python
# ❌ OLD (Pydantic v1)
user_dict = user.dict()
user_json = user.json()
user_json_dict = user.dict(exclude={'password'})

# ✅ NEW (Pydantic v2)
user_dict = user.model_dump()
user_json = user.model_dump_json()
user_json_dict = user.model_dump(exclude={'password'})
```

### Migration 4: Parse Methods

```python
# ❌ OLD (Pydantic v1)
user = User.parse_obj(data)
user = User.parse_raw(json_string)

# ✅ NEW (Pydantic v2)
user = User.model_validate(data)
user = User.model_validate_json(json_string)
```

### Migration 5: Custom Validators → Pydantic Native Types

```python
# ❌ OLD (Pydantic v1) - Custom validators duplicating Pydantic
class Config(BaseModel):
    port: int
    email: str

    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Invalid port')
        return v

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# ✅ NEW (Pydantic v2) - Use native types
from pydantic import EmailStr, Field
from typing import Annotated

class Config(BaseModel):
    port: Annotated[int, Field(ge=1, le=65535)]
    email: EmailStr  # Pydantic validates format natively
```

---

## Pydantic v2 Native Types

### Built-in Types

| Type | Purpose | Example |
|------|---------|---------|
| `EmailStr` | Email validation | `user_email: EmailStr` |
| `HttpUrl` | HTTP/HTTPS URL | `website: HttpUrl` |
| `SecretStr` | Passwords (masked in repr) | `password: SecretStr` |
| `UUID4` | UUID validation | `id: UUID4` |
| `PositiveInt` | Positive integers | `count: PositiveInt` |
| `NonNegativeInt` | Non-negative integers | `age: NonNegativeInt` |

### FlextTypes Domain Types

| Type | Range | Purpose |
|------|-------|---------|
| `PortNumber` | 1-65535 | Network ports |
| `TimeoutSeconds` | 0-300 | Timeout values |
| `RetryCount` | 0-10 | Retry attempts |
| `LogLevel` | DEBUG, INFO, WARNING, ERROR, CRITICAL | Log levels |
| `NonEmptyStr` | min_length=1 | Non-empty strings |

```python
from flext_core import PortNumber, TimeoutSeconds, LogLevel
from pydantic import BaseModel

class AppConfig(BaseModel):
    port: PortNumber        # Automatically validates 1-65535
    timeout: TimeoutSeconds # Automatically validates 0-300 seconds
    log_level: LogLevel     # Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
```

---

## Field Constraints (Use Instead of Custom Validators)

```python
from pydantic import Field
from typing import Annotated

class Product(BaseModel):
    # String constraints
    name: Annotated[str, Field(min_length=1, max_length=100)]
    sku: Annotated[str, Field(pattern=r'^[A-Z0-9]{5,10}$')]

    # Numeric constraints
    price: Annotated[float, Field(gt=0)]  # Greater than
    discount: Annotated[float, Field(ge=0, le=100)]  # Between 0-100
    quantity: Annotated[int, Field(multiple_of=1)]  # Must be multiple of 1

    # List constraints
    tags: Annotated[list[str], Field(min_length=1, max_length=10)]

    # Descriptions and examples
    description: Annotated[str, Field(
        min_length=10,
        max_length=1000,
        description="Product description",
        examples=["This is a great product!"]
    )]
```

---

## Performance Best Practices

### ⚡ JSON Parsing (Use model_validate_json)

```python
import json

# ✅ FAST (one-step, Rust-based) - O(1) in Pydantic's Rust validator
user = User.model_validate_json(json_string)

# ❌ SLOW (two-step, Python-based)
data = json.loads(json_string)  # Python JSON parsing
user = User.model_validate(data)  # Python validation

# Difference: ~2-3x faster with model_validate_json for large JSON
```

### ⚡ TypeAdapter (Module-level Constants)

```python
from pydantic import TypeAdapter
from typing import Final

# ✅ FAST - Created once, reused many times
_USER_LIST_ADAPTER: Final = TypeAdapter(list[User])
_CONFIG_DICT_ADAPTER: Final = TypeAdapter(dict[str, Config])

def validate_users(data):
    return _USER_LIST_ADAPTER.validate_python(data)

def validate_configs(data):
    return _CONFIG_DICT_ADAPTER.validate_python(data)

# ❌ SLOW - Created on every call
def validate_users(data):
    adapter = TypeAdapter(list[User])  # Recreated each call
    return adapter.validate_python(data)
```

### ⚡ Tagged Unions (Use Discriminator)

```python
from pydantic import Discriminator
from typing import Annotated, Union

# ✅ FAST - O(1) discriminator lookup
class Command:
    type: str = 'command'
    command: str

class Event:
    type: str = 'event'
    event_name: str

class Query:
    type: str = 'query'
    query: str

Message = Annotated[
    Union[Command, Event, Query],
    Discriminator('type')  # O(1) validation based on 'type' field
]

# ❌ SLOW - Tries each type until one matches (O(n))
Message = Command | Event | Query  # No discriminator
```

---

## Forbidden Patterns (WILL FAIL AUDIT)

### ❌ NO Pydantic v1 Patterns

```python
# FORBIDDEN - Will fail make audit-pydantic-v2
class Config(BaseModel):
    class Config:  # ❌ Use model_config = ConfigDict()
        pass

    model_dict = model.dict()  # ❌ Use model.model_dump()
    model_json = model.json()  # ❌ Use model.model_dump_json()
    parsed = User.parse_obj(data)  # ❌ Use User.model_validate(data)

    @validator('field')  # ❌ Use @field_validator
    def validate_field(cls, v):
        return v

    @root_validator  # ❌ Use @model_validator
    def validate_root(cls, values):
        return values
```

### ❌ NO Custom Validation Duplication

```python
# FORBIDDEN - Custom validators that duplicate Pydantic
class Config(BaseModel):
    email: str
    port: int

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):  # ❌ Use EmailStr type instead
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):  # ❌ Use Field(ge=1, le=65535) instead
        if not 1 <= v <= 65535:
            raise ValueError('Invalid port')
        return v
```

---

## Quality Gates (Automated Enforcement)

All Pydantic v2 compliance is automatically enforced:

```bash
# Check compliance before commit
make audit-pydantic-v2

# Full validation (includes Pydantic check)
make validate

# Pre-commit hook blocks Pydantic v1 patterns
# GitHub Actions CI/CD blocks PRs with violations
```

---

## References

- **Pydantic v2 Docs**: https://docs.pydantic.dev/latest/
- **FlextTypes**: `/home/marlonsc/flext/flext-core/src/flext_core/typings.py`
- **Audit Script**: `/home/marlonsc/flext/flext-core/docs/pydantic-v2-modernization/audit_pydantic_v2.py`
- **flext-core CLAUDE.md**: `Pydantic v2 Standards` section
- **Workspace CLAUDE.md**: `Pydantic v2 Standards` section

---

**Status**: ✅ ALL 33 FLEXT projects are 100% Pydantic v2 compliant (Phase 5 verified)

**Phase 6**: ✅ Quality gates (pre-commit, CI/CD, IDE) deployed and operational

**Next**: Phase 7 documentation complete, Phase 8+ (execution timeline, metrics)
