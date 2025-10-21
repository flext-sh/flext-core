# Part 7: Documentation & Training

**Status**: KNOWLEDGE TRANSFER (After Parts 2-6 implementation)
**Priority**: 🟡 MEDIUM (Enable long-term ecosystem consistency)
**Estimated Time**: 4-6 hours
**Impact**: Team enablement, onboarding, long-term maintainability

**Related**:
- Quality gates: [06-quality-gates.md](./06-quality-gates.md) (automated enforcement)
- Execution timeline: [08-execution-timeline.md](./08-execution-timeline.md) (scheduling)
- Success metrics: [09-metrics-risks.md](./09-metrics-risks.md) (measurement)
- Audit script: [audit_pydantic_v2.py](./audit_pydantic_v2.py) (automated checks)

**FLEXT Standards**:
- Workspace CLAUDE.md: `/home/marlonsc/flext/CLAUDE.md`
- flext-core CLAUDE.md: `/home/marlonsc/flext/flext-core/CLAUDE.md`
- Architecture patterns: `/home/marlonsc/flext/CLAUDE.md` (Clean Architecture section)

**Pydantic v2 References**:
- Validators: `/home/marlonsc/flext/docs/references/pydantic2/concepts/validators.md`
- Fields: `/home/marlonsc/flext/docs/references/pydantic2/concepts/fields.md`
- Models: `/home/marlonsc/flext/docs/references/pydantic2/concepts/models.md`
- Performance: `/home/marlonsc/flext/docs/references/pydantic2/concepts/performance.md`
- Serialization: `/home/marlonsc/flext/docs/references/pydantic2/concepts/serialization.md`

---

## Overview

Create comprehensive documentation and training materials for Pydantic v2 ecosystem adoption:
1. Update all CLAUDE.md files with standards
2. Create PYDANTIC_V2_PATTERNS.md best practices guide
3. Training examples and code samples
4. Migration guides for dependent projects
5. Team onboarding materials

---

## Section 7.1: Update CLAUDE.md Files

### flext-core/CLAUDE.md

**Add section**:

```markdown
## Pydantic v2 Standards (MANDATORY)

### Required Patterns

✅ **Model Configuration**:
```python
class MyModel(BaseModel):
    model_config = ConfigDict(frozen=True, validate_assignment=True)
```

✅ **Validators**:
```python
@field_validator('field_name')
@classmethod
def validate_field(cls, v): ...

@model_validator(mode='after')
@classmethod
def validate_model(cls, values): ...
```

✅ **Serialization**:
```python
model.model_dump()  # Python dict
model.model_dump_json()  # JSON string
model.model_dump(mode='json')  # JSON-compatible dict
```

✅ **Validation**:
```python
MyModel.model_validate(data)  # From dict
MyModel.model_validate_json(json_str)  # From JSON (faster!)
```

✅ **Reusable Types**:
```python
from flext_core import PortNumber, EmailAddress, TimeoutSeconds
from pydantic import Field
from typing import Annotated

CustomType = Annotated[int, Field(gt=0, le=100)]
```

### Forbidden Patterns

❌ **NO Pydantic v1**:
- `class Config:` → Use `model_config = ConfigDict()`
- `.dict()` → Use `.model_dump()`
- `.json()` → Use `.model_dump_json()`
- `parse_obj()` → Use `.model_validate()`
- `@validator` → Use `@field_validator`
- `@root_validator` → Use `@model_validator`

❌ **NO Duplication**:
- Don't create validation methods duplicating Pydantic
- Use Pydantic native types (EmailStr, HttpUrl, PositiveInt)
- Use FlextTypes domain types (PortNumber, TimeoutSeconds)

### Performance Best Practices

⚡ **JSON Parsing**:
```python
# ✅ FAST (one-step, Rust-based)
model = MyModel.model_validate_json(json_string)

# ❌ SLOW (two-step, Python-based)
import json
data = json.loads(json_string)
model = MyModel.model_validate(data)
```

⚡ **TypeAdapter**:
```python
# ✅ FAST (module-level, created once)
_ADAPTER: Final = TypeAdapter(list[MyModel])

def validate_items(data):
    return _ADAPTER.validate_python(data)

# ❌ SLOW (created every call)
def validate_items(data):
    adapter = TypeAdapter(list[MyModel])
    return adapter.validate_python(data)
```

⚡ **Tagged Unions**:
```python
# ✅ FAST (discriminator)
from pydantic import Discriminator

Message = Annotated[
    Command | Event | Query,
    Discriminator('type')
]

# ❌ SLOW (tries each type)
Message = Command | Event | Query
```
```

---

## Section 7.2: Create PYDANTIC_V2_PATTERNS.md

**File**: `flext-core/docs/PYDANTIC_V2_PATTERNS.md`

```markdown
# Pydantic v2 Patterns for FLEXT Ecosystem

**Version**: 1.0.0
**Date**: 2025-01-21
**Audience**: All FLEXT developers

---

## Quick Reference

### Model Definition

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated

class User(BaseModel):
    """User model with Pydantic v2 patterns."""
    
    # Configuration
    model_config = ConfigDict(
        frozen=False,  # Mutable by default
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True,  # Strip strings
    )
    
    # Fields with constraints
    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r'^[a-z0-9_]+$')]
    email: EmailStr  # Pydantic built-in
    age: Annotated[int, Field(ge=0, le=150)]
    port: PortNumber  # FlextTypes domain type
```

### Validators

```python
from pydantic import field_validator, model_validator

class User(BaseModel):
    username: str
    password: str
    password_confirm: str
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """Field validator - runs after Pydantic validation."""
        if not v.isalnum():
            raise ValueError('must be alphanumeric')
        return v.lower()
    
    @model_validator(mode='after')
    @classmethod
    def passwords_match(cls, model: 'User') -> 'User':
        """Model validator - validates entire model."""
        if model.password != model.password_confirm:
            raise ValueError('passwords do not match')
        return model
```

### Serialization

```python
user = User(username='john', email='john@example.com', age=30, port=8080)

# Python mode (preserves Python types)
python_dict = user.model_dump()
# {'username': 'john', 'email': 'john@example.com', 'age': 30, 'port': 8080}

# JSON mode (JSON-compatible types)
json_dict = user.model_dump(mode='json')
# Same for this model, but would convert tuple→list, etc.

# JSON string (fastest for JSON)
json_str = user.model_dump_json()
# '{"username":"john","email":"john@example.com","age":30,"port":8080}'

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

# From JSON string (FAST - use this for JSON!)
json_str = '{"username":"john","email":"john@example.com"}'
user = User.model_validate_json(json_str)

# Strict mode (no coercion)
user = User.model_validate(data, strict=True)

# Context (pass extra data to validators)
user = User.model_validate(data, context={'REDACTED_LDAP_BIND_PASSWORD': True})
```

### Reusable Types (FlextTypes)

```python
from flext_core import (
    PortNumber,  # Annotated[int, Field(ge=1, le=65535)] - MIN_PORT=1
    TimeoutSeconds,  # Annotated[float, Field(gt=0, le=300)]
    RetryCount,  # Annotated[int, Field(ge=0, le=10)]
    NonEmptyStr,  # Annotated[str, Field(min_length=1)]
    LogLevel,  # Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    HostName,  # Annotated[str, ...] with DNS validation
)

class Config(BaseModel):
    port: PortNumber  # Automatically validated (1-65535)
    timeout: TimeoutSeconds  # Automatically validated (0-300 seconds)
    retries: RetryCount  # Automatically validated (0-10)
    host: HostName  # Automatically validated + DNS check
    log_level: LogLevel  # Literal type, autocomplete in IDE
```

### Custom Reusable Types

```python
from typing import Annotated
from pydantic import Field, AfterValidator

# Simple constraint
PositiveEven = Annotated[int, Field(gt=0), Field(multiple_of=2)]

# With custom validator
def validate_uppercase(v: str) -> str:
    if v != v.upper():
        raise ValueError('must be uppercase')
    return v

UppercaseStr = Annotated[str, AfterValidator(validate_uppercase)]

# Use in models
class MyModel(BaseModel):
    count: PositiveEven  # Must be positive and even
    code: UppercaseStr  # Must be uppercase
```

---

## Common Migrations

### Migration 1: Class Config → ConfigDict

```python
# ❌ OLD (Pydantic v1)
class User(BaseModel):
    class Config:
        frozen = True
        validate_assignment = True

# ✅ NEW (Pydantic v2)
from pydantic import ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
    )
```

### Migration 2: Validators

```python
# ❌ OLD (Pydantic v1)
class User(BaseModel):
    username: str
    
    @validator('username')
    def username_lower(cls, v):
        return v.lower()

# ✅ NEW (Pydantic v2)
class User(BaseModel):
    username: str
    
    @field_validator('username')
    @classmethod
    def username_lower(cls, v: str) -> str:
        return v.lower()
```

### Migration 3: Serialization

```python
user = User(username='john')

# ❌ OLD (Pydantic v1)
user_dict = user.dict()
user_json = user.json()

# ✅ NEW (Pydantic v2)
user_dict = user.model_dump()
user_json = user.model_dump_json()
```

### Migration 4: Parsing

```python
data = {'username': 'john'}

# ❌ OLD (Pydantic v1)
user = User.parse_obj(data)
user = User.parse_raw(json_str)

# ✅ NEW (Pydantic v2)
user = User.model_validate(data)
user = User.model_validate_json(json_str)
```

---

## Examples by Use Case

### REST API Response

```python
from fastapi import FastAPI
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int) -> UserResponse:
    # FastAPI automatically calls model_dump() for response
    return UserResponse(id=user_id, username="john", email="john@example.com")
```

### Configuration File

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """Application configuration from environment."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )
    
    database_url: str
    api_key: SecretStr
    port: PortNumber = 8080
    log_level: LogLevel = "INFO"

# Load from environment
config = AppConfig()
```

### Data Validation Pipeline

```python
from pydantic import TypeAdapter

# Define schema
UserList = TypeAdapter(list[User])

# Validate data
users_data = [{'username': 'john'}, {'username': 'jane'}]
users = UserList.validate_python(users_data)

# From JSON (faster!)
json_data = '[{"username":"john"},{"username":"jane"}]'
users = UserList.validate_json(json_data)
```

---

## Performance Tips

### 1. Use model_validate_json() for JSON

```python
# ✅ 2-5x faster
user = User.model_validate_json(json_string)

# ❌ Slower (two-step)
import json
user = User.model_validate(json.loads(json_string))
```

### 2. Reuse TypeAdapter

```python
# ✅ Create once at module level
_USER_LIST_ADAPTER: Final = TypeAdapter(list[User])

def load_users(data: list[dict]) -> list[User]:
    return _USER_LIST_ADAPTER.validate_python(data)
```

### 3. Use Tagged Unions

```python
# ✅ Fast (checks 'type' field first)
Message = Annotated[
    Command | Event | Query,
    Discriminator('type')
]
```

---

## Testing with Pydantic v2

```python
import pytest
from pydantic import ValidationError

def test_user_validation():
    # Valid data
    user = User(username='john', email='john@example.com')
    assert user.username == 'john'
    
    # Invalid data
    with pytest.raises(ValidationError) as exc_info:
        User(username='', email='invalid')
    
    errors = exc_info.value.errors()
    assert len(errors) == 2
    assert errors[0]['type'] == 'string_too_short'

def test_frozen_model():
    from pydantic import ConfigDict
    
    class FrozenUser(BaseModel):
        model_config = ConfigDict(frozen=True)
        username: str
    
    user = FrozenUser(username='john')
    
    # Pydantic v2 raises AttributeError for frozen models
    with pytest.raises(AttributeError, match='frozen'):
        user.username = 'jane'
```

---

**For More**: See Pydantic v2 official docs at `/home/marlonsc/flext/docs/references/pydantic2/`
```

---

## Section 7.3: Training Examples

**File**: `flext-core/examples/pydantic_v2_complete.py`

```python
"""Complete Pydantic v2 examples for FLEXT ecosystem.

This file demonstrates all Pydantic v2 patterns used in FLEXT.
"""

from datetime import datetime
from typing import Annotated, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    field_serializer,
    computed_field,
    EmailStr,
)
from flext_core import PortNumber, TimeoutSeconds

# Example 1: Basic Model
class User(BaseModel):
    """Basic user model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    username: Annotated[str, Field(min_length=3, max_length=50)]
    email: EmailStr
    age: Annotated[int, Field(ge=0, le=150)]

# Example 2: Model with Validators
class Account(BaseModel):
    username: str
    password: str
    password_confirm: str
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('must be alphanumeric')
        return v.lower()
    
    @model_validator(mode='after')
    @classmethod
    def passwords_match(cls, model: 'Account') -> 'Account':
        if model.password != model.password_confirm:
            raise ValueError('passwords do not match')
        return model

# Example 3: Computed Fields
class Product(BaseModel):
    name: str
    price: float
    tax_rate: float = 0.1
    
    @computed_field
    @property
    def price_with_tax(self) -> float:
        return self.price * (1 + self.tax_rate)

# Example 4: Custom Serialization
class Event(BaseModel):
    name: str
    timestamp: datetime
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

# Example 5: Tagged Union (Fast)
class Command(BaseModel):
    type: Literal["command"] = "command"
    action: str

class Event(BaseModel):
    type: Literal["event"] = "event"
    name: str

Message = Annotated[Command | Event, Discriminator('type')]

# Example 6: FlextTypes Usage
class ServerConfig(BaseModel):
    host: str
    port: PortNumber  # Auto-validated (1-65535)
    timeout: TimeoutSeconds  # Auto-validated (0-300)

# Usage examples
if __name__ == "__main__":
    # Create and validate
    user = User(username="john", email="john@example.com", age=30)
    print(f"User: {user.model_dump()}")
    
    # Serialization
    print(f"JSON: {user.model_dump_json()}")
    
    # Validation from JSON (fast!)
    json_str = '{"username":"jane","email":"jane@example.com","age":25}'
    user2 = User.model_validate_json(json_str)
    print(f"From JSON: {user2.username}")
```

---

## Section 7.4: Migration Guide

**File**: `flext-core/docs/PYDANTIC_V2_MIGRATION.md`

Create step-by-step migration guide for team (see PYDANTIC_V2_PATTERNS.md above for content).

---

## Implementation Checklist

### Documentation Updates
- [ ] Update flext-core/CLAUDE.md with Pydantic v2 standards
- [ ] Create PYDANTIC_V2_PATTERNS.md
- [ ] Create PYDANTIC_V2_MIGRATION.md
- [ ] Update README.md with Pydantic v2 requirement
- [ ] Create examples/pydantic_v2_complete.py

### Training Materials
- [ ] Create slide deck (optional)
- [ ] Record video tutorial (optional)
- [ ] Schedule team training session
- [ ] Create FAQ document

### Distribution
- [ ] Share documentation with team
- [ ] Update onboarding materials
- [ ] Add to developer wiki/portal

---

## Success Criteria

After completing Part 7:
- ✅ **All documentation updated** with Pydantic v2 standards
- ✅ **Comprehensive examples** available
- ✅ **Migration guide** complete
- ✅ **Team trained** on new patterns

---

## Next Steps

After completing Part 7:
1. ✅ Distribute documentation to team
2. ✅ Conduct training session
3. ➡️ Proceed to Part 8: [Execution Timeline](./08-execution-timeline.md)

---

**Time Estimate**: 4-6 hours (can be split over multiple days)
