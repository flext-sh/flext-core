# Appendix C: Common Errors and Solutions

**Status**: TROUBLESHOOTING GUIDE
**Purpose**: Quick solutions for common Pydantic v2 migration issues
**Usage**: Reference when encountering errors

---

## Table of Contents

1. [Import Errors](#import-errors)
2. [Validation Errors](#validation-errors)
3. [Serialization Errors](#serialization-errors)
4. [Type Errors](#type-errors)
5. [Test Failures](#test-failures)
6. [Performance Issues](#performance-issues)

---

## Import Errors

### Error: "cannot import name 'validator'"

**Symptom**:
```python
ImportError: cannot import name 'validator' from 'pydantic'
```

**Cause**: Using Pydantic v1 import

**Solution**:
```python
# ❌ WRONG (v1)
from pydantic import validator

# ✅ CORRECT (v2)
from pydantic import field_validator
```

---

### Error: "cannot import name 'root_validator'"

**Symptom**:
```python
ImportError: cannot import name 'root_validator' from 'pydantic'
```

**Cause**: Using Pydantic v1 import

**Solution**:
```python
# ❌ WRONG (v1)
from pydantic import root_validator

# ✅ CORRECT (v2)
from pydantic import model_validator
```

---

### Error: "cannot import name 'BaseSettings'"

**Symptom**:
```python
ImportError: cannot import name 'BaseSettings' from 'pydantic'
```

**Cause**: BaseSettings moved to separate package

**Solution**:
```bash
# Install pydantic-settings
poetry add pydantic-settings
```

```python
# ❌ WRONG (v1)
from pydantic import BaseSettings

# ✅ CORRECT (v2)
from pydantic_settings import BaseSettings
```

---

## Validation Errors

### Error: "Extra inputs are not permitted"

**Symptom**:
```python
ValidationError: Extra inputs are not permitted [type=extra_forbidden]
```

**Cause**: Model has `extra="forbid"` (Pydantic v2 default changed)

**Solution**:
```python
from pydantic import BaseModel, ConfigDict

# Option 1: Allow extra fields
class User(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str

# Option 2: Ignore extra fields
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str

# Option 3: Keep strict (v2 default)
class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
```

---

### Error: "Field required"

**Symptom**:
```python
ValidationError: Field required [type=missing]
```

**Cause**: Required field not provided

**Solution**:
```python
# Option 1: Make field optional
class User(BaseModel):
    name: str
    email: str | None = None  # Optional

# Option 2: Provide default
class User(BaseModel):
    name: str
    role: str = "user"  # Default value

# Option 3: Provide value when creating
user = User(name="Alice", email="alice@example.com")
```

---

### Error: "Input should be a valid integer"

**Symptom**:
```python
ValidationError: Input should be a valid integer [type=int_type]
```

**Cause**: Pydantic v2 strict mode by default for some types

**Solution**:
```python
# Option 1: Disable strict mode
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(strict=False)
    age: int  # Will coerce "30" → 30

# Option 2: Use BeforeValidator
from typing import Annotated
from pydantic import BeforeValidator

def parse_int(value: str | int) -> int:
    return int(value)

class User(BaseModel):
    age: Annotated[int, BeforeValidator(parse_int)]
```

---

### Error: "instance of X, str or dict required"

**Symptom**:
```python
ValidationError: instance of User expected, not dict [type=model_type]
```

**Cause**: Nested model validation changed in v2

**Solution**:
```python
# ✅ CORRECT - Use model_validate for nested models
class Address(BaseModel):
    city: str

class User(BaseModel):
    name: str
    address: Address

# Pydantic v2 auto-converts dicts to nested models
user = User(name="Alice", address={"city": "NYC"})  # Works!

# If getting errors, ensure model_config allows it:
class User(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)
    name: str
    address: Address
```

---

## Serialization Errors

### Error: "AttributeError: 'User' object has no attribute 'dict'"

**Symptom**:
```python
AttributeError: 'User' object has no attribute 'dict'
```

**Cause**: Using Pydantic v1 method

**Solution**:
```python
# ❌ WRONG (v1)
data = user.dict()

# ✅ CORRECT (v2)
data = user.model_dump()
```

---

### Error: "AttributeError: 'User' object has no attribute 'json'"

**Symptom**:
```python
AttributeError: 'User' object has no attribute 'json'
```

**Cause**: Using Pydantic v1 method

**Solution**:
```python
# ❌ WRONG (v1)
json_str = user.json()

# ✅ CORRECT (v2)
json_str = user.model_dump_json()
```

---

### Error: "Object of type 'Decimal' is not JSON serializable"

**Symptom**:
```python
TypeError: Object of type 'Decimal' is not JSON serializable
```

**Cause**: Using `json.dumps()` instead of `model_dump_json()`

**Solution**:
```python
from decimal import Decimal

class Product(BaseModel):
    price: Decimal

product = Product(price=Decimal("19.99"))

# ❌ WRONG - json.dumps doesn't handle Decimal
import json
json_str = json.dumps(product.model_dump())  # Error!

# ✅ CORRECT - Use model_dump_json() or mode="json"
json_str = product.model_dump_json()  # Works!
# OR
data = product.model_dump(mode="json")  # Returns JSON-serializable types
```

---

## Type Errors

### Error: "Argument 1 to 'validator' has incompatible type"

**Symptom**:
```python
error: Argument 1 to "validator" has incompatible type [arg-type]
```

**Cause**: Using Pydantic v1 validator with v2 type checker

**Solution**:
```python
# ❌ WRONG (v1)
@validator("age")
def validate_age(cls, value):
    return value

# ✅ CORRECT (v2)
@field_validator("age")
@classmethod
def validate_age(cls, value: int) -> int:
    return value
```

---

### Error: "'Config' is not a valid field name"

**Symptom**:
```python
TypeError: 'Config' is not a valid field name [type=config_error]
```

**Cause**: Using `class Config` in Pydantic v2

**Solution**:
```python
# ❌ WRONG (v1)
class User(BaseModel):
    class Config:
        frozen = True
    name: str

# ✅ CORRECT (v2)
from pydantic import ConfigDict

class User(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
```

---

### Error: "Expected type, got 'Field'"

**Symptom**:
```python
error: Expected type in class pattern [misc]
```

**Cause**: Using Field without Annotated

**Solution**:
```python
# ❌ WRONG - Field as type annotation
class User(BaseModel):
    name: str = Field(min_length=1)  # Triggers warning in some tools

# ✅ CORRECT - Use Annotated
from typing import Annotated

class User(BaseModel):
    name: Annotated[str, Field(min_length=1)]
```

---

## Test Failures

### Error: "Expected ValidationError, got AttributeError"

**Symptom**:
```python
pytest.raises(ValidationError):
    frozen_model.field = "new value"  # Raises AttributeError instead
```

**Cause**: Frozen models raise AttributeError in v2, not ValidationError

**Solution**:
```python
# ❌ WRONG (v1 expectation)
with pytest.raises(ValidationError):
    frozen_model.field = "value"

# ✅ CORRECT (v2 behavior)
with pytest.raises(AttributeError, match="frozen"):
    frozen_model.field = "value"
```

---

### Error: "AssertionError: dict keys differ"

**Symptom**:
```python
AssertionError: {'name', 'age'} != {'name', 'age', 'computed_field'}
```

**Cause**: @computed_field automatically included in model_dump()

**Solution**:
```python
from pydantic import computed_field

class User(BaseModel):
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

# ✅ CORRECT - Computed field is in output
expected = {
    "first_name": "Alice",
    "last_name": "Smith",
    "full_name": "Alice Smith"  # Included!
}

# To exclude in tests:
data = user.model_dump(exclude={"full_name"})
```

---

### Error: "TypeError: 'tuple' object is not callable"

**Symptom**:
```python
TypeError: 'tuple' object is not callable [call-arg]
```

**Cause**: Multiple validators require tuple unpacking

**Solution**:
```python
# ❌ WRONG
@field_validator("name", "email")
def validate_fields(cls, value):  # Missing * unpacking
    return value

# ✅ CORRECT
@field_validator("name", "email")
@classmethod
def validate_fields(cls, value: str) -> str:
    return value

# OR with validation info
from pydantic import ValidationInfo

@field_validator("name", "email")
@classmethod
def validate_fields(cls, value: str, info: ValidationInfo) -> str:
    print(f"Validating field: {info.field_name}")
    return value
```

---

## Performance Issues

### Issue: "Slow JSON parsing"

**Symptom**: JSON parsing takes 2-3x longer than expected

**Cause**: Using two-step parsing (json.loads + model_validate)

**Solution**:
```python
import json

# ❌ SLOW - Two-pass (Python → Rust → Python)
data = json.loads(json_str)
model = User.model_validate(data)

# ✅ FAST - One-pass Rust parsing
model = User.model_validate_json(json_str)
```

**Benchmark**: 50-70% faster with model_validate_json()

---

### Issue: "TypeAdapter creation overhead"

**Symptom**: High CPU usage when processing many items

**Cause**: Creating TypeAdapter inside hot loop

**Solution**:
```python
from pydantic import TypeAdapter

# ❌ SLOW - Created every iteration
def process_users(items: list[dict]) -> list[User]:
    results = []
    for item in items:
        adapter = TypeAdapter(User)  # OVERHEAD!
        results.append(adapter.validate_python(item))
    return results

# ✅ FAST - Created once at module level
_USER_ADAPTER = TypeAdapter(User)

def process_users(items: list[dict]) -> list[User]:
    return [_USER_ADAPTER.validate_python(item) for item in items]
```

**Benchmark**: 30-40% faster with module-level adapter

---

### Issue: "Slow union validation"

**Symptom**: Union validation taking longer than expected

**Cause**: Plain unions without discriminator

**Solution**:
```python
from typing import Annotated, Literal
from pydantic import Discriminator

class Cat(BaseModel):
    type: Literal["cat"]
    meow: str

class Dog(BaseModel):
    type: Literal["dog"]
    bark: str

# ❌ SLOW - Plain union (tries each type)
Animal = Cat | Dog

# ✅ FAST - Tagged union (O(1) dispatch)
Animal = Annotated[
    Cat | Dog,
    Discriminator("type")
]
```

**Benchmark**: O(1) vs O(N) for union member count

---

## Error Code Reference

Common Pydantic v2 error codes:

| Error Code | Meaning | Solution |
|------------|---------|----------|
| `extra_forbidden` | Extra field not allowed | Set `extra="allow"` or remove field |
| `missing` | Required field missing | Provide value or make optional |
| `int_type` | Not a valid integer | Use BeforeValidator or disable strict |
| `model_type` | Not a valid model instance | Use nested dict or model_validate |
| `frozen_field` | Cannot modify frozen field | Remove `frozen=True` or don't modify |
| `json_invalid` | Invalid JSON string | Check JSON syntax |
| `value_error` | Custom validation failed | Fix input or adjust validator |

---

## Debugging Tips

### 1. Enable Verbose Validation Errors

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True, # Auto-strip whitespace
        use_enum_values=True,      # Use enum values in errors
    )
```

### 2. Inspect Validation Context

```python
from pydantic import field_validator, ValidationInfo

class User(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info: ValidationInfo) -> str:
        print(f"Field: {info.field_name}")
        print(f"Data: {info.data}")
        print(f"Context: {info.context}")
        return value
```

### 3. Catch and Inspect Validation Errors

```python
from pydantic import ValidationError

try:
    user = User(**data)
except ValidationError as e:
    print(e.json())  # Pretty-printed errors
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Message: {error['msg']}")
        print(f"Type: {error['type']}")
```

---

## Getting Help

1. **Check Official Docs**: `/home/marlonsc/flext/docs/references/pydantic2/`
2. **Search This Guide**: Common errors listed above
3. **Enable Debug Logging**: Set `PYDANTIC_DEBUG=1`
4. **Check Error Type**: Use error type to find solution

---

**Next**: [Appendix D: Glossary](./APPENDIX_D_GLOSSARY.md)
