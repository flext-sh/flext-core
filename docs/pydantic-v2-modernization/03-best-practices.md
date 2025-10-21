# Part 3: Pydantic v2 Best Practices Implementation

**Status**: MODERNIZATION CORE
**Priority**: 🔴 CRITICAL
**Estimated Time**: 8-12 hours
**Impact**: Eliminates ~270 lines duplication (verified), major performance gains

---

## Overview

This part covers the core modernization work:
1. Replace validation methods with Pydantic Annotated types
2. Adopt Pydantic v2 performance patterns
3. Enhance serialization patterns
4. Adopt advanced validator patterns

**References**: Based on `/home/marlonsc/flext/docs/references/pydantic2/concepts/` official documentation

---

## Section 3.1: Replace Validation Methods with Annotated Types

### Problem: Code Duplication in utilities.py

**File**: `src/flext_core/utilities.py`
**Issue**: 17+ methods duplicate Pydantic Field constraints

**Example Duplication**:
```python
# utilities.py - 10 lines to do what Pydantic does in 1
@staticmethod
def validate_port(value: int | str) -> FlextResult[int]:
    """Validate network port (1-65535)."""  # Note: MIN_PORT=1 per constants.py
    try:
        port = int(value)
        if port < 1 or port > 65535:
            return FlextResult[int].fail(f"Port must be 1-65535")
        return FlextResult[int].ok(port)
    except ValueError:
        return FlextResult[int].fail(f"Invalid port")
```

**Pydantic v2 Equivalent** (1 line):
```python
PortNumber = Annotated[int, Field(ge=1, le=65535)]  # Note: MIN_PORT=1 per constants.py
```

### Validation Methods Analysis Table

| Method | Lines | Pydantic Equivalent | Action |
|--------|-------|---------------------|--------|
| `validate_port()` | 10 | `Field(ge=1, le=65535)` *(MIN_PORT=1)* | REPLACE |
| `validate_email()` | 8 | `EmailStr` (built-in) | REPLACE |
| `validate_url()` | 8 | `HttpUrl` (built-in) | REPLACE |
| `validate_positive_integer()` | 6 | `PositiveInt` (built-in) | REPLACE |
| `validate_non_negative_integer()` | 6 | `NonNegativeInt` (built-in) | REPLACE |
| `validate_file_path()` | 8 | `FilePath` (built-in) | REPLACE |
| `validate_directory_path()` | 8 | `DirectoryPath` (built-in) | REPLACE |
| `validate_timeout_seconds()` | 7 | `Field(gt=0, le=300)` | REPLACE |
| `validate_retry_count()` | 7 | `Field(ge=0, le=10)` | REPLACE |
| `validate_string_length()` | 9 | `Field(min_length=X, max_length=Y)` | REPLACE |
| `validate_string_pattern()` | 8 | `Field(pattern=r'...')` | REPLACE |
| `validate_log_level()` | 6 | `Literal["DEBUG", ...]` | REPLACE |
| `validate_string_not_empty()` | 5 | `Field(min_length=1)` | REPLACE |
| `validate_pipeline()` | 15 | N/A | KEEP (business logic) |
| `validate_host()` | 12 | N/A | KEEP (DNS check) |

**Summary**: 14 methods to REPLACE (remove), 2-3 to KEEP

### Solution: Expand typings.py with Domain Types

**File**: `src/flext_core/typings.py` (EXPAND)

Add these reusable Annotated types:

```python
from typing import Annotated, Literal
from pydantic import Field, AfterValidator, HttpUrl, EmailStr
from pydantic import FilePath, DirectoryPath, PositiveInt, NonNegativeInt

class FlextTypes:
    # Network types
    type PortNumber = Annotated[
        int,
        Field(ge=1, le=65535, description="Network port (1-65535)"),  # MIN_PORT=1
    ]
    
    type TimeoutSeconds = Annotated[
        float,
        Field(gt=0, le=300, description="Timeout in seconds (max 5 min)"),
    ]
    
    type RetryCount = Annotated[
        int,
        Field(ge=0, le=10, description="Retry attempts (0-10)"),
    ]
    
    # String types
    type NonEmptyStr = Annotated[
        str,
        Field(min_length=1, description="Non-empty string"),
    ]
    
    type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    # Complex validators (can't be simple Field constraints)
    def _validate_dns_host(value: str) -> str:
        import socket
        try:
            socket.gethostbyname(value)
            return value
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve: {e}")
    
    type HostName = Annotated[
        str,
        Field(min_length=1, max_length=253),
        AfterValidator(_validate_dns_host),
    ]

# Module-level exports
PortNumber = FlextTypes.PortNumber
TimeoutSeconds = FlextTypes.TimeoutSeconds
RetryCount = FlextTypes.RetryCount
NonEmptyStr = FlextTypes.NonEmptyStr
LogLevel = FlextTypes.LogLevel
HostName = FlextTypes.HostName
```

### Migration Strategy

**Step 1: Deprecate Methods** (Week 1)

```python
# utilities.py
import warnings
from pydantic import TypeAdapter

def validate_port(value: int | str) -> FlextResult[int]:
    """DEPRECATED: Use PortNumber from FlextTypes.
    
    Removal: v1.2.0 (June 2025)
    Migration: from flext_core import PortNumber; TypeAdapter(PortNumber).validate_python(value)
    """
    warnings.warn("Use PortNumber from FlextTypes", DeprecationWarning, stacklevel=2)
    adapter = TypeAdapter(PortNumber)
    try:
        return FlextResult[int].ok(adapter.validate_python(value))
    except Exception as e:
        return FlextResult[int].fail(str(e))
```

**Step 2: Update Internal Usage** (Week 1-2)

BEFORE:
```python
result = FlextUtilities.validate_port(8080)
if result.is_success:
    port = result.unwrap()
```

AFTER:
```python
from pydantic import TypeAdapter
port = TypeAdapter(PortNumber).validate_python(8080)
```

**Step 3: Remove After 2 Versions** (v1.2.0, June 2025)

---

## Section 3.2: Pydantic v2 Performance Patterns

### Pattern 1: Use model_validate_json() Not json.loads() (BEST PRACTICE)

**Status**: ✅ **NOT A PROBLEM IN FLEXT-CORE** - Pattern does not currently exist in codebase
**Reference**: `/flext/docs/references/pydantic2/concepts/performance.md`

**Anti-Pattern to Avoid** (SLOW - Two steps):
```python
import json
data = json.loads(json_string)  # Python JSON parser
model = MyModel.model_validate(data)  # Then validate
```

**Best Practice** (FAST - One step):
```python
model = MyModel.model_validate_json(json_string)  # Rust parser + validate
```

**Why It Matters**: Pydantic v2's Rust-based pydantic-core is 10-50x faster than Python's `json.loads()`

**Verification**:
```bash
cd flext-core
grep -rn "json\.loads" src/
# Result (2025-01-21): 0 matches - pattern does not exist ✅
```

**Action**: Keep this pattern in mind for **future development** - if you need to parse JSON into Pydantic models, use `model_validate_json()` directly.

---

### Pattern 2: TypeAdapter at Module Level

**Problem** (Created every call):
```python
def validate_items(data: list[dict]) -> list[Item]:
    adapter = TypeAdapter(list[Item])  # ❌ Created EVERY call
    return adapter.validate_python(data)
```

**Solution** (Created once):
```python
from typing import Final
_ITEM_ADAPTER: Final = TypeAdapter(list[Item])  # ✅ Module level

def validate_items(data: list[dict]) -> list[Item]:
    return _ITEM_ADAPTER.validate_python(data)  # Reuse
```

**Audit Command**:
```bash
cd flext-core
grep -rn "TypeAdapter(" src/
```

---

### Pattern 3: Tagged Unions (Discriminated)

**Problem** (Slow - tries each type):
```python
Message = Command | Event | Query  # Sequential validation
```

**Solution** (Fast - checks discriminator):
```python
from pydantic import Discriminator
from typing import Annotated, Literal

class Command(BaseModel):
    type: Literal["command"] = "command"
    
class Event(BaseModel):
    type: Literal["event"] = "event"

class Query(BaseModel):
    type: Literal["query"] = "query"

Message = Annotated[
    Command | Event | Query,
    Discriminator("type")  # Checks 'type' field first
]
```

**Audit Command**:
```bash
cd flext-core
grep -rn "Union\[" src/
grep -rn " | " src/ | grep "type.*="
```

---

## Section 3.3: Serialization Patterns

### Use model_dump() Modes Correctly

**Pydantic v2 has TWO modes**:
1. **Python mode**: Preserves Python types (tuple, set, etc.)
2. **JSON mode**: Only JSON-compatible types (list, array, etc.)

**Example**:
```python
class Event(BaseModel):
    values: tuple[int, ...]

event = Event(values=(1, 2, 3))

# Python mode
python_dict = event.model_dump()
# {'values': (1, 2, 3)}  # Tuple preserved

# JSON mode
json_dict = event.model_dump(mode='json')
# {'values': [1, 2, 3]}  # Converted to list
```

**Use JSON mode when**:
- Sending to REST API
- Saving to JSON file
- Sending to JavaScript frontend

**Use Python mode when**:
- Internal processing
- Passing to other Python code
- Preserving exact Python types

---

### Use @field_serializer for Custom Serialization

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class Event(BaseModel):
    timestamp: datetime
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()  # Always ISO format

event = Event(timestamp=datetime.now())
print(event.model_dump())
# {'timestamp': '2025-01-21T10:30:00.123456'}
```

**Audit Command**:
```bash
cd flext-core
grep -rn "to_dict\|to_json\|serialize" src/flext_core/models.py
```

---

## Section 3.4: Advanced Validator Patterns

### Pydantic v2 Validator Types (Execution Order)

1. **Before Validators**: Run BEFORE Pydantic (coercion)
2. **Plain Validators**: REPLACE Pydantic validation
3. **After Validators**: Run AFTER Pydantic (most common)
4. **Wrap Validators**: WRAP around Pydantic (advanced)

### Current Usage: Decorator Pattern

**Current** (29 occurrences in config.py and models.py):
```python
class MyModel(BaseModel):
    username: str
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("Alphanumeric only")
        return v.lower()
```

### Recommended: Annotated Pattern (Reusable)

**Better** (reusable across ecosystem):
```python
# In typings.py - ONE definition
def _validate_alphanumeric_lower(value: str) -> str:
    if not value.isalnum():
        raise ValueError("Alphanumeric only")
    return value.lower()

AlphanumericLowerStr = Annotated[
    str,
    Field(min_length=1, max_length=50),
    AfterValidator(_validate_alphanumeric_lower),
]

# In models - REUSE everywhere
class UserModel(BaseModel):
    username: AlphanumericLowerStr  # Validated automatically

class AccountModel(BaseModel):
    account_name: AlphanumericLowerStr  # Same validation, zero duplication
```

**Benefits**:
- ✅ Reusable across all 32+ FLEXT projects
- ✅ Centralized logic (DRY principle)
- ✅ Self-documenting types
- ✅ Test once, use everywhere

---

## Implementation Checklist

### Phase 1: Preparation (Day 1)
- [ ] Backup current codebase
- [ ] Create feature branch
- [ ] Read all of Part 3
- [ ] Understand Pydantic v2 patterns

### Phase 2: Expand typings.py (Day 1-2)
- [ ] Add PortNumber type
- [ ] Add TimeoutSeconds type
- [ ] Add RetryCount type
- [ ] Add NonEmptyStr type
- [ ] Add LogLevel type
- [ ] Add HostName type
- [ ] Add module-level exports
- [ ] Run `make lint && make type-check`

### Phase 3: Deprecate utilities.py Methods (Day 2-3)
- [ ] Add deprecation warnings to 14 methods
- [ ] Update each to use Pydantic internally
- [ ] Document migration path in docstrings
- [ ] Run `make test` (should still pass)

### Phase 4: Update Internal Usage (Day 3-4)
- [ ] Find all validate_port() calls → use PortNumber
- [ ] Find all validate_email() calls → use EmailStr
- [ ] Find all validate_url() calls → use HttpUrl
- [ ] Continue for all 14 methods
- [ ] Run `make validate` after each change

### Phase 5: Performance Optimizations (Day 4-5)
- [x] ~~Audit for json.loads() + model_validate()~~ - Pattern does NOT exist in codebase ✅
- [ ] Find TypeAdapter in functions → move to module level (if pattern exists)
- [ ] Audit unions → add Discriminator where appropriate (if applicable)
- [ ] Run performance benchmarks (before/after) if optimizations applied

### Phase 6: Serialization Review (Day 5)
- [ ] Audit model_dump() calls → verify correct mode
- [ ] Find custom serialization → migrate to @field_serializer
- [ ] Test JSON API responses

### Phase 7: Validator Refactoring (Day 5-6)
- [ ] Audit 29 @field_validator usages
- [ ] Identify reusable patterns
- [ ] Create Annotated types for common patterns
- [ ] Migrate validators to Annotated types
- [ ] Document in PYDANTIC_V2_PATTERNS.md

### Phase 8: Testing & Verification (Day 6-7)
- [ ] Run full test suite: `make test`
- [ ] Run type checking: `make type-check`
- [ ] Run linting: `make lint`
- [ ] Run security: `make security`
- [ ] Full validation: `make validate`
- [ ] Performance benchmarks
- [ ] Code review

---

## Success Criteria

After completing Part 3:
- ✅ **~270 lines removed** from utilities.py (verified count)
- ✅ **typings.py expanded** with domain types
- ✅ **100% Pydantic v2 patterns** (no duplication)
- ✅ **Performance improved** (if TypeAdapter/Union patterns exist - verify first)
- ✅ **All quality gates pass** (`make validate`)
- ✅ **Deprecation warnings** in place (2-version cycle)

---

## Next Steps

After completing Part 3:
1. ✅ Verify all checklist items complete
2. ✅ Run full validation pipeline
3. ➡️ Proceed to Part 4: [Test Fixes](./04-test-fixes.md)

---

**Time Estimate**: 8-12 hours (can be done over 1-2 weeks with other work)
