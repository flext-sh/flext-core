# FLEXT Core Advanced Types

**Reality-Based Type System Documentation**

FLEXT Core's type system is based on the actual implementation in `src/flext_core/typings.py` and related modules.

## 📊 Current Type System Status

**ACTUAL IMPLEMENTATION**: Centralized in `typings.py` (single source of truth). A thin `types.py` remains as compatibility re-export.

## 🏗️ FlextTypes Hierarchical Structure

Based on the actual `typings.py` implementation:

```python
from flext_core.typings import FlextTypes

# Access hierarchical type categories
core_types = FlextTypes.Core
domain_types = FlextTypes.Domain
cqrs_types = FlextTypes.CQRS
data_types = FlextTypes.Data
infra_types = FlextTypes.Infrastructure
singer_types = FlextTypes.Singer
protocols = FlextTypes.Protocols
```

## 🔧 Available Protocols

### Comparable Protocol

For objects that can be compared:

```python
from flext_core.typings import FlextTypes

class User:
    def __init__(self, age: int):
        self.age = age

    def __lt__(self, other: object) -> bool:
        if isinstance(other, User):
            return self.age < other.age
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, User):
            return self.age <= other.age
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, User):
            return self.age > other.age
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, User):
            return self.age >= other.age
        return NotImplemented

# User now satisfies FlextTypes.Protocols.Comparable
```

### Serializable Protocol

For objects that can be serialized:

```python
from flext_core.typings import FlextTypes
import json

class Config:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# Config now satisfies FlextTypes.Protocols.Serializable
```

### Validatable Protocol

For objects with validation:

```python
from flext_core.types import FlextTypes

class EmailAddress:
    def __init__(self, email: str):
        self.email = email

    def validate(self) -> object:
        if "@" not in self.email:
            raise ValueError("Invalid email format")
        return self

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

# EmailAddress now satisfies FlextTypes.Protocols.Validatable
```

### Timestamped Protocol

For objects with timestamps:

```python
from flext_core.types import FlextTypes
from datetime import datetime

class AuditableEntity:
    def __init__(self):
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

# AuditableEntity satisfies FlextTypes.Protocols.Timestamped
```

## 🔄 Type System Migration

### Current Migration Status

Based on the actual code comments:

```python
# CURRENT (typings.py) - hierarchical system
from flext_core.typings import FlextTypes
```

### Migration Example

```python
# Old approach (being phased out)
from flext_core.types import TData, TEntity

def process_old(data: TData) -> TEntity:
    pass

# Current approach (types.py)
from flext_core.types import FlextTypes

def process_current(data: FlextTypes.Data.Input) -> FlextTypes.Domain.Entity:
    pass

# Future approach (semantic_types.py - when available)
from flext_core.semantic_types import FlextTypes

def process_future(data: FlextTypes.Data.Connection) -> FlextTypes.Core.Predicate:
    pass
```

## 🎯 Core TypeVars Available

From the actual `types.py` implementation:

```python
from flext_core.typings import T, U, V, R, E, P, F

# Generic programming with actual TypeVars
def transform_data(data: T, transformer: Callable[[T], U]) -> U:
    return transformer(data)

def handle_result(result: R, error_handler: Callable[[E], R]) -> R:
    # Result handling pattern
    pass

def process_payload(payload: P) -> R:
    # Payload processing
    pass
```

## 💡 Practical Usage Patterns

### Type-Safe Data Processing

```python
from flext_core import FlextResult
from flext_core.typings import T, U

def safe_transform(data: T, func: Callable[[T], U]) -> FlextResult[U]:
    """Type-safe transformation with error handling."""
    try:
        result = func(data)
        return FlextResult[None].ok(result)
    except Exception as e:
        return FlextResult[None].fail(f"Transform failed: {e}")

# Usage
def double_number(x: int) -> int:
    return x * 2

result = safe_transform(5, double_number)  # FlextResult[int]
```

### Protocol-Based Design

```python
from flext_core.typings import FlextTypes
from typing import TypeVar

T_Comparable = TypeVar('T_Comparable', bound=FlextTypes.Protocols.Comparable)

def find_max(items: list[T_Comparable]) -> T_Comparable | None:
    """Find maximum item using Comparable protocol."""
    if not items:
        return None

    max_item = items[0]
    for item in items[1:]:
        if item > max_item:
            max_item = item
    return max_item
```

## ⚠️ Current Limitations

Based on actual implementation analysis:

1. **Migration in Progress**: Multiple type modules exist simultaneously
2. **Incomplete Documentation**: Some type categories may be placeholders
3. **API Instability**: Type system is actively being refactored
4. **Limited Advanced Types**: Complex types like `Either`, `Maybe` don't exist yet

## 🔍 What's Actually Available

### Confirmed Available (from actual code)

```python
# Core TypeVars
from flext_core.types import T, U, V, R, E, P, F

# Hierarchical type access
from flext_core.types import FlextTypes
types = FlextTypes.Core      # ✅ Exists
types = FlextTypes.Domain    # ✅ Exists
types = FlextTypes.CQRS      # ✅ Exists
types = FlextTypes.Data      # ✅ Exists

# Protocols
FlextTypes.Protocols.Comparable    # ✅ Exists
FlextTypes.Protocols.Serializable  # ✅ Exists
FlextTypes.Protocols.Validatable   # ✅ Exists
FlextTypes.Protocols.Timestamped   # ✅ Exists
```

### NOT Available (common misconceptions)

```python
# These DON'T exist in current implementation:
from flext_core import Either        # ❌ Not implemented
from flext_core import Pipe          # ❌ Not implemented
from flext_core import Repository    # ❌ Not implemented
from flext_core import Factory       # ❌ Not implemented
```

## 📚 Recommended Usage

For new development, use the hierarchical FlextTypes system:

```python
from flext_core import FlextResult
from flext_core.types import FlextTypes, T

class DataProcessor:
    """Example using actual available types."""

    def process(self, data: T) -> FlextResult[T]:
        """Process data with type safety."""
        if not data:
            return FlextResult[None].fail("Empty data provided")

        return FlextResult[None].ok(data)

    def validate_serializable(self, obj: FlextTypes.Protocols.Serializable) -> bool:
        """Validate object can be serialized."""
        try:
            obj.to_dict()
            return True
        except Exception:
            return False
```

---

**This documentation reflects the ACTUAL type system implementation in FLEXT Core as of the current version. For the most up-to-date information, always refer to the source code in `src/flext_core/types.py`.**
