# Python Coding Standards

<!-- TOC START -->

- [Type Annotations](#type-annotations)
  - [ZERO TOLERANCE: Complete Type Annotations](#zero-tolerance-complete-type-annotations)
  - [Type Hints for Collections](#type-hints-for-collections)
  - [Type Variables for Generics](#type-variables-for-generics)
- [Code Style](#code-style)
  - [Line Length: 79 Characters (Strict)](#line-length-79-characters-strict)
  - [Imports: Organize by Category](#imports-organize-by-category)
  - [Naming Conventions](#naming-conventions)
- [Documentation](#documentation)
  - [Docstrings: Google Style](#docstrings-google-style)
  - [Module-Level Docstrings](#module-level-docstrings)
- [Error Handling](#error-handling)
  - [Railway-Oriented Always](#railway-oriented-always)
  - [No Bare Except](#no-bare-except)
- [Testing](#testing)
  - [Test Naming](#test-naming)
  - [One Assertion Per Test (Mostly)](#one-assertion-per-test-mostly)
- [Linting Rules](#linting-rules)
  - [Ruff: Zero Violations](#ruff-zero-violations)
  - [Type Checking: Pyrefly Strict](#type-checking-pyrefly-strict)
  - [Format with Ruff](#format-with-ruff)
- [Specific Rules](#specific-rules)
  - [No Mutable Default Arguments](#no-mutable-default-arguments)
  - [No Global State](#no-global-state)
  - [Comprehensions Over Loops](#comprehensions-over-loops)
  - [Context Managers for Resources](#context-managers-for-resources)
- [Quality Gate Checklist](#quality-gate-checklist)
- [Common Violations and Fixes](#common-violations-and-fixes)
- [Philosophy](#philosophy)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

FLEXT-Core enforces **zero-tolerance** quality standards. These are non-negotiable.

## Type Annotations

### ZERO TOLERANCE: Complete Type Annotations

**Every function, parameter, and return must have types:**

````python
# ✅ CORRECT
def get_user(user_id: str) -> FlextResult[User]:
    """Get user by ID."""
    pass

def process_items(items: list[str], multiplier: int = 1) -> list[int]:
    """Process items."""
    return [len(item) * multiplier for item in items]

# ❌ WRONG - Any of these:
def get_user(user_id):  # Missing parameter type
    pass

def get_user(user_id: str):  # Missing return type
    pass

def get_user(user_id: str) -> None:  # Wrong return type (should be FlextResult)
    pass

def get_user(user_id: str) -> Any:  # NO Any type
    pass
```text

### Type Hints for Collections

```python
# ✅ CORRECT - Be specific
def process_users(users: list[User]) -> dict[str, User]:
    pass

def get_first_or_default(items: list[str], default: str) -> str:
    pass

# ❌ WRONG - Vague
def process_users(users: list):  # Missing element type
    pass

def process_users(users: List[User]):  # Use lowercase list not List
    pass
```text

### Type Variables for Generics

```python
from typing import TypeVar

T = TypeVar('T')
U = TypeVar('U')

# ✅ CORRECT
def map_over(items: list[T], func: callable[[T], U]) -> list[U]:
    """Map function over items."""
    return [func(item) for item in items]

# ❌ WRONG
def map_over(items, func):  # No types
    pass
```text

## Code Style

### Line Length: 79 Characters (Strict)

**Flext-Core enforces 79 character maximum** (PEP 8 strict):

```python
# ✅ CORRECT - 79 chars or less
def very_long_function_name(
    parameter_one: str,
    parameter_two: int,
    parameter_three: bool,
) -> FlextResult[dict]:
    """Function with many parameters."""
    pass

# ❌ WRONG - Exceeds 79 chars
def very_long_function_name(parameter_one: str, parameter_two: int, parameter_three: bool) -> FlextResult[dict]:
    pass
```text

### Imports: Organize by Category

```python
# ✅ CORRECT - Three groups: stdlib, third-party, local
import os
import sys
from datetime import datetime
from typing import TypeVar

import pydantic
import structlog

from flext_core import FlextResult, FlextContainer

# ❌ WRONG - Mixed order
from flext_core import FlextResult
import os
import pydantic
from datetime import datetime
from flext_core import FlextContainer
```text

### Naming Conventions

```python
# ✅ CORRECT
class FlextResult:  # PascalCase for classes
    """Result monad."""

    def unwrap(self) -> Any:  # snake_case for methods
        pass

MAXIMUM_RETRIES = 3  # SCREAMING_SNAKE_CASE for constants

variable_name = "value"  # snake_case for variables

# ❌ WRONG
class flextResult:  # lowercase for class
    pass

def UnwrapValue():  # PascalCase for function
    pass

max_retries = 3  # lowercase for constants

VariableName = "value"  # PascalCase for variable
```text

## Documentation

### Docstrings: Google Style

```python
# ✅ CORRECT - Google style docstring
def create_user(
    name: str,
    email: str,
    age: int,
) -> FlextResult[User]:
    """Create new user with validation.

    Args:
        name: User's full name (min 1 char).
        email: User's email address (must be valid).
        age: User's age (must be >= 18).

    Returns:
        FlextResult[User]: Success with created user or failure with error.

    Raises:
        None: Uses FlextResult for errors.

    Example:
        >>> result = create_user("Alice", "alice@example.com", 25)
        >>> if result.is_success:
        ...     user = result.value
        ...     print(user.name)
    """
    # Implementation
    pass

# ❌ WRONG - Missing docstring
def create_user(name: str, email: str, age: int) -> FlextResult[User]:
    pass

# ❌ WRONG - Poor docstring
def create_user(name, email, age):
    """Creates a user."""  # Too vague
    pass
```text

### Module-Level Docstrings

```python
# ✅ CORRECT - Module docstring at top
"""User domain models and services.

This module provides:
- User entity with validation
- UserService for user operations
- User-related exceptions
"""

from flext_core import FlextModels, FlextService


class User(FlextModels.Entity):
    """User entity."""
    pass
```text

## Error Handling

### Railway-Oriented Always

```python
# ✅ CORRECT - Return FlextResult
def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult[str].fail("Invalid email")
    return FlextResult[str].ok(email)

# ❌ WRONG - Raising exceptions
def validate_email(email: str) -> str:
    if "@" not in email:
        raise ValueError("Invalid email")  # NO!
    return email
```text

### No Bare Except

```python
# ✅ CORRECT - Catch specific exceptions
try:
    result = operation()
except TimeoutError:
    return FlextResult.fail("Operation timeout")
except ValueError as e:
    return FlextResult.fail(f"Invalid input: {e}")

# ❌ WRONG - Bare except catches everything
try:
    result = operation()
except:  # NO! Catches KeyboardInterrupt, SystemExit, etc.
    pass

# ❌ WRONG - Too broad
except Exception:
    pass  # Still too vague
```text

## Testing

### Test Naming

```python
# ✅ CORRECT - Clear, descriptive
def test_create_user_succeeds_with_valid_data():
    pass

def test_create_user_fails_when_name_is_empty():
    pass

def test_email_validation_accepts_valid_format():
    pass

# ❌ WRONG - Vague
def test_user():
    pass

def test_create():
    pass

def test_validate():
    pass
```text

### One Assertion Per Test (Mostly)

```python
# ✅ CORRECT - One conceptual assertion
def test_result_ok_creates_success():
    result = FlextResult[int].ok(42)
    assert result.is_success

def test_result_ok_contains_value():
    result = FlextResult[int].ok(42)
    assert result.value == 42

# ❌ WRONG - Multiple concepts
def test_result():
    result = FlextResult[int].ok(42)
    assert result.is_success
    assert result.value == 42
    assert result.data == 42
    assert result.value == 42
```text

## Linting Rules

### Ruff: Zero Violations

**Run before every commit:**

```bash
# ✅ Must pass
ruff check src/

# ❌ Must NOT exist
# Output should be empty (no violations)
```text

### Type Checking: Pyrefly Strict

```bash
# ✅ Must pass
PYTHONPATH=src pyrefly check src/

# ❌ Must NOT have errors
# Should show: 0 errors
```text

### Format with Ruff

```bash
# Format code automatically
ruff format src/

# Check formatting
ruff format --check src/
```text

## Specific Rules

### No Mutable Default Arguments

```python
# ✅ CORRECT
def add_user(
    users: list[User],
    user: User,
) -> list[User]:
    new_users = users.copy()
    new_users.append(user)
    return new_users

# ❌ WRONG - Mutable default
def add_user(users: list[User] = []) -> list[User]:
    users.append(user)  # Changes default!
    return users
```text

### No Global State

```python
# ✅ CORRECT - Use dependency injection
class UserService:
    def __init__(self, logger):
        self.logger = logger

# ❌ WRONG - Global state
_logger = None

class UserService:
    def get_logger(self):
        global _logger
        return _logger
```text

### Comprehensions Over Loops

```python
# ✅ PREFERRED - Comprehension
even_numbers = [x for x in numbers if x % 2 == 0]

# ✅ ACCEPTABLE - When complex
result = []
for user in users:
    if user.is_active:
        result.append(transform(user))

# ❌ AVOID - Simple loops
for x in numbers:
    if x % 2 == 0:
        print(x)
```text

### Context Managers for Resources

```python
# ✅ CORRECT - Use context manager
with open('file.txt', 'r') as f:
    content = f.read()

# ❌ WRONG - Manual cleanup
f = open('file.txt', 'r')
content = f.read()
# File might not close if exception
```text

## Quality Gate Checklist

Before committing, verify:

- [ ] **Types**: `PYTHONPATH=src pyrefly check src/` - Zero errors
- [ ] **Linting**: `ruff check src/` - Zero violations
- [ ] **Format**: `ruff format --check src/` - Passes
- [ ] **Tests**: `pytest tests/` - All pass
- [ ] **Coverage**: 75%+ minimum
- [ ] **Line Length**: All lines ≤ 79 chars

```bash
# Run all checks at once
make validate  # In flext-core directory
```text

## Common Violations and Fixes

| Violation             | Issue                 | Fix                          |
| --------------------- | --------------------- | ---------------------------- |
| `Type is: Any`        | Using Any type        | Use specific type or TypeVar |
| `Unused import`       | Dead code             | Remove unused import         |
| `Line too long`       | Exceeds 79 chars      | Break into multiple lines    |
| `Missing return type` | Incomplete annotation | Add `-> ReturnType`          |
| `Bare except`         | Too broad             | Catch specific exception     |
| `Mutable argument`    | Default arg mutation  | Use None, copy data          |
| `Missing docstring`   | Undocumented          | Add Google-style docstring   |

## Philosophy

**FLEXT-Core enforces strict standards not for perfectionism, but for:**

- ✅ **Maintainability**: Consistent code is easy to understand
- ✅ **Reliability**: Type safety catches errors early
- ✅ **Scalability**: 32+ dependent projects rely on this
- ✅ **Quality**: Zero compromise on production readiness

**Standards apply to:**

- ✅ `src/` - All source code
- ✅ `tests/` - All test code
- ✅ Every pull request
- ✅ No exceptions

**If a standard seems wrong, raise an issue.** Standards can evolve, but when they exist, they apply universally.
````
