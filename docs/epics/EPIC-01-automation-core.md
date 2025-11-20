# EPIC-01: Automation Core + Helpers

**Phase**: 1
**Duration**: Week 1 (5 days)
**Risk**: 🟢 Low
**LOC Impact**: -50 to -100
**Dependencies**: EPIC-00 (baseline complete)
**Status**: 🟡 WAITING ON EPIC-00

---

## 🎯 OBJECTIVE

Create the **foundation** for all automation features:
1. Central `automation.py` module with 4 core approaches
2. Enhanced `FlextMixins` with reusable conversion helpers
3. Zero duplication of isinstance/conversion patterns

**Why This Matters**: Every other phase depends on these helpers. Get this right, and everything else becomes easier.

---

## 📋 TASKS CHECKLIST

### Task 1.1: Create `automation.py` Module

- [ ] Create `src/flext_core/automation.py`
- [ ] Implement 4 core functions:
  - `coerce_model()` - Universal model coercion
  - `autoschema()` - Decorator for auto-kwargs→model
  - `create_typed_dict()` - Generate TypedDict from BaseModel
  - `generate_dual_api_overloads()` - Code generation helper
- [ ] Add comprehensive docstrings with examples
- [ ] Add unit tests for each function

**Implementation**:

```python
"""
Automation utilities for quad API support (model/dict/kwargs/hybrid).

This module provides the core building blocks for implementing flexible APIs
that accept configuration in multiple formats without duplicating validation logic.
"""

from __future__ import annotations

from typing import Any, TypeVar, Type, Callable, overload, get_type_hints
from functools import wraps
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
F = TypeVar("F", bound=Callable[..., Any])


def coerce_model(
    model_class: Type[T],
    options: T | dict[str, Any] | None = None,
    **kwargs: Any,
) -> T:
    """
    Universal model coercion supporting quad API.

    Accepts:
    1. Model instance (passthrough)
    2. Dict (convert to model)
    3. kwargs (construct model)
    4. Hybrid (merge options dict + kwargs)

    Args:
        model_class: Target Pydantic model class
        options: Pre-existing model or dict
        **kwargs: Additional fields (merged with options if dict)

    Returns:
        Validated model instance

    Raises:
        ValidationError: If coercion fails

    Examples:
        >>> opts = coerce_model(RetryOptions, max_attempts=3)
        >>> opts = coerce_model(RetryOptions, {"max_attempts": 3})
        >>> existing = RetryOptions(max_attempts=3)
        >>> opts = coerce_model(RetryOptions, existing)  # passthrough
        >>> opts = coerce_model(RetryOptions, {"max_attempts": 3}, backoff=2.0)
    """
    # Case 1: Already a model instance
    if isinstance(options, model_class):
        if kwargs:
            # Hybrid: model + kwargs (create new instance with overrides)
            data = options.model_dump()
            data.update(kwargs)
            return model_class(**data)
        return options

    # Case 2: Dict or hybrid dict+kwargs
    if isinstance(options, dict):
        data = {**options, **kwargs}
        return model_class(**data)

    # Case 3: Pure kwargs
    if options is None:
        return model_class(**kwargs)

    # Case 4: Invalid type
    raise TypeError(
        f"options must be {model_class.__name__}, dict, or None; "
        f"got {type(options).__name__}"
    )


def autoschema(model_param: str = "options") -> Callable[[F], F]:
    """
    Decorator to automatically convert kwargs to Pydantic model.

    The decorated function signature should have:
    - A parameter with type annotation of BaseModel subclass
    - This parameter named as `model_param` (default: "options")

    Args:
        model_param: Name of the parameter to convert (default: "options")

    Returns:
        Decorator that performs automatic coercion

    Examples:
        >>> @autoschema()
        ... def process(data: str, options: RetryOptions) -> str:
        ...     print(f"Max attempts: {options.max_attempts}")
        ...     return data
        >>>
        >>> # All these work:
        >>> process("test", RetryOptions(max_attempts=3))
        >>> process("test", {"max_attempts": 3})
        >>> process("test", max_attempts=3)
    """

    def decorator(func: F) -> F:
        hints = get_type_hints(func)
        if model_param not in hints:
            raise ValueError(
                f"Parameter '{model_param}' not found in function signature"
            )

        model_class = hints[model_param]
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise ValueError(
                f"Parameter '{model_param}' must be annotated with "
                f"BaseModel subclass, got {model_class}"
            )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract the model parameter if present
            if model_param in kwargs:
                kwargs[model_param] = coerce_model(
                    model_class,
                    kwargs.get(model_param),
                    **{k: v for k, v in kwargs.items() if k != model_param},
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def create_typed_dict(model_class: Type[BaseModel], name: str | None = None) -> type:
    """
    Generate a TypedDict class from a Pydantic model.

    Useful for maintaining dict-based APIs while gaining type safety.

    Args:
        model_class: Source Pydantic model
        name: Name for generated TypedDict (default: f"{model_class.__name__}Dict")

    Returns:
        TypedDict class matching model fields

    Examples:
        >>> RetryOptionsDict = create_typed_dict(RetryOptions)
        >>> config: RetryOptionsDict = {"max_attempts": 3, "backoff": 2.0}
    """
    from typing_extensions import TypedDict

    if name is None:
        name = f"{model_class.__name__}Dict"

    fields = {}
    for field_name, field_info in model_class.model_fields.items():
        fields[field_name] = field_info.annotation

    return TypedDict(name, fields)  # type: ignore


def generate_dual_api_overloads(model_class: Type[BaseModel]) -> str:
    """
    Generate overload signatures for quad API pattern.

    This is a code generation helper for maintaining consistent
    overload signatures across decorators and functions.

    Args:
        model_class: Model class to generate overloads for

    Returns:
        String containing overload code (ready to paste)

    Examples:
        >>> print(generate_dual_api_overloads(RetryOptions))
        @overload
        def retry(options: RetryOptions) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
        @overload
        def retry(options: dict[str, Any]) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
        @overload
        def retry(**kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
    """
    model_name = model_class.__name__
    lines = [
        "@overload",
        f"def FUNC_NAME(options: {model_name}) -> RETURN_TYPE: ...",
        "@overload",
        "def FUNC_NAME(options: dict[str, Any]) -> RETURN_TYPE: ...",
        "@overload",
        "def FUNC_NAME(**kwargs: Any) -> RETURN_TYPE: ...",
        "",
        "def FUNC_NAME(",
        f"    options: {model_name} | dict[str, Any] | None = None,",
        "    **kwargs: Any,",
        ") -> RETURN_TYPE:",
        f'    """Quad API: accepts {model_name}, dict, or kwargs."""',
        f"    opts = coerce_model({model_name}, options, **kwargs)",
        "    # Implementation here",
    ]
    return "\n".join(lines)
```

**Tests** (`tests/unit/test_automation.py`):
```python
import pytest
from pydantic import BaseModel, Field
from flext_core.automation import coerce_model, autoschema


class MockOptions(BaseModel):
    value: int = Field(ge=0)
    name: str = "default"


class TestCoerceModel:
    def test_passthrough_model(self):
        opts = MockOptions(value=5, name="test")
        result = coerce_model(MockOptions, opts)
        assert result is opts

    def test_dict_to_model(self):
        result = coerce_model(MockOptions, {"value": 5, "name": "test"})
        assert isinstance(result, MockOptions)
        assert result.value == 5
        assert result.name == "test"

    def test_kwargs_only(self):
        result = coerce_model(MockOptions, value=5, name="test")
        assert result.value == 5
        assert result.name == "test"

    def test_hybrid_dict_plus_kwargs(self):
        result = coerce_model(MockOptions, {"value": 5}, name="override")
        assert result.value == 5
        assert result.name == "override"

    def test_hybrid_model_plus_kwargs(self):
        opts = MockOptions(value=5, name="original")
        result = coerce_model(MockOptions, opts, name="override")
        assert result.value == 5
        assert result.name == "override"

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="must be MockOptions, dict, or None"):
            coerce_model(MockOptions, "invalid")  # type: ignore


class TestAutoschema:
    def test_autoschema_with_kwargs(self):
        @autoschema()
        def process(data: str, options: MockOptions) -> int:
            return options.value

        result = process("test", value=10)
        assert result == 10

    def test_autoschema_with_dict(self):
        @autoschema()
        def process(data: str, options: MockOptions) -> int:
            return options.value

        result = process("test", options={"value": 15})
        assert result == 15

    def test_autoschema_with_model(self):
        @autoschema()
        def process(data: str, options: MockOptions) -> int:
            return options.value

        opts = MockOptions(value=20)
        result = process("test", options=opts)
        assert result == 20
```

---

### Task 1.2: Enhance `FlextMixins.ModelConversion`

- [ ] Add `to_dict()` static method
- [ ] Add `from_dict()` generic method
- [ ] Add `merge_models()` method for combining models
- [ ] Replace all duplicated isinstance patterns with helper usage
- [ ] Add unit tests

**Implementation** (add to `src/flext_core/mixins.py`):

```python
class ModelConversion:
    """Helpers for model ↔ dict conversions."""

    @staticmethod
    def to_dict(
        obj: BaseModel | dict[str, object] | None
    ) -> dict[str, object]:
        """
        Convert model or dict to dict, handling None.

        Args:
            obj: Model instance, dict, or None

        Returns:
            Dictionary representation

        Examples:
            >>> data = ModelConversion.to_dict(my_model)
            >>> data = ModelConversion.to_dict({"key": "value"})
            >>> data = ModelConversion.to_dict(None)  # returns {}
        """
        if obj is None:
            return {}
        return obj.model_dump() if isinstance(obj, BaseModel) else obj

    @staticmethod
    def from_dict[T: BaseModel](
        model_class: type[T],
        data: dict[str, Any] | T | None,
    ) -> T | None:
        """
        Convert dict to model, handling None and passthrough.

        Args:
            model_class: Target model class
            data: Dict, model instance, or None

        Returns:
            Model instance or None

        Examples:
            >>> model = ModelConversion.from_dict(MyModel, {"field": "value"})
            >>> model = ModelConversion.from_dict(MyModel, existing_model)  # passthrough
            >>> model = ModelConversion.from_dict(MyModel, None)  # returns None
        """
        if data is None:
            return None
        if isinstance(data, model_class):
            return data
        return model_class(**data) if isinstance(data, dict) else None

    @staticmethod
    def merge_models[T: BaseModel](
        base: T,
        overrides: dict[str, Any] | T | None,
    ) -> T:
        """
        Merge two models or model + dict, returning new instance.

        Args:
            base: Base model instance
            overrides: Overrides as dict or model

        Returns:
            New model instance with merged data

        Examples:
            >>> merged = ModelConversion.merge_models(base_opts, {"timeout": 60})
            >>> merged = ModelConversion.merge_models(opts1, opts2)
        """
        base_data = base.model_dump()

        if isinstance(overrides, BaseModel):
            override_data = overrides.model_dump()
        elif isinstance(overrides, dict):
            override_data = overrides
        else:
            override_data = {}

        merged_data = {**base_data, **override_data}
        return type(base)(**merged_data)
```

---

### Task 1.3: Enhance `FlextMixins.ResultHandling`

- [ ] Add `ensure_result()` method
- [ ] Add `wrap_exception()` method
- [ ] Replace all `if not isinstance(result, FlextResult)` patterns
- [ ] Add unit tests

**Implementation**:

```python
class ResultHandling:
    """Helpers for FlextResult operations."""

    @staticmethod
    def ensure_result[T](value: T | FlextResult[T]) -> FlextResult[T]:
        """
        Ensure value is wrapped in FlextResult.

        Args:
            value: Raw value or FlextResult

        Returns:
            FlextResult instance

        Examples:
            >>> result = ResultHandling.ensure_result(42)  # wraps in .ok()
            >>> result = ResultHandling.ensure_result(FlextResult.ok(42))  # passthrough
        """
        return (
            value
            if isinstance(value, FlextResult)
            else FlextResult[T].ok(value)
        )

    @staticmethod
    def wrap_exception[T](
        exc: Exception,
        context: dict[str, Any] | None = None,
    ) -> FlextResult[T]:
        """
        Wrap exception in FlextResult.failed().

        Args:
            exc: Exception to wrap
            context: Additional context

        Returns:
            FlextResult.failed() with exception

        Examples:
            >>> result = ResultHandling.wrap_exception(ValueError("bad"))
        """
        return FlextResult[T].failed(
            error=exc,
            message=str(exc),
            context=context or {},
        )
```

---

### Task 1.4: Replace Duplication in Codebase

- [ ] Search for `isinstance(*, BaseModel)` + `model_dump()` patterns
- [ ] Replace with `ModelConversion.to_dict()`
- [ ] Search for `if not isinstance(result, FlextResult)` patterns
- [ ] Replace with `ResultHandling.ensure_result()`
- [ ] Run tests after each replacement batch

**Search & Replace Commands**:

```bash
# Find all isinstance + model_dump patterns
grep -rn "if isinstance.*BaseModel.*model_dump" src/

# Find all isinstance result checks
grep -rn "if not isinstance.*FlextResult" src/

# Manual replacement (too complex for sed)
# Use IDE refactoring or careful manual edits
```

---

### Task 1.5: Update Imports

- [ ] Add `from flext_core.automation import coerce_model` where needed
- [ ] Update `FlextMixins` imports to include new helpers
- [ ] Verify no circular imports introduced
- [ ] Run `python scripts/detect_cycles.py`

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] `automation.py` module created with 4 functions
- [ ] `FlextMixins` enhanced with conversion + result helpers
- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`make type-check`)
- [ ] Lint passes (`make lint`)
- [ ] No circular dependencies (`python scripts/detect_cycles.py`)
- [ ] At least 10 duplication patterns replaced
- [ ] Coverage maintained (≥79%)
- [ ] PR created: `feat(core): add automation helpers and mixins`

### Validation Checklist

```bash
cd /home/marlonsc/flext/flext-core

# Run quality gates
make lint
make type-check
make test

# Verify no cycles
python scripts/detect_cycles.py

# Check LOC reduction
cloc src/ > docs/metrics/phase1_loc.txt
diff docs/metrics/baseline_loc.txt docs/metrics/phase1_loc.txt

# Expected: -50 to -100 LOC
```

---

## 📊 SUCCESS METRICS

### Quantitative
- LOC reduced by 50-100 lines
- 10+ duplication patterns eliminated
- 0 circular dependencies
- Coverage ≥79%

### Qualitative
- Helpers are intuitive to use
- Code review feedback positive
- No confusion about when to use which helper

---

## 🔗 DEPENDENCIES

### Requires
- **EPIC-00**: Baseline metrics and duplication list

### Blocks
- **EPIC-02**: Models will use `automation.py`
- **EPIC-03**: Exceptions will use helpers
- **EPIC-04**: Decorators will use `coerce_model()`
- **EPIC-05**: Dispatcher will use `ensure_result()`

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Helper API not intuitive | High | Get early feedback from team |
| Regression from replacements | Medium | Replace in small batches, test each |
| Performance overhead | Low | Helpers are thin wrappers, profile if needed |

---

**Prev**: [EPIC-00: Baseline](./EPIC-00-baseline.md)
**Next**: [EPIC-02: Models & Facade](./EPIC-02-models-facade.md)

**Status**: 🟡 WAITING ON EPIC-00
