# Development Guide

**FLEXT-Core Foundation Library Development**
**Date**: September 17, 2025 | **Version**: 0.9.0

---

## Foundation Library Responsibilities

FLEXT-Core serves as the architectural foundation for 45+ FLEXT ecosystem projects. All development must maintain the highest quality standards to ensure ecosystem stability.

**Workspace Development Standards**: [Development Best Practices](../../docs/development/best-practices.md)

---

## Development Environment Setup

### Prerequisites

```bash
# Required tools
python --version     # 3.13+
poetry --version     # Latest
git --version        # 2.0+

# Install development tools
make setup           # Complete environment setup
```

### Project Structure

```
flext-core/
├── src/flext_core/           # Foundation library source
│   ├── result.py             # Railway pattern (869 lines)
│   ├── container.py          # DI container (1,142 lines)
│   ├── config.py             # Configuration (1,710 lines)
│   ├── models.py             # Domain models (351 lines)
│   ├── validations.py        # Validation patterns (1,024 lines)
│   ├── commands.py           # CQRS patterns (1,059 lines)
│   ├── adapters.py           # Type adapters (22 actual, minimal design)
│   └── ...                   # Other foundation modules
├── src/flext_tests/          # Test infrastructure
├── tests/                    # Test suite (2,271 tests)
├── docs/                     # Foundation-specific documentation
└── pyproject.toml            # Project configuration
```

---

## Foundation Development Standards

### Quality Gates (MANDATORY)

All commits must pass these quality gates:

```bash
# Complete validation pipeline
make validate           # All checks (lint + type + test + security)

# Individual checks
make lint              # Ruff linting (zero violations)
make type-check        # MyPy strict mode (zero errors)
make test              # Test suite (84%+ coverage)
make format            # Code formatting
```

### Code Quality Requirements

1. **Type Safety**: MyPy strict mode compliance
   ```bash
   # Must pass with zero errors
   mypy src/ --strict --show-error-codes
   ```

2. **Code Quality**: Ruff linting
   ```bash
   # Must pass with zero violations
   ruff check src/
   ```

3. **Test Coverage**: Maintain 84%+ coverage
   ```bash
   # Current: 84% (6,691 statements, 1,065 missing)
   pytest tests/ --cov=src --cov-report=term
   ```

4. **API Compatibility**: Backward compatibility required
   ```python
   # Both patterns must work (ecosystem dependency)
   result = FlextResult[str].ok("test")
   assert result.value == "test"  # New API
   assert result.data == "test"   # Legacy API (maintain)
   ```

---

## Foundation Architecture Patterns

### 1. Railway-Oriented Programming

FlextResult[T] implementation principles:

```python
# Core implementation pattern
class FlextResult[T]:
    def __init__(self, success: bool, data: T | None, error: str | None):
        self._success = success
        self._data = data
        self._error = error

    @property
    def value(self) -> T:
        """New API - preferred access pattern"""
        return self._data

    @property
    def data(self) -> T:
        """Legacy API - maintain for ecosystem compatibility"""
        return self._data

    def map(self, func: Callable[[T], U]) -> "FlextResult[U]":
        """Functor pattern - transform success values"""
        if self.is_failure:
            return FlextResult[U].fail(self.error)
        return FlextResult[U].ok(func(self.value))

    def flat_map(self, func: Callable[[T], "FlextResult[U]"]) -> "FlextResult[U]":
        """Monad pattern - chain operations"""
        if self.is_failure:
            return FlextResult[U].fail(self.error)
        return func(self.value)
```

### 2. Dependency Injection Container

Singleton pattern implementation:

```python
class FlextContainer:
    _instance: Optional["FlextContainer"] = None
    _services: dict[str, Any] = {}

    @classmethod
    def get_global(cls) -> "FlextContainer":
        """Singleton access pattern"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, key: str, service: Any) -> FlextResult[None]:
        """Service registration with error handling"""
        if key in self._services:
            return FlextResult[None].fail(f"Service '{key}' already registered")

        self._services[key] = service
        return FlextResult[None].ok(None)

    def get(self, key: str) -> FlextResult[Any]:
        """Service retrieval with error handling"""
        if key not in self._services:
            return FlextResult[Any].fail(f"Service '{key}' not found")

        return FlextResult[Any].ok(self._services[key])
```

### 3. Domain Modeling Patterns

DDD implementation with Pydantic integration:

```python
class FlextEntity(BaseModel):
    """Base entity with identity and domain events"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    _domain_events: list[dict] = PrivateAttr(default_factory=list)

    def add_domain_event(self, event_type: str, data: dict) -> None:
        """Record domain events for processing"""
        self._domain_events.append({
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

class FlextValue(BaseModel):
    """Base value object (immutable)"""
    class Config:
        frozen = True  # Immutable
        allow_mutation = False

class FlextAggregateRoot(FlextEntity):
    """Base aggregate root for consistency boundaries"""
    version: int = Field(default=1)

    def mark_events_as_committed(self) -> None:
        """Clear domain events after processing"""
        self._domain_events.clear()
```

---

## Testing Standards

### Test Organization

```
tests/
├── unit/                     # Fast, isolated tests
│   ├── test_result.py        # FlextResult functionality
│   ├── test_container.py     # DI container
│   ├── test_models.py        # Domain models
│   └── ...
├── integration/              # Component interaction tests
├── performance/              # Performance benchmarks
└── conftest.py              # Shared fixtures
```

### Test Quality Standards

1. **Real Functional Tests**: Minimize mocking
   ```python
   # ✅ Preferred - test actual functionality
   def test_result_chain_operations():
       result = (
           FlextResult[int].ok(5)
           .map(lambda x: x * 2)
           .flat_map(lambda x: safe_divide(x, 2))
       )
       assert result.unwrap() == 5

   # ❌ Avoid excessive mocking
   def test_with_too_many_mocks():
       mock1 = Mock()
       mock2 = Mock()
       # ... excessive mocking
   ```

2. **Performance Testing**: Benchmarks for core operations
   ```python
   def test_result_performance(benchmark):
       """Benchmark FlextResult operations"""
       def create_and_chain():
           return (
               FlextResult[int].ok(1)
               .map(lambda x: x + 1)
               .flat_map(lambda x: FlextResult[int].ok(x * 2))
           )

       result = benchmark(create_and_chain)
       assert result.unwrap() == 4
   ```

3. **Property-Based Testing**: Use Hypothesis for edge cases
   ```python
   from hypothesis import given, strategies as st

   @given(st.text(), st.text())
   def test_result_error_handling(data, error_msg):
       result = FlextResult[str].fail(error_msg)
       assert result.is_failure
       assert result.error == error_msg
   ```

---

## Foundation Development Workflow

### Before Making Changes

1. **Understand Ecosystem Impact**
   ```bash
   # Check dependent projects (45+ projects depend on this)
   grep -r "from flext_core import" ../flext-*/
   ```

2. **Run Full Validation**
   ```bash
   make validate          # Must pass before changes
   ```

### Making Changes

1. **Follow Patterns**
   - Use FlextResult[T] for all operations that can fail
   - Maintain both `.value` and `.data` access patterns
   - Add comprehensive type annotations
   - Include domain events for entities

2. **Add Tests First** (TDD approach)
   ```python
   def test_new_functionality():
       # Test the desired behavior first
       result = new_function("input")
       assert result.is_success
       assert result.unwrap() == expected_value
   ```

3. **Implement with Quality**
   ```python
   def new_function(input_data: str) -> FlextResult[str]:
       """New foundation functionality."""
       if not input_data:
           return FlextResult[str].fail("Input required")

       # Business logic
       processed = input_data.upper()
       return FlextResult[str].ok(processed)
   ```

### After Making Changes

1. **Quality Validation**
   ```bash
   make validate          # All quality gates
   make test             # Verify tests pass
   ```

2. **Ecosystem Testing** (for breaking changes)
   ```bash
   # Test impact on dependent projects
   cd ../flext-api && make test
   cd ../flext-cli && make test
   ```

3. **Documentation Updates**
   - Update docstrings for all public APIs
   - Add examples to documentation
   - Update type annotations

---

## API Design Principles

### 1. Backward Compatibility

**CRITICAL**: 45+ projects depend on FLEXT-Core APIs

```python
# ✅ Maintain both access patterns
class FlextResult[T]:
    @property
    def value(self) -> T:
        """New preferred API"""
        return self._data

    @property
    def data(self) -> T:
        """Legacy API - MUST MAINTAIN"""
        return self._data

# ✅ Add new methods without breaking existing
def new_method(self) -> FlextResult[U]:
    """New functionality without breaking changes"""
    pass

# ❌ NEVER remove or change existing methods
# def old_method(self):  # Don't remove or change signature
```

### 2. Type Safety

All APIs must have complete type annotations:

```python
# ✅ Complete type annotations
def process_data(
    input_data: dict[str, Any],
    transformer: Callable[[str], str],
    validator: Optional[Callable[[str], bool]] = None
) -> FlextResult[dict[str, str]]:
    """Process data with complete type information."""
    pass

# ❌ Missing or incomplete types
def process_data(input_data, transformer, validator=None):  # No types
    pass
```

### 3. Error Handling

All operations that can fail must return FlextResult[T]:

```python
# ✅ Explicit error handling
def risky_operation(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult[str].fail("Data required")

    try:
        result = process(data)
        return FlextResult[str].ok(result)
    except Exception as e:
        return FlextResult[str].fail(f"Processing failed: {e}")

# ❌ Exceptions for business logic
def risky_operation(data: str) -> str:
    if not data:
        raise ValueError("Data required")  # Don't use exceptions
    return process(data)
```

---

## Performance Considerations

### Optimization Guidelines

1. **Hot Path Optimization**: Focus on FlextResult operations
   ```python
   # Optimized for common usage patterns
   def map(self, func: Callable[[T], U]) -> "FlextResult[U]":
       # Fast path for success case (most common)
       if self._success:
           return FlextResult._create_success(func(self._data))
       return FlextResult._create_failure(self._error)
   ```

2. **Memory Efficiency**: Minimize object creation
   ```python
   # Reuse common failure cases
   _EMPTY_DATA_ERROR = FlextResult[Any].fail("Empty data")

   def validate_not_empty(data: Any) -> FlextResult[Any]:
       if not data:
           return _EMPTY_DATA_ERROR
       return FlextResult[Any].ok(data)
   ```

3. **Benchmark Critical Paths**
   ```bash
   # Run performance tests
   pytest tests/performance/ -v

   # Benchmark specific operations
   python -m pytest tests/performance/test_result_performance.py::test_map_performance -s
   ```

---

## Security Guidelines

### Dependency Management

```bash
# Security scanning
pip-audit                    # Check for vulnerabilities
bandit -r src/              # Static security analysis

# Dependency updates
poetry update               # Update dependencies
poetry show --outdated     # Check for updates
```

### Code Security

1. **Input Validation**: Always validate inputs
   ```python
   def secure_operation(user_input: str) -> FlextResult[str]:
       # Validate and sanitize
       if not isinstance(user_input, str):
           return FlextResult[str].fail("Invalid input type")

       sanitized = user_input.strip()[:1000]  # Limit size
       return FlextResult[str].ok(sanitized)
   ```

2. **Secret Management**: No secrets in code
   ```python
   # ✅ Use environment variables
   class SecureConfig(FlextConfig):
       api_key: SecretStr          # Masked in logs
       database_url: str

   # ❌ Never hardcode secrets
   API_KEY = "secret-key-here"     # Don't do this
   ```

---

## Documentation Standards

### Code Documentation

```python
def complex_operation(
    data: dict[str, Any],
    options: Optional[dict[str, Any]] = None
) -> FlextResult[ProcessedData]:
    """Process complex data with business logic.

    Args:
        data: Input data dictionary with required fields
        options: Optional processing configuration

    Returns:
        FlextResult containing processed data or error message

    Examples:
        >>> result = complex_operation({"key": "value"})
        >>> if result.is_success:
        ...     processed = result.unwrap()

    Note:
        This operation follows the railway pattern for error handling.
        All business validation errors are returned as FlextResult failures.
    """
    # Implementation
```

### API Documentation

- All public APIs must have complete docstrings
- Include working examples
- Document error conditions
- Specify return types clearly

---

## Contributing Guidelines

### Foundation Library Checklist

- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`make type-check`)
- [ ] Linting passes (`make lint`)
- [ ] Coverage maintained at 84%+
- [ ] API compatibility preserved
- [ ] Documentation updated
- [ ] Examples work with current implementation

### Code Review Standards

1. **Quality Focus**: Functionality, performance, maintainability
2. **Ecosystem Impact**: Consider effects on dependent projects
3. **Pattern Consistency**: Follow established foundation patterns
4. **Security Review**: Check for vulnerabilities

---

**Foundation Development Guide** - Building reliable patterns for the FLEXT ecosystem