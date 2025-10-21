# Appendix D: Glossary of Terms

**Status**: TERMINOLOGY REFERENCE
**Purpose**: Define all terms used in this plan
**Usage**: Quick lookup for unfamiliar terms

---

## A

**Annotated**
Python typing construct combining a type with metadata. In Pydantic v2, used to attach Field constraints and validators to types for reusability.
```python
PortNumber = Annotated[int, Field(ge=1, le=65535)]  # MIN_PORT=1
```

**AfterValidator**  
Pydantic v2 validator that runs AFTER type coercion. Used in Annotated types.
```python
def validate_positive(v: int) -> int:
    if v <= 0: raise ValueError
    return v
PositiveInt = Annotated[int, AfterValidator(validate_positive)]
```

**Aggregate Root**  
Domain-Driven Design pattern. Entity that acts as consistency boundary for a cluster of objects. In FLEXT: `FlextModels.AggregateRoot`.

---

## B

**BaseModel**  
Pydantic's base class for data models with automatic validation and serialization.
```python
class User(BaseModel):
    name: str
```

**BeforeValidator**  
Pydantic v2 validator that runs BEFORE type coercion. Used for parsing/normalization.
```python
def parse_int(v: str | int) -> int:
    return int(v)
ParsedInt = Annotated[int, BeforeValidator(parse_int)]
```

**Breaking Change**  
API modification that requires dependent code to be updated. In FLEXT ecosystem (32+ projects), requires careful deprecation cycle.

---

## C

**ConfigDict**  
Pydantic v2 configuration object replacing `class Config`. Passed to `model_config`.
```python
model_config = ConfigDict(frozen=True, extra="forbid")
```

**Computed Field**  
Property included in model serialization using `@computed_field` decorator.
```python
@computed_field
@property
def full_name(self) -> str:
    return f"{self.first_name} {self.last_name}"
```

**Coverage**  
Test coverage percentage. FLEXT targets: 79%+ (flext-core), 75%+ (others).

---

## D

**Deprecation Cycle**  
Period where old API remains functional with warnings. FLEXT requires minimum 2 versions (6+ months).

**Discriminator**  
Field used to identify union member in tagged unions for O(1) validation.
```python
Animal = Annotated[Cat | Dog, Discriminator("type")]
```

**Domain-Driven Design (DDD)**  
Software design approach organizing code around business domain. FLEXT uses Entity/Value Object/Aggregate Root patterns.

---

## E

**Entity**  
DDD pattern. Object with identity that persists over time. In FLEXT: `FlextModels.Entity`.

**Extra Fields**  
Fields in input not defined in model. Pydantic v2 behavior controlled by `extra` in ConfigDict ("allow", "forbid", "ignore").

---

## F

**Field**  
Pydantic function for adding validation constraints and metadata to model fields.
```python
age: int = Field(ge=0, le=150, description="User age")
```

**field_validator**  
Pydantic v2 decorator for field-level validation, replacing `@validator`.
```python
@field_validator("age")
@classmethod
def validate_age(cls, value: int) -> int:
    return value
```

**field_serializer**  
Pydantic v2 decorator for custom field serialization.
```python
@field_serializer("price")
def serialize_price(self, value: Decimal) -> str:
    return f"${value:.2f}"
```

**FlextResult[T]**  
FLEXT railway pattern for error handling. Replaces exception-based error handling.
```python
def validate_user(data: dict) -> FlextResult[User]:
    if not data: return FlextResult[User].fail("Data required")
    return FlextResult[User].ok(User(**data))
```

**Frozen**  
Model configuration making instances immutable. Used for Value Objects.
```python
model_config = ConfigDict(frozen=True)
```

---

## L

**Layer Hierarchy**  
FLEXT architectural principle. Code organized in layers (0: Constants, 1: Foundation, 2: Domain, 3: Application, 4: Infrastructure). Higher layers import from lower only.

**Legacy Pattern**  
Pydantic v1 API that should be replaced (`.dict()`, `parse_obj()`, `@validator`, `class Config`).

---

## M

**model_dump()**  
Pydantic v2 method for exporting model to dict, replacing `.dict()`.
```python
data = user.model_dump()
data = user.model_dump(mode="json")  # JSON-serializable types
```

**model_dump_json()**  
Pydantic v2 method for exporting model to JSON string, replacing `.json()`.
```python
json_str = user.model_dump_json()
```

**model_validate()**  
Pydantic v2 method for validating dict/object into model, replacing `parse_obj()`.
```python
user = User.model_validate({"name": "Alice"})
```

**model_validate_json()**  
Pydantic v2 method for parsing JSON directly into model (Rust-optimized), replacing `parse_raw()`.
```python
user = User.model_validate_json('{"name": "Alice"}')
```

**model_validator**  
Pydantic v2 decorator for model-level validation, replacing `@root_validator`.
```python
@model_validator(mode="after")
def validate_model(self) -> "MyModel":
    return self
```

**Monadic Bind**  
Functional programming pattern. In FlextResult, implemented as `.flat_map()` for chaining operations that return FlextResult.

---

## P

**Pydantic v1**  
Previous major version (1.x). Uses `.dict()`, `parse_obj()`, `@validator`, `class Config`.

**Pydantic v2**  
Current major version (2.x). Uses `model_dump()`, `model_validate()`, `@field_validator`, `ConfigDict`. Written in Rust for performance.

**Pyrefly**  
Type checker successor to MyPy used in FLEXT ecosystem. Supports strict mode.

---

## R

**Railway-Oriented Programming**  
Functional error handling pattern. Operations return Result type (success/failure) instead of throwing exceptions. FLEXT uses FlextResult[T].

**Root Import**  
Import from package root. FLEXT ecosystem standard: `from flext_core import X` (not `from flext_core.module import X`).

**Ruff**  
Fast Python linter/formatter used in FLEXT. Replaces Black, isort, flake8.

---

## S

**SOLID**  
Five object-oriented design principles. FLEXT enforces: Single class per module, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.

**Strict Mode**  
Pydantic configuration disabling automatic type coercion.
```python
model_config = ConfigDict(strict=True)
```

---

## T

**Tagged Union**  
Discriminated union using a field to identify type. Enables O(1) validation vs O(N) for plain unions.
```python
Animal = Annotated[Cat | Dog, Discriminator("type")]
```

**TypeAdapter**  
Pydantic utility for validating non-BaseModel types (lists, dicts, primitives).
```python
_ADAPTER = TypeAdapter(list[User])
users = _ADAPTER.validate_python(data)
```

**Type Safety**  
Compile-time verification that types are used correctly. FLEXT requires complete type annotations with zero `Any` types.

---

## V

**ValidationError**  
Pydantic exception raised when validation fails. Contains detailed error information.
```python
try:
    user = User(**data)
except ValidationError as e:
    print(e.errors())
```

**Value Object**  
DDD pattern. Immutable object compared by value, not identity. In FLEXT: `FlextModels.Value`.

---

## Z

**Zero Tolerance**  
FLEXT policy: no compromises on quality. All errors must be fixed, all duplication removed, all deprecated patterns replaced.

---

## Acronyms

**API**: Application Programming Interface  
**CI/CD**: Continuous Integration/Continuous Deployment  
**DDD**: Domain-Driven Design  
**DI**: Dependency Injection  
**DRY**: Don't Repeat Yourself  
**JSON**: JavaScript Object Notation  
**OOP**: Object-Oriented Programming  
**RC**: Release Candidate  
**SOLID**: Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion  
**TDD**: Test-Driven Development  

---

## FLEXT-Specific Terms

**flext-core**  
Foundation library for FLEXT ecosystem. Provides FlextResult, FlextContainer, FlextModels, etc. Used by 32+ dependent projects.

**FlextContainer**  
Dependency injection container. Global singleton pattern.
```python
container = FlextContainer.get_global()
container.register("db", DatabaseService())
```

**FlextModels**  
Base classes for DDD patterns: Entity, Value, AggregateRoot.

**FlextResult[T]**  
Railway pattern for error handling. Core FLEXT pattern replacing exceptions.

**Quality Gates**  
Automated checks before commits: `make validate` = lint + type-check + security + test.

---

## Migration-Specific Terms

**Backward Compatibility**  
Maintaining old API while introducing new API. FLEXT requires both `.data` and `.value` on FlextResult during transition.

**Deprecation Warning**  
Python warning indicating function/API will be removed in future version.
```python
warnings.warn("Use model_dump() instead", DeprecationWarning)
```

**Ecosystem Impact**  
Effect of changes on dependent projects. flext-core changes affect 32+ projects.

**Migration Path**  
Step-by-step process for updating code from old to new patterns.

---

**Usage**: Search this glossary when encountering unfamiliar terms in the plan.

**Next**: [Appendix E: Code Examples](./APPENDIX_E_CODE_EXAMPLES.md)
