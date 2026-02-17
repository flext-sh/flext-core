# Pydantic v2 Patterns for FLEXT Ecosystem

**Status**: Production Ready | **Version**: 0.10.0 | **Target**: flext-core and dependent projects

This guide documents essential Pydantic v2 patterns used throughout the FLEXT ecosystem, with examples from the codebase.

## Canonical Rules

- Follow root governance in `CLAUDE.md`.
- Keep all examples pure Pydantic v2 (`model_dump`, `model_validate`, `ConfigDict`).
- Keep guidance consistent with `lib-pydantic-v2` and `lib-pydantic-settings` rules.

## Core Principles

FLEXT projects use **pure Pydantic v2 patterns** (no v1 compatibility layer):

1. Use `BaseModel` and `BaseSettings` from pydantic
2. Use `ConfigDict` for model configuration
3. Use `Field()` with constraints and descriptions
4. Use `@field_validator` for field-level validation
5. Use `@model_validator` for cross-field validation
6. Use `Annotated` types for semantic meaning
7. Use `computed_field` for derived properties

**Important**: Do NOT use old Pydantic v1 patterns:

- ❌ `.dict()` → Use `.model_dump()`
- ❌ `.json()` → Use `.model_dump_json()`
- ❌ `.parse_obj()` → Use `.model_validate()`
- ❌ `class Config:` → Use `model_config = ConfigDict(...)`
- ❌ `@validator` → Use `@field_validator`

## Pattern 1: Basic Model with Constraints

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    """User model with field constraints."""
    user_id: str = Field(
        description="Unique user identifier",
        example="user_001",
    )
    username: str = Field(
        min_length=3,
        max_length=20,
        description="Username (3-20 characters)",
        example="alice",
    )
    email: str = Field(
        pattern=r'^[^@]+@[^@]+\.[^@]+$',
        description="User email address",
        example="alice@example.com",
    )
    age: int = Field(
        ge=0,
        le=150,
        description="User age (0-150)",
        example=30,
    )
    is_active: bool = Field(
        default=True,
        description="Whether user is active",
    )

# Usage
user = User(
    user_id="user_001",
    username="alice",
    email="alice@example.com",
    age=30,
)

# Serialize
data = user.model_dump()
json_str = user.model_dump_json()

# Deserialize
user2 = User.model_validate({"user_id": "...", ...})
user3 = User.model_validate_json('{"user_id": "..."}')
```

**From FLEXT**: `src/flext_core/config.py` - FlextSettings uses this pattern extensively.

## Pattern 2: ConfigDict for Model Settings

```python
from pydantic import BaseModel, ConfigDict, Field

class ApiResponse(BaseModel):
    """API response with strict configuration."""
    model_config = ConfigDict(
        # Validation
        validate_assignment=True,      # Validate on attribute assignment
        validate_default=True,         # Validate default values
        use_enum_values=True,          # Use enum values in serialization

        # Serialization
        ser_json_timedelta="float",    # Serialize timedelta as float
        ser_json_bytes="utf8",         # Serialize bytes as UTF-8 string

        # Extra fields
        extra="forbid",                # Reject extra fields

        # Aliases
        populate_by_name=True,         # Allow both field name and alias

        # Freezing
        frozen=False,                  # Allow mutation

        # JSON schema
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "data": {"id": 123},
                }
            ]
        },
    )

    status: str = Field(
        pattern="^(success|error)$",
        description="Response status",
    )
    data: dict | None = Field(
        default=None,
        description="Response data",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status is error",
    )
```

**From FLEXT**: `src/flext_core/config.py` uses ConfigDict for strict validation and serialization control.

## Pattern 3: Field Validators

### Single Field Validator

```python
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    """Product with field validation."""
    name: str = Field(min_length=1)
    price: float = Field(gt=0)  # Positive number
    sku: str = Field(min_length=5, max_length=20)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize product name - remove extra whitespace."""
        return v.strip().title()

    @field_validator("price")
    @classmethod
    def round_price(cls, v: float) -> float:
        """Round price to 2 decimals."""
        return round(v, 2)

    @field_validator("sku")
    @classmethod
    def validate_sku_format(cls, v: str) -> str:
        """SKU must be alphanumeric."""
        if not v.replace("-", "").isalnum():
            raise ValueError("SKU must be alphanumeric")
        return v.upper()

# Usage
product = Product(
    name="  laptop computer  ",  # Normalized to "Laptop Computer"
    price=999.995,  # Rounded to 999.99
    sku="abc-123",  # Uppercased to "ABC-123"
)
```

### Validator Modes: before, after, wrap

```python
from pydantic import field_validator, mode

class Config(BaseModel):
    timeout: int = Field(gt=0, le=3600)
    retries: int = Field(ge=0, le=10)

    @field_validator("timeout", mode="before")
    @classmethod
    def parse_timeout_before(cls, v):
        """Runs BEFORE Pydantic's type coercion."""
        if isinstance(v, str):
            return int(v)  # Parse string to int
        return v

    @field_validator("timeout", mode="after")
    @classmethod
    def validate_timeout_after(cls, v):
        """Runs AFTER Pydantic's type coercion."""
        if v < 10:
            raise ValueError("Timeout must be >= 10")
        return v

    @field_validator("retries", mode="wrap")
    @classmethod
    def wrap_retries(cls, v, handler, info):
        """Wraps the entire validation pipeline."""
        result = handler(v)  # Call original validator
        if info.context and info.context.get("strict"):
            if result == 0:
                raise ValueError("Retries must be > 0 in strict mode")
        return result

# Usage
config = Config(
    timeout="300",  # Parsed from string
    retries=3,
    context={"strict": True},
)
```

### Validate Multiple Fields

```python
from pydantic import field_validator

class DateRange(BaseModel):
    start_date: date
    end_date: date

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates_not_future(cls, v):
        """Validate both fields the same way."""
        if v > date.today():
            raise ValueError("Dates cannot be in the future")
        return v
```

## Pattern 4: Model Validators (Cross-Field)

```python
from pydantic import BaseModel, model_validator

class PasswordChange(BaseModel):
    """Password change with cross-field validation."""
    current_password: str
    new_password: str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        """Validate that new and confirm passwords match."""
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

    @model_validator(mode="after")
    def password_different(self):
        """Validate that new password is different from current."""
        if self.current_password == self.new_password:
            raise ValueError("New password must be different from current")
        return self
```

**From FLEXT**: `src/flext_core/config.py` uses model validators for complex configuration rules.

## Pattern 5: Computed Fields

```python
from pydantic import BaseModel, computed_field

class Person(BaseModel):
    """Person with computed derived properties."""
    first_name: str
    last_name: str
    birth_year: int

    @computed_field
    @property
    def full_name(self) -> str:
        """Computed: full name."""
        return f"{self.first_name} {self.last_name}"

    @computed_field
    @property
    def age(self) -> int:
        """Computed: age based on birth year."""
        from datetime import date
        return date.today().year - self.birth_year

# Usage
person = Person(first_name="Alice", last_name="Smith", birth_year=1990)

# Serialization includes computed fields
data = person.model_dump()
# {
#     "first_name": "Alice",
#     "last_name": "Smith",
#     "birth_year": 1990,
#     "full_name": "Alice Smith",      # Computed
#     "age": 34,                        # Computed
# }
```

**From FLEXT**: `src/flext_core/models.py` uses computed_field for semantic properties.

## Pattern 6: Annotated Types for Semantic Meaning

```python
from typing import Annotated
from pydantic import BaseModel, Field

# Define semantic types
UserId = Annotated[str, Field(description="Unique user identifier", example="user_001")]
Email = Annotated[str, Field(pattern=r'^[^@]+@[^@]+\.[^@]+$', description="Email address")]
PositiveInt = Annotated[int, Field(gt=0, description="Positive integer")]
PortNumber = Annotated[int, Field(ge=1, le=65535, description="Network port (1-65535)")]

class ServiceConfig(BaseModel):
    """Service configuration using semantic types."""
    service_id: UserId
    REDACTED_LDAP_BIND_PASSWORD_email: Email
    worker_count: PositiveInt
    port: PortNumber

# Usage - type checker knows constraints
config = ServiceConfig(
    service_id="svc_001",
    REDACTED_LDAP_BIND_PASSWORD_email="REDACTED_LDAP_BIND_PASSWORD@example.com",
    worker_count=4,
    port=8080,
)
```

**From FLEXT**: `src/flext_core/typings.py` defines 30+ semantic Annotated types used across projects.

## Pattern 7: Settings with Environment Variables

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    """Application settings with environment variables."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",    # Use __ for nested fields
        case_sensitive=False,          # Case-insensitive env vars
        extra="forbid",                # Reject unknown fields
    )

    # Simple fields
    app_name: str = Field(default="MyApp", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Sensitive fields
    database_url: SecretStr = Field(env="DATABASE_URL")
    api_key: SecretStr = Field(env="API_KEY")

    # Nested settings
    class Database(BaseSettings):
        host: str = "localhost"
        port: int = 5432
        name: str

    database: Database = Field(default_factory=Database)

# Usage - automatically loads from environment
settings = Settings()
# Settings(
#     app_name="MyApp",
#     debug=False,
#     log_level="INFO",
#     database_url=SecretStr("***"),
#     api_key=SecretStr("***"),
#     database=Database(host="localhost", port=5432, name="mydb")
# )

# Access values
print(f"Database: {settings.database.host}:{settings.database.port}")

# Access secret values securely
print(f"API Key: {settings.api_key.get_secret_value()}")  # Only when needed
```

**From FLEXT**: `src/flext_core/config.py` uses BaseSettings for environment configuration.

## Pattern 8: Custom Types

```python
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from typing import Annotated

# Define custom validator type
def validate_country_code(code: str) -> str:
    """Validate ISO 3166-1 alpha-2 country code."""
    valid_codes = {"US", "CA", "GB", "DE", "FR", "JP", "AU"}
    if code.upper() not in valid_codes:
        raise ValueError(f"Invalid country code: {code}")
    return code.upper()

CountryCode = Annotated[
    str,
    Field(description="ISO 3166-1 alpha-2 country code"),
]

class Address(BaseModel):
    street: str
    city: str
    postal_code: str
    country: CountryCode

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        return validate_country_code(v)

# Usage
address = Address(
    street="123 Main St",
    city="New York",
    postal_code="10001",
    country="us",  # Normalized to "US"
)
```

## Pattern 9: Discriminated Unions for Polymorphism

```python
from typing import Annotated, Literal, Union
from pydantic import Discriminator, Field, BaseModel

class Tap(BaseModel):
    """Source tap configuration."""
    type: Literal["tap"] = "tap"
    name: str
    connector: str

class Target(BaseModel):
    """Destination target configuration."""
    type: Literal["target"] = "target"
    name: str
    connector: str

class Dbt(BaseModel):
    """Data transformation configuration."""
    type: Literal["dbt"] = "dbt"
    name: str
    project_path: str

# Discriminated union - type-safe polymorphism
ComponentConfig = Annotated[
    Union[Tap, Target, Dbt],
    Discriminator("type"),
]

class Pipeline(BaseModel):
    """Pipeline with multiple component types."""
    components: list[ComponentConfig]

# Usage - type-safe
pipeline_data = {
    "components": [
        {"type": "tap", "name": "source", "connector": "postgres"},
        {"type": "target", "name": "sink", "connector": "snowflake"},
        {"type": "dbt", "name": "transformations", "project_path": "./dbt"},
    ]
}

pipeline = Pipeline.model_validate(pipeline_data)

# Type checker knows exact types
for component in pipeline.components:
    if isinstance(component, Tap):
        print(f"Tap: {component.name} -> {component.connector}")
    elif isinstance(component, Target):
        print(f"Target: {component.name} -> {component.connector}")
    elif isinstance(component, Dbt):
        print(f"DBT: {component.name} @ {component.project_path}")
```

## Pattern 10: JSON Schema Generation

```python
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

class User(BaseModel):
    name: str
    email: str
    age: int

class Product(BaseModel):
    name: str
    price: float

# Generate JSON schemas for multiple models
schemas = models_json_schema(
    [
        (User, "validation"),
        (Product, "validation"),
    ],
    by_alias=True,
    ref_template="#/definitions/{model}",
)

import json
print(json.dumps(schemas, indent=2))

# Output includes:
# {
#   "$defs": {
#     "User": {
#       "properties": {
#         "name": {"type": "string"},
#         "email": {"type": "string", "format": "email"},
#         "age": {"type": "integer", "minimum": 0}
#       },
#       "required": ["name", "email", "age"],
#       "type": "object"
#     },
#     ...
#   }
# }
```

## Integration with FlextResult

Always wrap Pydantic validation in FlextResult:

```python
from flext_core import FlextResult
from pydantic import BaseModel, ValidationError

class UserModel(BaseModel):
    email: str
    password: str

def validate_user(data: dict) -> FlextResult[UserModel]:
    """Validate user data with FlextResult."""
    try:
        user = UserModel(**data)
        return FlextResult[UserModel].ok(user)
    except ValidationError as e:
        return FlextResult[UserModel].fail(
            f"User validation failed: {e}",
            error_code="USER_VALIDATION_ERROR",
            error_data={"errors": e.errors()},
        )

# Usage
result = validate_user({"email": "user@example.com", "password": "secret123"})
if result.is_success:
    user = result.value
else:
    print(f"Validation failed: {result.error}")
```

## Best Practices

1. **Use `Field()` for documentation**

   ```python
   # Good - includes description and example
   name: str = Field(description="User name", example="Alice")

   # Avoid - no documentation
   name: str
   ```

2. **Validate in `@field_validator`**

   ```python
   # Good - validation is explicit
   @field_validator("age")
   @classmethod
   def validate_age(cls, v):
       if v < 0:
           raise ValueError("Age must be positive")
       return v

   # Avoid - validation hidden in constraints
   age: int = Field(ge=0)  # Only for simple cases
   ```

3. **Use ConfigDict for strict validation**

   ```python
   # Good - strict configuration
   model_config = ConfigDict(validate_assignment=True, extra="forbid")

   # Avoid - permissive configuration
   class Config:
       validate_assignment = False
       extra = "allow"
   ```

4. **Use computed_field for derived properties**

   ```python
   # Good - automatically included in serialization
   @computed_field
   @property
   def full_name(self) -> str:
       return f"{self.first_name} {self.last_name}"

   # Avoid - manual properties
   @property
   def full_name(self) -> str:  # Not in model_dump()
       return f"{self.first_name} {self.last_name}"
   ```

## Checklists

### Model Configuration

- ✅ Use `ConfigDict` instead of `class Config`
- ✅ Set `extra="forbid"` to reject unknown fields
- ✅ Set `validate_assignment=True` for runtime validation
- ✅ Add `json_schema_extra` with examples
- ❌ Don't use old Pydantic v1 Config class

### Field Validation

- ✅ Use `@field_validator` for field-level validation
- ✅ Use `@model_validator` for cross-field validation
- ✅ Include `description` and `example` in Field()
- ✅ Wrap Pydantic errors in FlextResult
- ❌ Don't use exceptions for validation failures

### Serialization

- ✅ Use `model_dump()` instead of `.dict()`
- ✅ Use `model_dump_json()` instead of `.json()`
- ✅ Use `model_validate()` instead of `.parse_obj()`
- ✅ Use `model_validate_json()` instead of `.parse_raw()`
- ❌ Don't use old Pydantic v1 serialization methods

## See Also

- [Railway-Oriented Programming](./railway-oriented-programming.md)
- [Anti-Patterns and Best Practices](./anti-patterns-best-practices.md)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/2.4/)
- **FLEXT CLAUDE.md**: Development standards and patterns

---

**Example from FLEXT**: See `src/flext_core/config.py` (423 lines) for comprehensive Pydantic v2 usage patterns in production code.

**Updated**: 2025-12-07 | **Version**: 0.10.0 | **Pydantic**: v2.12.3+
