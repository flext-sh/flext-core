#!/usr/bin/env python3
"""Complete Pydantic v2 Examples for FLEXT Ecosystem.

This file demonstrates all Pydantic v2 patterns used across the FLEXT ecosystem:
- Model configuration with ConfigDict
- Field validators with @field_validator
- Model validators with @model_validator
- Serialization methods (model_dump, model_dump_json)
- Validation methods (model_validate, model_validate_json)
- Field constraints and domain types
- Performance best practices

These are production-ready patterns used in:
- flext-core, flext-ldap, flext-ldif, flext-cli, algar-oud-mig.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

from flext_core import FlextResult

# ============================================================================
# PATTERN 1: Model Configuration with ConfigDict
# ============================================================================


class UserModel(BaseModel):
    """✅ CORRECT - Use model_config = ConfigDict()."""

    model_config = ConfigDict(
        frozen=False,  # Allow modifications after creation
        validate_assignment=True,  # Validate on attribute assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
        use_attribute_docstrings=True,  # Use docstrings as field descriptions
    )

    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    age: Annotated[int, Field(ge=0, le=150)]


# ============================================================================
# PATTERN 2: Field Validators (@field_validator)
# ============================================================================


class CommandConfig(BaseModel):
    """✅ CORRECT - Use @field_validator for field-level validation."""

    command: str
    timeout_seconds: int

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command format."""
        if not v.strip():
            error_msg = "Command cannot be empty"
            raise ValueError(error_msg)
        return v.lower()

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout range."""
        if not (1 <= v <= 300):
            error_msg = "Timeout must be between 1 and 300 seconds"
            raise ValueError(error_msg)
        return v


# ============================================================================
# PATTERN 3: Model Validators (@model_validator)
# ============================================================================


class PasswordModel(BaseModel):
    """✅ CORRECT - Use @model_validator(mode='after') for cross-field validation."""

    password: str
    password_confirm: str

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets minimum requirements."""
        if len(v) < 8:
            msg = "Password must be at least 8 characters"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "PasswordModel":
        """Validate passwords match."""
        if self.password != self.password_confirm:
            msg = "Passwords do not match"
            raise ValueError(msg)
        return self


# ============================================================================
# PATTERN 4: Serialization Methods (model_dump, model_dump_json)
# ============================================================================


class ServerConfig(BaseModel):
    """Demonstrate serialization patterns."""

    host: str
    port: Annotated[int, Field(ge=1, le=65535)]
    ssl_enabled: bool


def demonstrate_serialization() -> None:
    """Show serialization methods."""
    config = ServerConfig(host="localhost", port=8080, ssl_enabled=True)

    # ✅ FAST - To Python dict
    config_dict = config.model_dump()
    print(f"Python dict: {config_dict}")

    # ✅ FASTEST - To JSON string (Rust-based)
    config_json = config.model_dump_json()
    print(f"JSON string: {config_json}")

    # ✅ CORRECT - JSON-compatible dict with exclude_unset
    config_partial = config.model_dump(exclude_unset=True)
    print(f"Partial dict: {config_partial}")

    # ✅ CORRECT - With custom serializer
    config_serialized = config.model_dump(mode="json")
    print(f"JSON mode dict: {config_serialized}")


# ============================================================================
# PATTERN 5: Validation Methods (model_validate, model_validate_json)
# ============================================================================


def demonstrate_validation() -> None:
    """Show validation methods."""
    # ✅ CORRECT - From dict
    data = {"host": "example.com", "port": 443, "ssl_enabled": True}
    config = ServerConfig.model_validate(data)
    print(f"Validated from dict: {config}")

    # ✅ FASTEST - From JSON string (Rust-based)
    json_data = '{"host": "api.example.com", "port": 8443, "ssl_enabled": true}'
    config_from_json = ServerConfig.model_validate_json(json_data)
    print(f"Validated from JSON: {config_from_json}")


# ============================================================================
# PATTERN 6: Field Constraints and Domain Types
# ============================================================================


class AdvancedFieldConfig(BaseModel):
    """✅ CORRECT - Use Field constraints and Pydantic built-in types."""

    # String constraints
    username: Annotated[str, Field(min_length=1, max_length=50, pattern=r"^\w+$")]

    # Numeric constraints
    port: Annotated[int, Field(ge=1, le=65535)]
    timeout: Annotated[float, Field(gt=0, le=300.0)]
    percentage: Annotated[float, Field(ge=0.0, le=100.0)]

    # Built-in types
    website: HttpUrl
    contact_email: EmailStr

    # Pydantic advanced types
    balance: Decimal = Field(decimal_places=2, max_digits=10)

    # List constraints
    tags: Annotated[list[str], Field(min_items=1, max_items=10)]

    # Dict constraints
    metadata: Annotated[dict[str, str], Field(max_length=5)]


# ============================================================================
# PATTERN 7: Enums and String Choices
# ============================================================================


class EnvironmentEnum(StrEnum):
    """Environment choices."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentConfig(BaseModel):
    """Use enums for constrained choices."""

    environment: EnvironmentEnum
    version: str


# ============================================================================
# PATTERN 8: Nested Models
# ============================================================================


class DatabaseConnection(BaseModel):
    """Nested model for database configuration."""

    host: str
    port: Annotated[int, Field(ge=1, le=65535)]
    database: str
    username: str


class ApplicationConfig(BaseModel):
    """Parent model with nested models."""

    name: str
    database: DatabaseConnection
    environment: EnvironmentEnum
    debug: bool = False


def demonstrate_nested_models() -> None:
    """Show nested model validation."""
    config_data = {
        "name": "MyApp",
        "database": {
            "host": "db.example.com",
            "port": 5432,
            "database": "myapp_db",
            "username": "app_user",
        },
        "environment": "production",
        "debug": False,
    }

    config = ApplicationConfig.model_validate(config_data)
    print(f"Nested config: {config}")
    print(f"Database host: {config.database.host}")


# ============================================================================
# PATTERN 9: Performance - TypeAdapter for Batch Operations
# ============================================================================

# ✅ FAST - Module-level adapter (created once)
_USER_ADAPTER: Final = TypeAdapter(list[UserModel])
_CONFIG_ADAPTER: Final = TypeAdapter(list[ServerConfig])


def validate_batch_users(data_list: list[dict]) -> list[UserModel]:
    """✅ FAST - Use TypeAdapter at module level for batch validation."""
    return _USER_ADAPTER.validate_python(data_list)


def validate_batch_configs(data_list: list[dict]) -> list[ServerConfig]:
    """✅ FAST - Reuse module-level adapter."""
    return _CONFIG_ADAPTER.validate_python(data_list)


# ============================================================================
# PATTERN 10: LDAP-Specific Patterns (from flext-ldap)
# ============================================================================


class LDAPConnectionConfig(BaseModel):
    """LDAP-specific configuration pattern."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    # Basic connection
    server: str = Field(min_length=1, description="LDAP server hostname or IP")
    port: Annotated[int, Field(ge=1, le=65535, description="LDAP server port")]
    use_ssl: bool = Field(default=False, description="Use SSL/TLS")

    # Authentication
    bind_dn: str | None = Field(default=None, description="Bind DN for authentication")
    bind_password: str | None = Field(default=None, description="Bind password")

    # Business logic validators
    @field_validator("bind_dn")
    @classmethod
    def validate_bind_dn(cls, v: str | None) -> str | None:
        """Business logic: DN must contain attribute=value pairs."""
        if v is None:
            return v
        if "=" not in v:
            msg = "DN must contain attribute=value pairs (e.g., cn=admin,dc=example,dc=com)"
            raise ValueError(msg)
        return v

    @field_validator("server")
    @classmethod
    def validate_server_format(cls, v: str) -> str:
        """Business logic: Server must be valid hostname or IP."""
        v = v.strip()
        if not v:
            msg = "Server cannot be empty"
            raise ValueError(msg)
        return v


# ============================================================================
# PATTERN 11: Railway Pattern with Validation
# ============================================================================


def process_with_validation(data: dict) -> FlextResult[UserModel]:
    """✅ CORRECT - Combine Pydantic validation with FlextResult."""
    try:
        user = UserModel.model_validate(data)
        return FlextResult[UserModel].ok(user)
    except Exception as e:
        return FlextResult[UserModel].fail(f"Validation failed: {e!s}")


# ============================================================================
# PATTERN 12: Complete Application Example
# ============================================================================


class ServiceRequest(BaseModel):
    """Complete service request model with all patterns."""

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # Request metadata
    request_id: str = Field(
        default_factory=lambda: f"req_{datetime.now(UTC).isoformat()}"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Service configuration
    service_name: str = Field(min_length=1, max_length=100)
    operation: str = Field(min_length=1, max_length=50)

    # Parameters
    parameters: Annotated[dict[str, object], Field(max_length=20)] = {}

    # Status and results
    status: Annotated[str, Field(pattern="^(pending|running|completed|failed)$")] = (
        "pending"
    )
    result: str | None = None

    # Validators
    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, v: str) -> str:
        """Validate service name format."""
        return v.lower().replace(" ", "_")

    @model_validator(mode="after")
    def validate_completion_status(self) -> "ServiceRequest":
        """If completed, result must be set."""
        if self.status == "completed" and not self.result:
            msg = "Result required when status is completed"
            raise ValueError(msg)
        return self


def demonstrate_complete_example() -> None:
    """Show complete workflow."""
    # ✅ Create request
    request_data = {
        "service_name": "User Service",
        "operation": "create_user",
        "parameters": {
            "username": "john_doe",
            "email": "john@example.com",
        },
        "status": "pending",
    }

    # ✅ Validate
    request = ServiceRequest.model_validate(request_data)
    print(f"✅ Request created: {request.request_id}")

    # ✅ Serialize for transport
    json_data = request.model_dump_json()
    print(f"✅ Serialized: {json_data[:100]}...")

    # ✅ Deserialize from transport
    received = ServiceRequest.model_validate_json(json_data)
    print(f"✅ Deserialized request: {received.request_id}")


# ============================================================================
# MAIN - Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PYDANTIC V2 COMPLETE EXAMPLES FOR FLEXT ECOSYSTEM")
    print("=" * 80)

    print("\n1️⃣  SERIALIZATION EXAMPLES")
    print("-" * 80)
    demonstrate_serialization()

    print("\n2️⃣  VALIDATION EXAMPLES")
    print("-" * 80)
    demonstrate_validation()

    print("\n3️⃣  NESTED MODELS EXAMPLES")
    print("-" * 80)
    demonstrate_nested_models()

    print("\n4️⃣  BATCH OPERATIONS WITH TYPEADAPTER")
    print("-" * 80)
    users_data = [
        {"name": "Alice", "email": "alice@example.com", "age": 30},
        {"name": "Bob", "email": "bob@example.com", "age": 25},
    ]
    users = validate_batch_users(users_data)
    print(f"✅ Validated {len(users)} users")

    print("\n5️⃣  LDAP PATTERN EXAMPLE")
    print("-" * 80)
    ldap_config = LDAPConnectionConfig(
        server="ldap.example.com",
        port=389,
        use_ssl=False,
        bind_dn="cn=admin,dc=example,dc=com",
        bind_password="secret",
    )
    print(f"✅ LDAP config: {ldap_config.server}:{ldap_config.port}")

    print("\n6️⃣  RAILWAY PATTERN EXAMPLE")
    print("-" * 80)
    result = process_with_validation({
        "name": "Test",
        "email": "test@example.com",
        "age": 25,
    })
    if result.is_success:
        user = result.unwrap()
        print(f"✅ User validated: {user.name}")
    else:
        print(f"❌ Validation failed: {result.error}")

    print("\n7️⃣  COMPLETE APPLICATION EXAMPLE")
    print("-" * 80)
    demonstrate_complete_example()

    print("\n" + "=" * 80)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
