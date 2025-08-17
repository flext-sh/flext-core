"""Modern Pydantic TypeAdapter patterns for FLEXT Core."""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import TypeVar

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from flext_core.result import FlextResult

# =============================================================================
# TYPE VARIABLES FOR GENERIC CONSTRAINTS
# =============================================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# =============================================================================
# TYPE ALIASES FOR SIMPLIFIED VALIDATION
# =============================================================================

# Simple scalar types
EntityId = str
Version = int
Timestamp = datetime
Host = str
Port = int

# Business domain types
EmailAddress = str
ServiceName = str
ErrorCode = str
ErrorMessage = str

# Collection types - use constrained generics instead of Any
Metadata = dict[str, object]
EventList = list[dict[str, object]]
ConfigDict_Type = dict[str, object]

# Complex types
ConnectionString = str
Percentage = float

# =============================================================================
# TYPEADAPTER FACTORY FOR CONSISTENT CREATION
# =============================================================================


class TypeAdapterFactory:
    """Factory for creating TypeAdapter instances with consistent configuration.

    This centralizes TypeAdapter creation with standard configuration
    across the FLEXT ecosystem, ensuring consistent validation behavior.
    """

    @staticmethod
    def create_adapter[T](type_: type[T]) -> TypeAdapter[T]:
      """Create TypeAdapter with standard FLEXT configuration.

      Args:
          type_: The type to create an adapter for

      Returns:
          Configured TypeAdapter instance

      """
      return TypeAdapter(type_)

    @staticmethod
    def create_list_adapter[T](item_type: type[T]) -> TypeAdapter[list[T]]:
      """Create TypeAdapter for lists.

      Args:
          item_type: The type of items in the list

      Returns:
          TypeAdapter for List[T]

      """
      return TypeAdapter(list[item_type])  # type: ignore[valid-type]

    @staticmethod
    def create_dict_adapter[T](value_type: type[T]) -> TypeAdapter[dict[str, T]]:
      """Create TypeAdapter for dictionaries.

      Args:
          value_type: The type of values in the dict

      Returns:
          TypeAdapter for Dict[str, T]

      """
      return TypeAdapter(dict[str, value_type])  # type: ignore[valid-type]


# =============================================================================
# VALIDATION ADAPTERS WITH BUSINESS RULES
# =============================================================================


class ValidationAdapters:
    """Centralized validation adapters for common FLEXT types.

    Each adapter includes appropriate validation rules and provides
    both Python and JSON validation capabilities.
    """

    # Core entity types
    entity_id_adapter = TypeAdapter(EntityId)
    version_adapter = TypeAdapter(Version)
    timestamp_adapter = TypeAdapter(Timestamp)

    # Network types
    host_adapter = TypeAdapter(Host)
    port_adapter = TypeAdapter(Port)

    # Business types
    email_adapter = TypeAdapter(EmailAddress)
    service_name_adapter = TypeAdapter(ServiceName)
    error_code_adapter = TypeAdapter(ErrorCode)
    error_message_adapter = TypeAdapter(ErrorMessage)

    # Collection types
    metadata_adapter = TypeAdapter(Metadata)
    event_list_adapter = TypeAdapter(EventList)
    config_dict_adapter = TypeAdapter(ConfigDict_Type)

    # Complex types
    connection_string_adapter = TypeAdapter(ConnectionString)
    percentage_adapter = TypeAdapter(Percentage)

    @classmethod
    def validate_entity_id(cls, value: object) -> FlextResult[str]:
      """Validate entity ID with business rules.

      Args:
          value: Value to validate

      Returns:
          FlextResult containing validated entity ID or error

      """
      try:
          if not value or (isinstance(value, str) and len(value.strip()) == 0):
              return FlextResult.fail("Entity ID cannot be empty")

          validated = cls.entity_id_adapter.validate_python(value)
          return FlextResult.ok(validated)
      except Exception as e:
          return FlextResult.fail(f"Invalid entity ID: {e}")

    @classmethod
    def validate_version(cls, value: object) -> FlextResult[int]:
      """Validate version with business rules.

      Args:
          value: Value to validate

      Returns:
          FlextResult containing validated version or error

      """
      try:
          validated = cls.version_adapter.validate_python(value)
          if validated < 1:
              return FlextResult.fail("Version must be >= 1")
          return FlextResult.ok(validated)
      except Exception as e:
          return FlextResult.fail(f"Invalid version: {e}")

    @classmethod
    def validate_email(cls, value: object) -> FlextResult[str]:
      """Validate email address with business rules.

      Args:
          value: Value to validate

      Returns:
          FlextResult containing validated email or error

      """
      try:
          validated = cls.email_adapter.validate_python(value)
          # Basic email validation
          if "@" not in validated or "." not in validated.split("@")[1]:
              return FlextResult.fail("Invalid email format")
          return FlextResult.ok(validated)
      except Exception as e:
          return FlextResult.fail(f"Invalid email: {e}")

    @classmethod
    def validate_service_name(cls, value: object) -> FlextResult[str]:
      """Validate service name with business rules.

      Args:
          value: Value to validate

      Returns:
          FlextResult containing validated service name or error

      """
      try:
          validated = cls.service_name_adapter.validate_python(value)
          # Service name validation
          max_service_name_length = 64
          if len(validated) > max_service_name_length:
              return FlextResult.fail(
                  f"Service name too long (max {max_service_name_length} chars)",
              )
          if not validated.replace("-", "").replace("_", "").isalnum():
              return FlextResult.fail("Service name must be alphanumeric with - or _")
          return FlextResult.ok(validated)
      except Exception as e:
          return FlextResult.fail(f"Invalid service name: {e}")

    @classmethod
    def validate_host_port(
      cls,
      host: object,
      port: object,
    ) -> FlextResult[tuple[str, int]]:
      """Validate host and port combination.

      Args:
          host: Host value to validate
          port: Port value to validate

      Returns:
          FlextResult containing validated (host, port) tuple or error

      """
      try:
          validated_host = cls.host_adapter.validate_python(host)
          validated_port = cls.port_adapter.validate_python(port)

          # Port range validation
          min_port = 1
          max_port = 65535
          if not (min_port <= validated_port <= max_port):
              return FlextResult.fail(
                  f"Port must be between {min_port} and {max_port}",
              )

          return FlextResult.ok((validated_host, validated_port))
      except Exception as e:
          return FlextResult.fail(f"Invalid host/port: {e}")

    @classmethod
    def validate_percentage(cls, value: object) -> FlextResult[float]:
      """Validate percentage with business rules.

      Args:
          value: Value to validate

      Returns:
          FlextResult containing validated percentage or error

      """
      try:
          validated = cls.percentage_adapter.validate_python(value)
          min_percentage = 0.0
          max_percentage = 100.0
          if not (min_percentage <= validated <= max_percentage):
              return FlextResult.fail(
                  f"Percentage must be between {min_percentage} and {max_percentage}",
              )
          return FlextResult.ok(validated)
      except Exception as e:
          return FlextResult.fail(f"Invalid percentage: {e}")


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


class SerializationHelpers:
    """Helper functions for common serialization tasks using TypeAdapter."""

    @staticmethod
    def to_json[T](adapter: TypeAdapter[T], value: T) -> FlextResult[str]:
      """Serialize value to JSON string.

      Args:
          adapter: TypeAdapter instance
          value: Value to serialize

      Returns:
          FlextResult containing JSON string or error

      """
      try:
          json_bytes = adapter.dump_json(value)
          return FlextResult.ok(json_bytes.decode("utf-8"))
      except Exception as e:
          return FlextResult.fail(f"Serialization failed: {e}")

    @staticmethod
    def from_json[T](adapter: TypeAdapter[T], json_str: str) -> FlextResult[T]:
      """Deserialize value from JSON string.

      Args:
          adapter: TypeAdapter instance
          json_str: JSON string to deserialize

      Returns:
          FlextResult containing deserialized value or error

      """
      try:
          value = adapter.validate_json(json_str)
          return FlextResult.ok(value)
      except Exception as e:
          return FlextResult.fail(f"Deserialization failed: {e}")

    @staticmethod
    def to_dict[T](adapter: TypeAdapter[T], value: T) -> FlextResult[dict[str, object]]:
      """Serialize value to Python dictionary.

      Args:
          adapter: TypeAdapter instance
          value: Value to serialize

      Returns:
          FlextResult containing dictionary or error

      """
      try:
          result = adapter.dump_python(value)
          if isinstance(result, dict):
              return FlextResult.ok(result)
          return FlextResult.fail("Value did not serialize to dictionary")
      except Exception as e:
          return FlextResult.fail(f"Dictionary serialization failed: {e}")

    @staticmethod
    def from_dict[T](
      adapter: TypeAdapter[T],
      data: dict[str, object],
    ) -> FlextResult[T]:
      """Deserialize value from Python dictionary.

      Args:
          adapter: TypeAdapter instance
          data: Dictionary to deserialize

      Returns:
          FlextResult containing deserialized value or error

      """
      try:
          value = adapter.validate_python(data)
          return FlextResult.ok(value)
      except Exception as e:
          return FlextResult.fail(f"Dictionary deserialization failed: {e}")


# =============================================================================
# SCHEMA GENERATION HELPERS
# =============================================================================


class SchemaHelpers:
    """Helper functions for JSON schema generation using TypeAdapter."""

    @staticmethod
    def generate_schema[T](adapter: TypeAdapter[T]) -> FlextResult[dict[str, object]]:
      """Generate JSON schema for TypeAdapter.

      Args:
          adapter: TypeAdapter instance

      Returns:
          FlextResult containing JSON schema or error

      """
      try:
          schema = adapter.json_schema()
          return FlextResult.ok(schema)
      except Exception as e:
          return FlextResult.fail(f"Schema generation failed: {e}")

    @staticmethod
    def generate_multiple_schemas(
      adapters: dict[str, TypeAdapter[object]],
    ) -> FlextResult[dict[str, dict[str, object]]]:
      """Generate schemas for multiple TypeAdapters.

      Args:
          adapters: Dictionary mapping names to TypeAdapter instances

      Returns:
          FlextResult containing dictionary of schemas or error

      """
      try:
          schemas = {}
          for name, adapter in adapters.items():
              schema_result = SchemaHelpers.generate_schema(adapter)
              if schema_result.is_failure:
                  return FlextResult.fail(
                      f"Failed to generate schema for {name}: {schema_result.error}",
                  )
              schemas[name] = schema_result.unwrap()
          return FlextResult.ok(schemas)
      except Exception as e:
          return FlextResult.fail(f"Multiple schema generation failed: {e}")


# =============================================================================
# PRACTICAL USAGE EXAMPLES
# =============================================================================


class TypeAdapterExamples:
    """Practical examples showing TypeAdapter usage patterns.

    These examples demonstrate real-world usage of TypeAdapter
    patterns throughout the FLEXT ecosystem.
    """

    @staticmethod
    def user_validation_example() -> None:
      """Example: User data validation with TypeAdapter."""

      # Define user structure
      @dataclass(
          config=ConfigDict(
              extra="forbid",
              validate_assignment=True,
              str_strip_whitespace=True,
          ),
      )
      class User:
          name: str
          email: str
          age: int

          def __post_init__(self) -> None:
              # Business rule validation
              if self.age < 0:
                  msg = "Age cannot be negative"
                  raise ValueError(msg)

      # Create adapter
      user_adapter = TypeAdapter(User)

      # Validate from dictionary
      with contextlib.suppress(Exception):
          user_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
          user = user_adapter.validate_python(user_data)

      # Validate from JSON
      with contextlib.suppress(Exception):
          json_data = '{"name": "Jane Doe", "email": "jane@example.com", "age": 25}'
          user = user_adapter.validate_json(json_data)

      # Serialize to JSON
      with contextlib.suppress(Exception):
          user_adapter.dump_json(user)

    @staticmethod
    def configuration_validation_example() -> None:
      """Example: Configuration validation with TypeAdapter."""

      # Define configuration structure
      @dataclass(
          config=ConfigDict(
              extra="forbid",
              validate_assignment=True,
          ),
      )
      class DatabaseConfig:
          host: str
          port: int
          username: str
          password: str
          database: str

          def __post_init__(self) -> None:
              # Validation rules
              min_port = 1
              max_port = 65535
              if not (min_port <= self.port <= max_port):
                  msg = f"Port must be between {min_port} and {max_port}"
                  raise ValueError(msg)
              if not self.host.strip():
                  msg = "Host cannot be empty"
                  raise ValueError(msg)

      # Create adapter
      config_adapter = TypeAdapter(DatabaseConfig)

      # Load from environment-like dictionary
      config_data = {
          "host": "localhost",
          "port": 5432,
          "username": "postgres",
          "password": "secret",
          "database": "flext",
      }

      with contextlib.suppress(Exception):
          config_adapter.validate_python(config_data)

          # Generate schema for documentation
          config_adapter.json_schema()


# =============================================================================
# MIGRATION HELPERS
# =============================================================================


class MigrationHelpers:
    """Helpers for migrating from BaseModel to TypeAdapter patterns."""

    @staticmethod
    def convert_basemodel_to_dataclass(model_class: type) -> str:
      """Generate dataclass code from BaseModel definition.

      Args:
          model_class: BaseModel class to convert

      Returns:
          String containing equivalent dataclass definition

      """
      # This would analyze the BaseModel and generate equivalent dataclass code
      # Implementation would inspect fields, validators, etc.
      return f"# Convert {model_class.__name__} to dataclass with TypeAdapter"

    @staticmethod
    def create_adapter_for_legacy_model[T](model_class: type[T]) -> TypeAdapter[T]:
      """Create TypeAdapter for existing model class.

      Args:
          model_class: Existing model class

      Returns:
          TypeAdapter instance for the model

      """
      return TypeAdapter(model_class)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ConfigDict_Type",
    "ConnectionString",
    "EmailAddress",
    # Core types
    "EntityId",
    "ErrorCode",
    "ErrorMessage",
    "EventList",
    "Host",
    "Metadata",
    "MigrationHelpers",
    "Percentage",
    "Port",
    "SchemaHelpers",
    "SerializationHelpers",
    "ServiceName",
    "Timestamp",
    "TypeAdapterExamples",
    # Factory and helpers
    "TypeAdapterFactory",
    "ValidationAdapters",
    "Version",
]
