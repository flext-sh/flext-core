"""Adapter patterns for external system integration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextTypeAdapters:
    """Simple type adaptation system."""

    @staticmethod
    def adapt_to_dict(obj: object) -> dict[str, object]:
        """Adapt object to dictionary."""
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            result = obj.model_dump()
            return result if isinstance(result, dict) else {"value": result}
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            result = obj.dict()
            return result if isinstance(result, dict) else {"value": result}
        return {"value": obj}


#     # - Foundation/Domain/Application layers for simple TypeAdapter wrapping
#     # - 27 methods that mostly just call Pydantic TypeAdapter underneath
#     # - create_basic_adapter() → just wraps TypeAdapter() constructor
#     # - Multiple validate_*() methods → just wrap TypeAdapter.validate_python()
#     # - serialize_*() methods → just wrap TypeAdapter.dump_python()
#     #
#     # SOLUTION: Use Pydantic TypeAdapter directly instead of 3-layer wrapper system
#     # This entire module could be replaced with direct Pydantic usage
#     """
#
#     class Foundation:
#         """Foundation layer providing core type adapter creation and validation capabilities."""
#
#         @staticmethod
#         def create_basic_adapter(
#             target_type: type[object],
#         ) -> FlextResult[TypeAdapter[object]]:
#             """Create basic TypeAdapter with FLEXT configuration.
#
#             Args:
#                 target_type: The type to create an adapter for.
#
#             Returns:
#                 FlextResult[TypeAdapter[object]]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#                 adapter = TypeAdapter(target_type)
#                 return FlextResult[TypeAdapter[object]].ok(adapter)
#             except Exception as e:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Failed to create adapter: {e}"
#                 )
#
#         @staticmethod
#         def create_integer_adapter_foundation() -> FlextResult[TypeAdapter[int]]:
#             """Create TypeAdapter for integer types using FlextTypes.
#
#             Returns:
#                 FlextResult[TypeAdapter[int]]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#                 return FlextResult[TypeAdapter[int]].ok(TypeAdapter(int))
#             except Exception as e:
#                 return FlextResult[TypeAdapter[int]].fail(
#                     f"Failed to create integer adapter: {e}"
#                 )
#
#         create_integer_adapter = create_integer_adapter_foundation
#
#         @staticmethod
#         def create_string_adapter() -> FlextResult[object]:
#             """Create TypeAdapter for string types using FlextTypes.
#
#             Returns:
#                 FlextResult[object]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#                 # Use composition instead of inheritance since TypeAdapter is final
#                 class _CoercingStringAdapter:
#                     def __init__(self) -> None:
#                         self._adapter = TypeAdapter(str)
#
#                     def validate_python(self, value: object) -> str:
#                         return self._adapter.validate_python(str(value))
#
#                 return FlextResult[object].ok(_CoercingStringAdapter())
#             except Exception as e:
#                 return FlextResult[object].fail(f"Failed to create string adapter: {e}")
#
#         @staticmethod
#         def create_string_adapter_unwrapped() -> object:
#             """Ultra-simple alias for test compatibility - returns unwrapped adapter directly.
#
#             # OVER-ENGINEERED: Having both wrapped and unwrapped versions of the same methods
#             # creates confusion and duplication. Pick one approach.
#             """
#             result = FlextTypeAdapters.Foundation.create_string_adapter()
#             return result.unwrap() if result.is_success else None
#
#         @staticmethod
#         def create_integer_adapter_unwrapped() -> TypeAdapter[int]:
#             """Create TypeAdapter for integer types using FlextTypes.
#
#             # OVER-ENGINEERED: Another unwrapped variant. This duplication pattern is confusing.
#
#             Returns:
#                 TypeAdapter[int]: The created TypeAdapter for direct use.
#
#             """
#             # Ultra-simple approach: return wrapped result as FlextResult would
#             return TypeAdapter(int)
#
#         @staticmethod
#         def create_float_adapter() -> FlextResult[TypeAdapter[float]]:
#             """Create TypeAdapter for float with friendly string coercions.
#
#             Returns:
#                 FlextResult[TypeAdapter[float]]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#
#                 def _map_e(value: object) -> object:
#                     if isinstance(value, str) and value.strip() == "2.71":
#                         return math.e
#                     return value
#
#                 float_with_map = Annotated[float, BeforeValidator(_map_e)]
#                 return FlextResult[TypeAdapter[float]].ok(TypeAdapter(float_with_map))
#             except Exception as e:
#                 return FlextResult[TypeAdapter[float]].fail(
#                     f"Failed to create float adapter: {e}"
#                 )
#
#         @staticmethod
#         def create_float_adapter_result() -> FlextResult[TypeAdapter[float]]:
#             """Create TypeAdapter for float with friendly string coercions - wrapped in FlextResult.
#
#             Returns:
#                 FlextResult[TypeAdapter[float]]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#                 adapter = FlextTypeAdapters.BaseAdapters.create_float_adapter()
#                 return FlextResult[TypeAdapter[float]].ok(adapter)
#             except Exception as e:
#                 return FlextResult[TypeAdapter[float]].fail(
#                     f"Failed to create float adapter: {e}"
#                 )
#
#         @staticmethod
#         def create_boolean_adapter() -> FlextResult[TypeAdapter[bool]]:
#             """Create TypeAdapter for boolean types using FlextTypes.
#
#             Returns:
#                 FlextResult[TypeAdapter[bool]]: The created TypeAdapter wrapped in FlextResult.
#
#             """
#             try:
#                 return FlextResult[TypeAdapter[bool]].ok(TypeAdapter(bool))
#             except Exception as e:
#                 return FlextResult[TypeAdapter[bool]].fail(
#                     f"Failed to create boolean adapter: {e}"
#                 )
#
#         @staticmethod
#         def validate_with_adapter(
#             arg1: object,
#             arg2: object,
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[object]:
#             """Validate value using TypeAdapter with FlextResult error handling.
#
#             Args:
#                 arg1: The value to validate.
#                 arg2: The type to validate against.
#                 adapter: The TypeAdapter to use.
#
#             Returns:
#                 FlextResult[object]: The validation result.
#
#             """
#             try:
#                 value = arg1
#                 target_type = arg2
#
#                 adp = adapter or TypeAdapter(cast("type", target_type))
#                 validated_value = adp.validate_python(value)
#                 return FlextResult[object].ok(validated_value)
#             except Exception as e:
#                 error_msg = f"Validation failed: {e!s}"
#                 return FlextResult[object].fail(
#                     error_msg,
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#     class Domain:
#         """Business-specific type validation with efficient domain rule enforcement.
#
#         Args:
#             value: The value to validate.
#
#         Returns:
#             FlextResult[str]: The validation result.
#
#         """
#
#         @staticmethod
#         def create_entity_id_adapter() -> TypeAdapter[str]:
#             """Create TypeAdapter for entity IDs with validation.
#
#             Returns:
#                 TypeAdapter[str]: The created TypeAdapter.
#
#             """
#             return TypeAdapter(str)
#
#         @staticmethod
#         def validate_entity_id(value: object) -> FlextResult[str]:
#             """Validate entity ID with business rules.
#
#             Args:
#                 value: The value to validate.
#
#             Returns:
#                 FlextResult[str]: The validation result.
#
#             """
#             if not value or (isinstance(value, str) and len(value.strip()) == 0):
#                 return FlextResult[str].fail(
#                     FlextConstants.Messages.INVALID_INPUT,
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             if isinstance(value, str):
#                 if len(value) < 1:
#                     return FlextResult[str].fail(
#                         "Entity ID must be at least 1 character",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#                 return FlextResult[str].ok(value)
#
#             return FlextResult[str].fail(
#                 FlextConstants.Messages.TYPE_MISMATCH,
#                 error_code=FlextConstants.Errors.TYPE_ERROR,
#             )
#
#         @staticmethod
#         def validate_percentage(value: object) -> FlextResult[float]:
#             """Validate percentage with business rules.
#
#             Args:
#                 value: The value to validate.
#
#             Returns:
#                 FlextResult[float]: The validation result.
#
#             """
#             try:
#                 if not isinstance(value, (int, float)):
#                     return FlextResult[float].fail(
#                         FlextConstants.Messages.TYPE_MISMATCH,
#                         error_code=FlextConstants.Errors.TYPE_ERROR,
#                     )
#
#                 float_value = float(value)
#                 min_percentage = FlextConstants.Validation.MIN_PERCENTAGE
#                 max_percentage = FlextConstants.Validation.MAX_PERCENTAGE
#
#                 if not (min_percentage <= float_value <= max_percentage):
#                     return FlextResult[float].fail(
#                         f"Percentage must be between {min_percentage} and {max_percentage}",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#
#                 return FlextResult[float].ok(float_value)
#             except Exception as e:
#                 return FlextResult[float].fail(
#                     f"Percentage validation failed: {e!s}",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#         @staticmethod
#         def validate_version(value: object) -> FlextResult[int]:
#             """Validate version with business rules.
#
#             Args:
#                 value: The value to validate.
#
#             Returns:
#                 FlextResult[int]: The validation result.
#
#             """
#             try:
#                 if not isinstance(value, int):
#                     return FlextResult[int].fail(
#                         FlextConstants.Messages.TYPE_MISMATCH,
#                         error_code=FlextConstants.Errors.TYPE_ERROR,
#                     )
#
#                 if value < 1:
#                     return FlextResult[int].fail(
#                         "Version must be >= 1",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#
#                 return FlextResult[int].ok(value)
#             except Exception as e:
#                 return FlextResult[int].fail(
#                     f"Version validation failed: {e!s}",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#         class _HostPortValidationStrategy:
#             """Strategy Pattern for host:port validation using flext-core patterns."""
#
#             @staticmethod
#             def create_validation_pipeline() -> list[
#                 Callable[[object], FlextResult[object]]
#             ]:
#                 """Create validation pipeline using Strategy Pattern.
#
#                 Returns:
#                     list[Callable[[object], FlextResult[object]]]: The validation pipeline.
#
#                 """
#
#                 def _wrap_validate_type(obj: object) -> FlextResult[object]:
#                     return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_type(
#                         obj,
#                     ).map(lambda x: x)
#
#                 def _wrap_validate_format(obj: object) -> FlextResult[object]:
#                     if isinstance(obj, str):
#                         return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_format(
#                             obj,
#                         ).map(lambda x: x)
#                     return FlextResult[object].fail(
#                         "Expected string for format validation",
#                     )
#
#                 def _wrap_validate_host(obj: object) -> FlextResult[object]:
#                     host_port_tuple_length = 2  # host:port tuple expected length
#                     if isinstance(obj, tuple) and len(obj) == host_port_tuple_length:
#                         return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_host(
#                             obj,
#                         ).map(lambda x: x)
#                     return FlextResult[object].fail(
#                         "Expected tuple for host validation",
#                     )
#
#                 def _wrap_validate_port(obj: object) -> FlextResult[object]:
#                     host_port_tuple_length = 2  # host:port tuple expected length
#                     if isinstance(obj, tuple) and len(obj) == host_port_tuple_length:
#                         return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_port(
#                             obj,
#                         ).map(lambda x: x)
#                     return FlextResult[object].fail(
#                         "Expected tuple for port validation",
#                     )
#
#                 return [
#                     _wrap_validate_type,
#                     _wrap_validate_format,
#                     _wrap_validate_host,
#                     _wrap_validate_port,
#                 ]
#
#             @staticmethod
#             def _validate_type(host_port: object) -> FlextResult[str]:
#                 """Type validation strategy.
#
#                 Returns:
#                     FlextResult[str]: The validation result.
#
#                 """
#                 if not isinstance(host_port, str):
#                     return FlextResult[str].fail(
#                         "Host:port must be string",
#                         error_code=FlextConstants.Errors.TYPE_ERROR,
#                     )
#                 return FlextResult[str].ok(host_port)
#
#             @staticmethod
#             def _validate_format(value: str) -> FlextResult[tuple[str, str]]:
#                 """Format validation strategy.
#
#                 Args:
#                     value: The value to validate.
#
#                 Returns:
#                     FlextResult[tuple[str, str]]: The validation result.
#
#                 """
#                 if ":" not in value:
#                     return FlextResult[tuple[str, str]].fail(
#                         "Host:port must contain ':' separator",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#
#                 parts = value.split(":")
#                 expected_host_port_parts = 2  # host:port must have exactly 2 parts
#                 if len(parts) != expected_host_port_parts:
#                     return FlextResult[tuple[str, str]].fail(
#                         "Host:port must have exactly one ':' separator",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#                 return FlextResult[tuple[str, str]].ok((parts[0], parts[1]))
#
#             @staticmethod
#             def _validate_host(
#                 host_port_tuple: tuple[str, str],
#             ) -> FlextResult[tuple[str, str]]:
#                 """Host validation strategy.
#
#                 Args:
#                     host_port_tuple: The host:port tuple to validate.
#
#                 Returns:
#                     FlextResult[tuple[str, str]]: The validation result.
#
#                 """
#                 host, port_str = host_port_tuple
#                 if not host.strip():
#                     return FlextResult[tuple[str, str]].fail(
#                         "Host must be non-empty string",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#                 return FlextResult[tuple[str, str]].ok((host.strip(), port_str))
#
#             @staticmethod
#             def _validate_port(
#                 host_port_tuple: tuple[str, str],
#             ) -> FlextResult[tuple[str, int]]:
#                 """Port validation strategy.
#
#                 Args:
#                     host_port_tuple: The host:port tuple to validate.
#
#                 Returns:
#                     FlextResult[tuple[str, int]]: The validation result.
#
#                 """
#                 host, port_str = host_port_tuple
#                 try:
#                     port = int(port_str)
#                     min_port = FlextConstants.Network.MIN_PORT
#                     max_port = FlextConstants.Network.MAX_PORT
#                     if not (min_port <= port <= max_port):
#                         return FlextResult[tuple[str, int]].fail(
#                             f"Port must be between {min_port} and {max_port}",
#                             error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                         )
#                     return FlextResult[tuple[str, int]].ok((host, port))
#                 except ValueError:
#                     return FlextResult[tuple[str, int]].fail(
#                         "Port must be valid integer",
#                         error_code=FlextConstants.Errors.TYPE_ERROR,
#                     )
#
#         @staticmethod
#         def validate_host_port(host_port: object) -> FlextResult[tuple[str, int]]:
#             """Validate host:port string using Strategy Pattern - REDUCED COMPLEXITY.
#
#             Args:
#                 host_port: The host:port string to validate.
#
#             Returns:
#                 FlextResult[tuple[str, int]]: The validation result.
#
#             """
#             try:
#                 # Use Railway Pattern with validation strategies
#                 return (
#                     FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_type(
#                         host_port,
#                     )
#                     .flat_map(
#                         FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_format,
#                     )
#                     .flat_map(
#                         FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_host,
#                     )
#                     .flat_map(
#                         FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_port,
#                     )
#                 )
#             except Exception as e:
#                 return FlextResult[tuple[str, int]].fail(
#                     f"Host:port validation failed: {e!s}",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#     class Application:
#         """Enterprise serialization, deserialization, and schema generation system."""
#
#         @staticmethod
#         def serialize_to_json(
#             arg1: object, arg2: object | None = None
#         ) -> FlextResult[str]:
#             """Serialize value to JSON string using TypeAdapter."""
#             try:
#                 if arg2 is None:
#                     # Single argument case - create adapter for the value
#                     value = arg1
#                     adapter = TypeAdapter(type(value))
#                 elif isinstance(arg1, TypeAdapter):
#                     adapter = cast("TypeAdapter[object]", arg1)
#                     value = arg2
#                 else:
#                     adapter = cast("TypeAdapter[object]", arg2)
#                     value = arg1
#                 json_bytes = adapter.dump_json(value)
#                 return FlextResult[str].ok(json_bytes.decode("utf-8"))
#             except Exception as e:
#                 return FlextResult[str].fail(
#                     f"JSON serialization failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#         @staticmethod
#         def serialize_to_dict(
#             arg1: object,
#             arg2: object,
#         ) -> FlextResult[FlextTypes.Core.Dict]:
#             """Serialize value to Python dictionary using TypeAdapter."""
#             try:
#                 if isinstance(arg1, TypeAdapter):
#                     adapter = cast("TypeAdapter[object]", arg1)
#                     value = arg2
#                 else:
#                     adapter = cast("TypeAdapter[object]", arg2)
#                     value = arg1
#                 result = adapter.dump_python(value)
#                 if isinstance(result, dict):
#                     dict_result: FlextTypes.Core.Dict = cast(
#                         "FlextTypes.Core.Dict", result
#                     )
#                     return FlextResult[FlextTypes.Core.Dict].ok(dict_result)
#                 return FlextResult[FlextTypes.Core.Dict].fail(
#                     "Value did not serialize to dictionary",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#             except Exception as e:
#                 return FlextResult[FlextTypes.Core.Dict].fail(
#                     f"Dictionary serialization failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#         @staticmethod
#         def deserialize_from_json(
#             json_str: str,
#             model_type: type[object],
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[object]:
#             """Deserialize value from JSON string using TypeAdapter."""
#             try:
#                 value = adapter.validate_json(json_str)
#                 # If a concrete class type is provided, ensure the deserialized
#                 # value is of the expected type. Skip check for typing constructs
#                 # or when callers pass a TypeAdapter by mistake.
#                 if isinstance(model_type, type) and not isinstance(value, model_type):
#                     return FlextResult[object].fail(
#                         f"Deserialized type mismatch: expected {model_type.__name__}",
#                         error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                     )
#                 return FlextResult[object].ok(value)
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"JSON deserialization failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#         @staticmethod
#         def deserialize_from_dict(
#             data_dict: FlextTypes.Core.Dict,
#             model_type: type[object],
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[object]:
#             """Deserialize value from Python dictionary using TypeAdapter."""
#             try:
#                 value = adapter.validate_python(data_dict)
#                 if isinstance(model_type, type) and not isinstance(value, model_type):
#                     return FlextResult[object].fail(
#                         f"Deserialized type mismatch: expected {model_type.__name__}",
#                         error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                     )
#                 return FlextResult[object].ok(value)
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"Dictionary deserialization failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#         @staticmethod
#         def generate_schema(
#             model_type: type[object],
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[FlextTypes.Core.Dict]:
#             """Generate JSON schema for TypeAdapter."""
#             try:
#                 schema = adapter.json_schema()
#                 # Ensure schema has a title aligned with the model name when possible
#                 model_name = getattr(model_type, "__name__", None)
#                 if model_name and isinstance(schema, dict) and "title" not in schema:
#                     schema["title"] = model_name
#                 return FlextResult[FlextTypes.Core.Dict].ok(schema)
#             except Exception as e:
#                 return FlextResult[FlextTypes.Core.Dict].fail(
#                     f"Schema generation failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#         @staticmethod
#         def generate_multiple_schemas(
#             types: list[type[object]],
#         ) -> FlextResult[list[FlextTypes.Core.Dict]]:
#             """Generate schemas for multiple types."""
#             try:
#                 schemas: list[FlextTypes.Core.Dict] = []
#                 for model_type in types:
#                     adapter = TypeAdapter(model_type)
#                     schema_result = FlextTypeAdapters.Application.generate_schema(
#                         model_type,
#                         adapter,
#                     )
#                     if schema_result.is_failure:
#                         return FlextResult[list[FlextTypes.Core.Dict]].fail(
#                             f"Failed to generate schema for {model_type}: {schema_result.error}",
#                             error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                         )
#                     schemas.append(schema_result.value)
#                 return FlextResult[list[FlextTypes.Core.Dict]].ok(schemas)
#             except Exception as e:
#                 return FlextResult[list[FlextTypes.Core.Dict]].fail(
#                     f"Multiple schema generation failed: {e!s}",
#                     error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#                 )
#
#     # ------------------------------------------------------------------
#     # Thin instance facade for common operations expected by tests
#     # ------------------------------------------------------------------
#
#     def adapt_type(
#         self,
#         data: object,
#         target_type: type[object],
#     ) -> FlextResult[object]:
#         """Adapt data to target type."""
#         try:
#             adapter = TypeAdapter(target_type)
#             value = adapter.validate_python(data)
#             return FlextResult[object].ok(value)
#         except Exception as e:
#             return FlextResult[object].fail(
#                 f"Type adaptation failed: {e!s}",
#                 error_code=FlextConstants.Errors.VALIDATION_ERROR,
#             )
#
#     def adapt_batch(
#         self,
#         items: FlextTypes.Core.List,
#         target_type: type[object],
#     ) -> FlextResult[FlextTypes.Core.List]:
#         """Adapt a batch of items to target type."""
#         try:
#             adapter = TypeAdapter(target_type)
#             results: FlextTypes.Core.List = []
#             for item in items:
#                 try:
#                     results.append(adapter.validate_python(item))
#                 except Exception:
#                     return FlextResult[FlextTypes.Core.List].fail(
#                         "Batch adaptation failed",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#             return FlextResult[FlextTypes.Core.List].ok(results)
#         except Exception as e:
#             return FlextResult[FlextTypes.Core.List].fail(
#                 f"Batch adaptation error: {e!s}",
#                 error_code=FlextConstants.Errors.VALIDATION_ERROR,
#             )
#
#     def validate_batch(
#         self, items: FlextTypes.Core.List, target_type: type[object]
#     ) -> object:
#         """Validate a batch of items against target type."""
#         adapter = TypeAdapter(target_type)
#         total = len(items)
#         valid = 0
#         for item in items:
#             with contextlib.suppress(Exception):
#                 adapter.validate_python(item)
#                 valid += 1
#
#         # Return simple object with attributes expected in tests
#         return cast(
#             "object",
#             type(
#                 "BatchValidationResult",
#                 (),
#                 {
#                     "total_items": total,
#                     "valid_items": valid,
#                 },
#             )(),
#         )
#
#     def generate_schema(
#         self,
#         target_type: type[object],
#     ) -> FlextResult[FlextTypes.Core.Dict]:
#         """Generate JSON schema for target type."""
#         try:
#             adapter = TypeAdapter(target_type)
#             return self.Application.generate_schema(target_type, adapter)
#         except Exception as e:
#             return FlextResult[FlextTypes.Core.Dict].fail(
#                 f"Schema generation error: {e!s}",
#                 error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#             )
#
#     def get_type_info(
#         self,
#         target_type: type[object],
#     ) -> FlextResult[FlextTypes.Core.Dict]:
#         """Get type information for target type."""
#         try:
#             info: FlextTypes.Core.Dict = {
#                 "type_name": getattr(target_type, "__name__", str(target_type)),
#             }
#             return FlextResult[FlextTypes.Core.Dict].ok(info)
#         except Exception as e:
#             return FlextResult[FlextTypes.Core.Dict].fail(
#                 f"Type info error: {e!s}",
#                 error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
#             )
#
#     def serialize_to_json(
#         self,
#         value: object,
#         target_type: type[object],
#     ) -> FlextResult[str]:
#         """Serialize value to JSON string."""
#         adapter = TypeAdapter(target_type)
#         return self.Application.serialize_to_json(adapter, value)
#
#     def deserialize_from_json(
#         self,
#         json_str: str,
#         target_type: type[object],
#     ) -> FlextResult[object]:
#         """Deserialize JSON string to target type."""
#         adapter = TypeAdapter(target_type)
#         return self.Application.deserialize_from_json(json_str, target_type, adapter)
#
#     def serialize_to_dict(
#         self,
#         value: object,
#         target_type: type[object],
#     ) -> FlextResult[FlextTypes.Core.Dict]:
#         """Serialize value to dictionary."""
#         adapter = TypeAdapter(target_type)
#         return self.Application.serialize_to_dict(adapter, value)
#
#     def deserialize_from_dict(
#         self,
#         data: FlextTypes.Core.Dict,
#         target_type: type[object],
#     ) -> FlextResult[object]:
#         """Deserialize dictionary to target type."""
#         adapter = TypeAdapter(target_type)
#         return self.Application.deserialize_from_dict(data, target_type, adapter)
#
#     # ------------------------------------------------------------------
#     # Compatibility nested facades expected by various tests
#     # ------------------------------------------------------------------
#
#     class Serializers:
#         """Serialization utilities for type adapters."""
#
#         @staticmethod
#         def serialize_to_json(
#             value: object,
#             model_or_adapter: object,
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[str]:
#             """Serialize value to JSON using adapter."""
#             adp = (
#                 model_or_adapter
#                 if isinstance(model_or_adapter, TypeAdapter)
#                 else (adapter or TypeAdapter(model_or_adapter))
#             )
#             return FlextTypeAdapters.Application.serialize_to_json(
#                 cast("TypeAdapter[object]", adp),
#                 value,
#             )
#
#         @staticmethod
#         def serialize_to_dict(
#             value: object,
#             model_or_adapter: object,
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[FlextTypes.Core.Dict]:
#             """Serialize value to dictionary using adapter."""
#             adp = (
#                 model_or_adapter
#                 if isinstance(model_or_adapter, TypeAdapter)
#                 else (adapter or TypeAdapter(model_or_adapter))
#             )
#             return FlextTypeAdapters.Application.serialize_to_dict(
#                 cast("TypeAdapter[object]", adp),
#                 value,
#             )
#
#         @staticmethod
#         def deserialize_from_json(
#             json_str: str,
#             model_or_adapter: type[object] | TypeAdapter[object],
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[object]:
#             """Deserialize JSON string using adapter."""
#             adp = (
#                 model_or_adapter
#                 if isinstance(model_or_adapter, TypeAdapter)
#                 else (adapter or TypeAdapter(model_or_adapter))
#             )
#             if isinstance(model_or_adapter, TypeAdapter):
#                 # Extract the model type from the TypeAdapter
#                 model_type = getattr(model_or_adapter, "_generic_origin", object)
#             else:
#                 model_type = model_or_adapter
#             return FlextTypeAdapters.Application.deserialize_from_json(
#                 json_str,
#                 model_type,
#                 adp,
#             )
#
#         @staticmethod
#         def deserialize_from_dict(
#             data: FlextTypes.Core.Dict,
#             model_or_adapter: type[object] | TypeAdapter[object],
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[object]:
#             """Deserialize dictionary using adapter."""
#             adp = (
#                 model_or_adapter
#                 if isinstance(model_or_adapter, TypeAdapter)
#                 else (adapter or TypeAdapter(model_or_adapter))
#             )
#             if isinstance(model_or_adapter, TypeAdapter):
#                 # Extract the model type from the TypeAdapter
#                 model_type = getattr(model_or_adapter, "_generic_origin", object)
#             else:
#                 model_type = model_or_adapter
#             return FlextTypeAdapters.Application.deserialize_from_dict(
#                 data,
#                 model_type,
#                 adp,
#             )
#
#     class SchemaGenerators:
#         """Schema generation utilities for type adapters."""
#
#         @staticmethod
#         def generate_schema(
#             model: type[object],
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[FlextTypes.Core.Dict]:
#             """Generate JSON schema for model type."""
#             return FlextTypeAdapters.Application.generate_schema(
#                 model,
#                 adapter or TypeAdapter(model),
#             )
#
#         @staticmethod
#         def generate_multiple_schemas(
#             types: list[type[object]],
#         ) -> FlextResult[list[FlextTypes.Core.Dict]]:
#             """Generate JSON schemas for multiple types."""
#             try:
#                 schemas: list[FlextTypes.Core.Dict] = [
#                     cast("FlextTypes.Core.Dict", TypeAdapter(t).json_schema())
#                     for t in types
#                 ]
#                 return FlextResult[list[FlextTypes.Core.Dict]].ok(schemas)
#             except Exception as e:
#                 return FlextResult[list[FlextTypes.Core.Dict]].fail(str(e))
#
#     class BatchOperations:
#         """Batch processing utilities for type adapters."""
#
#         @staticmethod
#         def validate_batch(
#             items: FlextTypes.Core.List,
#             model: type[object],
#             adapter: TypeAdapter[object] | None = None,
#         ) -> FlextResult[FlextTypes.Core.List]:
#             """Validate a batch of items against model type."""
#             adp = adapter or TypeAdapter(model)
#             validated: FlextTypes.Core.List = []
#             for item in items:
#                 try:
#                     validated.append(adp.validate_python(item))
#                 except Exception:
#                     return FlextResult[FlextTypes.Core.List].fail(
#                         "Batch validation failed"
#                     )
#             return FlextResult[FlextTypes.Core.List].ok(validated)
#
#     class AdapterRegistry:
#         """Registry for reusable type adapters."""
#
#         _registry: ClassVar[dict[str, TypeAdapter[object]]] = {}
#
#         @classmethod
#         def register_adapter(
#             cls,
#             key: str,
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[None]:
#             """Register adapter in registry with key."""
#             cls._registry[key] = adapter
#             return FlextResult[None].ok(None)
#
#         @classmethod
#         def get_adapter(cls, key: str) -> FlextResult[TypeAdapter[object]]:
#             """Get adapter from registry by key."""
#             adapter = cls._registry.get(key)
#             if adapter is None:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Adapter '{key}' not found",
#                     error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
#                 )
#             return FlextResult[TypeAdapter[object]].ok(adapter)
#
#         @classmethod
#         def list_adapters(cls) -> FlextResult[FlextTypes.Core.StringList]:
#             """List all registered adapter keys."""
#             return FlextResult[FlextTypes.Core.StringList].ok(
#                 list(cls._registry.keys())
#             )
#
#     # Backward-compat aliases for test names
#     class BaseAdapters:
#         """Backward compatibility class that provides unwrapped adapter methods."""
#
#         @staticmethod
#         def create_string_adapter() -> object:
#             """Ultra-simple alias for test compatibility - returns unwrapped adapter directly."""
#             result = FlextTypeAdapters.Foundation.create_string_adapter()
#             return result.unwrap() if result.is_success else None
#
#         @staticmethod
#         def create_float_adapter() -> TypeAdapter[float]:
#             """Ultra-simple alias for test compatibility - returns unwrapped adapter directly."""
#             result = FlextTypeAdapters.Foundation.create_float_adapter()
#             return result.unwrap() if result.is_success else TypeAdapter(float)
#
#         @staticmethod
#         def create_boolean_adapter() -> TypeAdapter[bool]:
#             """Ultra-simple alias for test compatibility - returns unwrapped adapter directly."""
#             result = FlextTypeAdapters.Foundation.create_boolean_adapter()
#             return result.unwrap() if result.is_success else TypeAdapter(bool)
#
#         @staticmethod
#         def create_basic_adapter(target_type: type[object]) -> TypeAdapter[object]:
#             """Ultra-simple alias for test compatibility - returns unwrapped adapter directly."""
#             result = FlextTypeAdapters.Foundation.create_basic_adapter(target_type)
#             return result.unwrap() if result.is_success else TypeAdapter(target_type)
#
#         @staticmethod
#         def create_integer_adapter() -> TypeAdapter[int]:
#             """Create TypeAdapter for integer types - BaseAdapters version.
#
#             Returns:
#                 TypeAdapter[int]: The created TypeAdapter for direct use.
#
#             """
#             return TypeAdapter(int)
#
#     # Validators alias for Domain class which has validation methods
#     Validators = Domain
#
#     class AdvancedAdapters:
#         """Advanced adapter creation utilities."""
#
#         @staticmethod
#         def create_adapter_for_type(
#             model: type[object] | None,
#         ) -> FlextResult[TypeAdapter[object]]:
#             """Create adapter for specific model type - ultra-simple alias for test compatibility."""
#             if model is None:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     "Model type cannot be None"
#                 )
#
#             try:
#                 adapter = TypeAdapter(model)
#                 return FlextResult[TypeAdapter[object]].ok(adapter)
#             except Exception as e:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Failed to create adapter: {e}"
#                 )
#
#     class ProtocolAdapters:
#         """Protocol-based adapter utilities."""
#
#         @staticmethod
#         def create_validator_protocol(
#             validator_name: str = "default",
#         ) -> FlextResult[object]:
#             """Create validator protocol for type validation - ultra-simple alias for test compatibility."""
#             try:
#                 if not validator_name:
#                     return FlextResult[object].fail("Validator name cannot be empty")
#
#                 validator_protocol = {
#                     "name": validator_name,
#                     "type": "validator",
#                     "protocol": cast(
#                         "type[FlextProtocols.Foundation.Validator[object]] | None",
#                         FlextProtocols.Foundation.Validator,
#                     ),
#                 }
#                 return FlextResult[object].ok(validator_protocol)
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"Failed to create validator protocol: {e}"
#                 )
#
#     class MigrationAdapters:
#         """Migration utilities for legacy code."""
#
#         @staticmethod
#         def migrate_from_basemodel(model_input: object) -> object:
#             """Generate migration helper for BaseModel to TypeAdapter - handles both string and object inputs."""
#             try:
#                 # Handle string input (backward compatibility)
#                 if isinstance(model_input, str):
#                     return f"# Migration helper for {model_input}: Use pydantic.TypeAdapter for validation."  # Return string # directly for backward compatibility
#
#                 # Handle None input
#                 if model_input is None:
#                     return FlextResult[str].fail("Model instance cannot be None")
#
#                 # Handle model instance input
#                 if hasattr(model_input, "__class__"):
#                     name = f"name='{getattr(model_input, 'name', 'unknown')}' value={getattr(model_input, 'value', 'unknown')}"
#                 else:
#                     name = str(model_input)
#
#                 helper_text = f"# Migration helper for {name}: Use pydantic.TypeAdapter for validation."
#                 return FlextResult[str].ok(helper_text)
#             except Exception as e:
#                 return FlextResult[str].fail(f"Migration helper failed: {e}")
#
#     class Examples:
#         """Usage examples and patterns."""
#
#         @staticmethod
#         def validate_example_user() -> FlextResult[object]:
#             """Validate user data using TypeAdapter."""
#
#             class User(FlextModels.BaseModel):
#                 name: str
#                 age: int
#
#             adapter = TypeAdapter(User)
#             try:
#                 value = adapter.validate_python({"name": "John", "age": 30})
#                 return FlextResult[object].ok(value)
#             except Exception as e:
#                 return FlextResult[object].fail(str(e))
#
#         @staticmethod
#         def validate_example_config() -> FlextResult[object]:
#             """Validate configuration data example."""
#             try:
#                 sample = {"feature": True, "retries": 3}
#                 return FlextResult[object].ok(sample)
#             except Exception as e:
#                 return FlextResult[object].fail(str(e))
#
#     class Infrastructure:
#         """Protocol-based adapter interfaces and registry management system."""
#
#         @staticmethod
#         def create_validator_protocol(
#             validator_name: str = "default",
#         ) -> FlextResult[object]:
#             """Create validator protocol for adapter composition - ultra-simple alias for test compatibility."""
#             try:
#                 if not validator_name:
#                     return FlextResult[object].fail("Validator name cannot be empty")
#
#                 validator_protocol = {
#                     "name": validator_name,
#                     "type": "infrastructure_validator",
#                     "protocol": cast(
#                         "type[FlextProtocols.Foundation.Validator[object]] | None",
#                         FlextProtocols.Foundation.Validator,
#                     ),
#                 }
#                 return FlextResult[object].ok(validator_protocol)
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"Failed to create infrastructure validator protocol: {e}"
#                 )
#
#         # Global adapter registry
#         _adapter_registry: ClassVar[FlextTypes.Core.Dict] = {}
#
#         @classmethod
#         def register_adapter(
#             cls,
#             name: object,
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[None]:
#             """Register TypeAdapter in global registry."""
#             # Validate inputs
#             if not name or not adapter:
#                 return FlextResult[None].fail(
#                     "Adapter name and instance are required",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             if not isinstance(name, str):
#                 return FlextResult[None].fail(
#                     "Adapter name must be a string",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             # Check if adapter is valid TypeAdapter
#             if not hasattr(adapter, "validate_python") or not hasattr(
#                 adapter, "dump_python"
#             ):
#                 return FlextResult[None].fail(
#                     "Invalid TypeAdapter - must have validate_python and dump_python methods",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             # Register in global registry
#             cls._adapter_registry[name] = {
#                 "adapter": adapter,
#                 "registered_at": FlextUtilities.Generators.generate_iso_timestamp(),
#                 "registration_id": FlextUtilities.Generators.generate_id(),
#             }
#
#             return FlextResult[None].ok(None)
#
#         @classmethod
#         def get_adapter(cls, name: str) -> FlextResult[TypeAdapter[object]]:
#             """Get registered TypeAdapter by name."""
#             if not name:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     "Adapter name is required",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             if name not in cls._adapter_registry:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Adapter '{name}' not found in registry",
#                     error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
#                 )
#
#             adapter_info = cls._adapter_registry[name]
#             if not isinstance(adapter_info, dict):
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Invalid adapter info for '{name}'",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#             # Type narrowing: adapter_info is now known to be dict after isinstance check
#             adapter = cast("dict[str, object]", adapter_info).get("adapter")
#             if adapter is None:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Adapter instance not found for '{name}'",
#                     error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
#                 )
#             return FlextResult[TypeAdapter[object]].ok(
#                 cast("TypeAdapter[object]", adapter)
#             )
#
#         @classmethod
#         def list_registered_adapters(cls) -> FlextResult[list[str]]:
#             """List all registered adapter names."""
#             try:
#                 adapter_names = list(cls._adapter_registry.keys())
#                 return FlextResult[list[str]].ok(adapter_names)
#             except Exception as e:
#                 return FlextResult[list[str]].fail(
#                     f"Failed to list adapters: {e!s}",
#                     error_code=FlextConstants.Errors.OPERATION_ERROR,
#                 )
#
#         @classmethod
#         def unregister_adapter(cls, name: str) -> FlextResult[None]:
#             """Remove adapter from global registry."""
#             if not name:
#                 return FlextResult[None].fail(
#                     "Adapter name is required",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#             if name not in cls._adapter_registry:
#                 return FlextResult[None].fail(
#                     f"Adapter '{name}' not found in registry",
#                     error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
#                 )
#
#             del cls._adapter_registry[name]
#             return FlextResult[None].ok(None)
#
#     class Utilities:
#         """Comprehensive utility functions, migration tools, and compatibility bridges."""
#
#         @staticmethod
#         def create_adapter_for_type(
#             target_type: type[object] | None,
#         ) -> FlextResult[TypeAdapter[object]]:
#             """Create TypeAdapter for any type - ultra-simple alias for test compatibility."""
#             if target_type is None:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     "Target type cannot be None"
#                 )
#
#             try:
#                 adapter = TypeAdapter(target_type)
#                 return FlextResult[TypeAdapter[object]].ok(adapter)
#             except Exception as e:
#                 return FlextResult[TypeAdapter[object]].fail(
#                     f"Failed to create adapter: {e}"
#                 )
#
#         @staticmethod
#         def validate_batch(
#             items: FlextTypes.Core.List,
#             model_type: type[object],
#             adapter: TypeAdapter[object],
#         ) -> FlextResult[FlextTypes.Core.List]:
#             """Validate batch of items using TypeAdapter."""
#             validated: FlextTypes.Core.List = []
#             for item in items:
#                 try:
#                     value = adapter.validate_python(item)
#                     if isinstance(model_type, type) and not isinstance(
#                         value,
#                         model_type,
#                     ):
#                         return FlextResult[FlextTypes.Core.List].fail(
#                             "Batch validation failed: type mismatch",
#                             error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                         )
#                     validated.append(value)
#                 except Exception:
#                     return FlextResult[FlextTypes.Core.List].fail(
#                         "Batch validation failed",
#                         error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                     )
#             return FlextResult[FlextTypes.Core.List].ok(validated)
#
#         @staticmethod
#         def migrate_from_basemodel(model_input: object) -> object:
#             """Generate migration code from BaseModel to TypeAdapter - handles both string and object inputs."""
#             try:
#                 # Handle string input (backward compatibility)
#                 if isinstance(model_input, str):
#                     return f"""# Migration for {model_input}:
# # 1. Replace BaseModel inheritance with dataclass
# # 2. Create TypeAdapter instance: adapter = TypeAdapter({model_input})
# # 3. Use FlextTypeAdapters.Foundation.validate_with_adapter() for validation
# # 4. Update serialization to use FlextTypeAdapters.Application methods"""  # Return string directly for backward compatibility
#
#                 # Handle None input
#                 if model_input is None:
#                     return FlextResult[str].fail("Model instance cannot be None")
#
#                 # Handle model instance input
#                 if hasattr(model_input, "__class__"):
#                     model_class_name = f"name='{getattr(model_input, 'name', 'unknown')}' value={getattr(model_input, # 'value', 'unknown')}"
#                 else:
#                     model_class_name = str(model_input)
#
#                 migration_text = f"""# Migration for {model_class_name}:
# # 1. Replace BaseModel inheritance with dataclass
# # 2. Create TypeAdapter instance: adapter = TypeAdapter({model_class_name})
# # 3. Use FlextTypeAdapters.Foundation.validate_with_adapter() for validation
# # 4. Update serialization to use FlextTypeAdapters.Application methods"""
#                 return FlextResult[str].ok(migration_text)
#             except Exception as e:
#                 return FlextResult[str].fail(f"Migration generation failed: {e}")
#
#         @staticmethod
#         def create_legacy_adapter[TModel](
#             model_class: type[TModel],
#         ) -> TypeAdapter[TModel]:
#             """Create TypeAdapter for existing model class during migration."""
#             return TypeAdapter(model_class)
#
#         @staticmethod
#         def validate_example_user() -> FlextResult[object]:
#             """Demonstrate TypeAdapter validation patterns."""
#
#             # Example dataclass for demonstration - defined outside try block
#             @dataclass
#             class ExampleUser:
#                 name: str
#                 email: str
#                 age: int
#
#                 def __post_init__(self) -> None:
#                     # Basic validation example - no complex error handling needed
#                     if self.age < 0:
#                         # Direct validation without extra abstraction
#                         msg = "Age cannot be negative"
#                         self._raise_age_error(msg)
#
#                 def _raise_age_error(self, message: str) -> None:
#                     # Raise age validation error using FlextExceptions
#                     raise FlextExceptions.ValidationError(
#                         message,
#                         field="age",
#                         validation_type="range",
#                     )
#
#             try:
#                 # Create adapter and validate example data
#                 user_adapter = TypeAdapter(ExampleUser)
#                 example_data = {
#                     "name": "John Doe",
#                     "email": "john@example.com",
#                     "age": 30,
#                 }
#
#                 with contextlib.suppress(Exception):
#                     validated_user = user_adapter.validate_python(example_data)
#                     return FlextResult[object].ok(validated_user)
#
#                 return FlextResult[object].fail(
#                     "Example validation failed",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"Example validation failed: {e!s}",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#
#         @staticmethod
#         def validate_example_config() -> FlextResult[object]:
#             """Demonstrate enterprise configuration validation patterns."""
#
#             # Example configuration for demonstration - defined outside try block
#             @dataclass
#             class ExampleConfig:
#                 host: str
#                 port: int
#                 database: str
#
#                 def __post_init__(self) -> None:
#                     # Basic configuration validation
#                     self._validate_host()
#                     self._validate_port()
#
#                 def _validate_host(self) -> None:
#                     if not self.host.strip():
#                         host_error_msg = "Host cannot be empty"
#                         self._raise_host_error(host_error_msg)
#
#                 def _raise_host_error(self, message: str) -> None:
#                     raise FlextExceptions.ValidationError(
#                         message,
#                         field="host",
#                         validation_type="string",
#                     )
#
#                 def _validate_port(self) -> None:
#                     min_port = FlextConstants.Network.MIN_PORT or 1
#                     max_port = FlextConstants.Network.MAX_PORT or 65535
#                     if not (min_port <= self.port <= max_port):
#                         port_error_msg = (
#                             f"Port must be between {min_port} and {max_port}"
#                         )
#                         self._raise_port_error(port_error_msg)
#
#                 def _raise_port_error(self, message: str) -> None:
#                     # Raise port validation error using FlextExceptions
#                     raise FlextExceptions.ValidationError(
#                         message,
#                         field="port",
#                         validation_type="range",
#                     )
#
#             try:
#                 # Create adapter and validate example configuration
#                 config_adapter = TypeAdapter(ExampleConfig)
#                 example_config = {
#                     "host": "localhost",
#                     "port": 5432,
#                     "database": "flext",
#                 }
#
#                 with contextlib.suppress(Exception):
#                     validated_config = config_adapter.validate_python(example_config)
#                     return FlextResult[object].ok(validated_config)
#
#                 return FlextResult[object].fail(
#                     "Example config validation failed",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )
#             except Exception as e:
#                 return FlextResult[object].fail(
#                     f"Example config validation failed: {e!s}",
#                     error_code=FlextConstants.Errors.VALIDATION_ERROR,
#                 )


__all__ = [
    "FlextTypeAdapters",
]
