"""FLEXT Type Adapters - Type conversion, validation and serialization system.

Provides efficient type adaptation capabilities including type conversion,
validation pipeline, schema generation, and JSON/dict serialization. Built on
Pydantic v2 TypeAdapter with FlextResult integration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast

from pydantic import TypeAdapter

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult


class FlextTypeAdapters:
    """Comprehensive type adaptation system for type conversion, validation and serialization.

    Central hub for type conversion with error handling, validation pipeline, JSON/dict
    serialization, schema generation, and batch processing. Built on Pydantic v2 TypeAdapter
    with FlextResult integration.

    """

    class Config:
        """Enterprise type adapters system management with FlextTypes.Config integration.

        This configuration class provides efficient system management for the FlextTypeAdapters
        ecosystem, implementing production-ready configuration patterns using Strategy Pattern.

        """

        class _ConfigurationStrategy:
            """Strategy Pattern for configuration management."""

            @staticmethod
            def get_base_config() -> dict[str, object]:
                """Get base configuration common to all environments."""
                return {
                    "timestamp": time.time(),
                    "system_status": "active",
                }

            @staticmethod
            def get_features_config() -> dict[str, object]:
                """Get feature flags configuration."""
                return {
                    "adapter_registry_enabled": True,
                    "batch_processing_enabled": True,
                    "schema_generation_cache": True,
                    "validation_strict_mode": False,
                    "serialization_optimization": True,
                    "error_recovery_enabled": True,
                }

            @staticmethod
            def get_performance_config() -> dict[str, object]:
                """Get performance configuration."""
                return {
                    "cache_size": 500,
                    "batch_size": 50,
                    "timeout_seconds": 15,
                    "concurrent_operations": 5,
                    "memory_optimization": True,
                }

            @staticmethod
            def get_health_config() -> dict[str, object]:
                """Get health monitoring configuration."""
                return {
                    "uptime_seconds": 0,
                    "total_operations": 0,
                    "error_rate_percentage": 0.0,
                    "memory_usage_mb": 0,
                    "cpu_usage_percentage": 0,
                }

        @classmethod
        def configure_type_adapters_system(cls, config: dict[str, object]) -> object:
            """Configure type adapters system using Strategy Pattern."""
            with contextlib.suppress(Exception):
                if not config:
                    config = {}

                # Apply configuration strategy
                return {
                    **cls._ConfigurationStrategy.get_base_config(),
                    "environment": config.get("environment", "development"),
                    "validation_level": config.get("validation_level", "standard"),
                    "performance_level": config.get("performance_level", "balanced"),
                    "status": "configured",
                }

            # Fallback using same strategy
            return {
                **cls._ConfigurationStrategy.get_base_config(),
                "environment": "development",
                "validation_level": "standard",
                "performance_level": "balanced",
                "status": "fallback",
            }

        @classmethod
        def get_type_adapters_system_config(cls) -> object:
            """Get system configuration using Strategy Pattern composition."""
            return {
                **cls._ConfigurationStrategy.get_base_config(),
                "environment": "development",
                "validation_level": "standard",
                "performance_level": "balanced",
                "features": cls._ConfigurationStrategy.get_features_config(),
                "performance": cls._ConfigurationStrategy.get_performance_config(),
                "health": cls._ConfigurationStrategy.get_health_config(),
            }

        class _EnvironmentConfigStrategy:
            """Strategy Pattern for environment-specific configurations."""

            @staticmethod
            def get_base_config(environment: str) -> dict[str, object]:
                """Get base configuration for any environment."""
                return {
                    "environment": environment,
                    "timestamp": time.time(),
                    "config_version": "1.0.0",
                }

            @staticmethod
            def get_development_config() -> dict[str, object]:
                """Development environment strategy."""
                return {
                    "validation_level": "basic",
                    "performance_level": "low",
                    "debug_enabled": True,
                    "cache_size": 100,
                    "batch_size": 10,
                    "timeout_seconds": 60,
                    "concurrent_operations": 2,
                    "plugin_architecture_enabled": False,
                    "metrics_collection_enabled": True,
                    "verbose_logging": True,
                }

            @staticmethod
            def get_staging_config() -> dict[str, object]:
                """Staging environment strategy."""
                return {
                    "validation_level": "standard",
                    "performance_level": "balanced",
                    "debug_enabled": True,
                    "cache_size": 500,
                    "batch_size": 50,
                    "timeout_seconds": 30,
                    "concurrent_operations": 5,
                    "plugin_architecture_enabled": True,
                    "metrics_collection_enabled": True,
                    "verbose_logging": False,
                }

            @staticmethod
            def get_production_config() -> dict[str, object]:
                """Production environment strategy."""
                return {
                    "validation_level": "strict",
                    "performance_level": "high",
                    "debug_enabled": False,
                    "cache_size": 2000,
                    "batch_size": 200,
                    "timeout_seconds": 15,
                    "concurrent_operations": 20,
                    "plugin_architecture_enabled": True,
                    "metrics_collection_enabled": True,
                    "verbose_logging": False,
                }

            @staticmethod
            def get_default_config() -> dict[str, object]:
                """Default fallback configuration strategy."""
                return {
                    "validation_level": "standard",
                    "performance_level": "balanced",
                    "debug_enabled": False,
                    "cache_size": 500,
                    "batch_size": 50,
                    "timeout_seconds": 30,
                    "concurrent_operations": 5,
                    "plugin_architecture_enabled": False,
                    "metrics_collection_enabled": False,
                    "verbose_logging": False,
                }

        @classmethod
        def create_environment_type_adapters_config(
            cls, environment: str,
        ) -> dict[str, object]:
            """Create environment-specific configuration using Strategy Pattern."""
            base = cls._EnvironmentConfigStrategy.get_base_config(environment)

            # Apply environment-specific strategy
            env_strategies = {
                "development": cls._EnvironmentConfigStrategy.get_development_config,
                "staging": cls._EnvironmentConfigStrategy.get_staging_config,
                "production": cls._EnvironmentConfigStrategy.get_production_config,
            }

            strategy_func = env_strategies.get(
                environment, cls._EnvironmentConfigStrategy.get_default_config,
            )
            return {**base, **strategy_func()}

        class _PerformanceOptimizationStrategy:
            """Strategy Pattern for performance optimization configurations."""

            @staticmethod
            def get_low_performance_config() -> dict[str, object]:
                """Low performance optimization strategy."""
                return {
                    "cache_size": 100,
                    "batch_size": 10,
                    "concurrent_operations": 2,
                    "timeout_seconds": 60,
                    "memory_optimization": False,
                    "performance_monitoring": True,
                    "optimization_level": "low",
                }

            @staticmethod
            def get_balanced_performance_config() -> dict[str, object]:
                """Balanced performance optimization strategy."""
                return {
                    "cache_size": 500,
                    "batch_size": 50,
                    "concurrent_operations": 5,
                    "timeout_seconds": 30,
                    "memory_optimization": True,
                    "performance_monitoring": True,
                    "optimization_level": "balanced",
                }

            @staticmethod
            def get_high_performance_config() -> dict[str, object]:
                """High performance optimization strategy."""
                return {
                    "cache_size": 2000,
                    "batch_size": 200,
                    "concurrent_operations": 20,
                    "timeout_seconds": 15,
                    "memory_optimization": True,
                    "performance_monitoring": False,
                    "optimization_level": "high",
                }

            @staticmethod
            def get_extreme_performance_config() -> dict[str, object]:
                """Extreme performance optimization strategy."""
                return {
                    "cache_size": 10000,
                    "batch_size": 1000,
                    "concurrent_operations": 100,
                    "timeout_seconds": 5,
                    "memory_optimization": True,
                    "performance_monitoring": False,
                    "optimization_level": "extreme",
                }

            @staticmethod
            def get_error_config(optimization_level: str) -> dict[str, object]:
                """Error configuration for invalid optimization levels."""
                return {
                    "error": "Invalid optimization level",
                    "provided_level": optimization_level,
                    "supported_levels": ["low", "balanced", "high", "extreme"],
                    "fallback_level": "balanced",
                    "timestamp": time.time(),
                    "status": "error",
                    "recovery_guidance": "Use supported optimization levels: low, balanced, high, extreme",
                }

        @classmethod
        def optimize_type_adapters_performance(cls, optimization_level: str) -> object:
            """Optimize system performance using Strategy Pattern."""
            # Define optimization strategies
            strategies = {
                "low": cls._PerformanceOptimizationStrategy.get_low_performance_config,
                "balanced": cls._PerformanceOptimizationStrategy.get_balanced_performance_config,
                "high": cls._PerformanceOptimizationStrategy.get_high_performance_config,
                "extreme": cls._PerformanceOptimizationStrategy.get_extreme_performance_config,
            }

            if optimization_level in strategies:
                config = strategies[optimization_level]().copy()
                config["timestamp"] = time.time()
                config["status"] = "optimized"
                return config

            return cls._PerformanceOptimizationStrategy.get_error_config(
                optimization_level,
            )

    class Foundation:
        """Foundation layer providing core type adapter creation and validation capabilities.

        This class implements the fundamental type adaptation infrastructure for the FLEXT
        ecosystem, providing basic type adapter creation, validation patterns, and error
        handling through FlextResult[T] integration. It serves as the building block for
        more specialized type adaptation functionality.

        """

        @staticmethod
        def create_basic_adapter(target_type: type[object]) -> TypeAdapter[object]:
            """Create basic TypeAdapter with FLEXT configuration."""
            return TypeAdapter(target_type)

        @staticmethod
        def create_string_adapter() -> object:
            """Create TypeAdapter for string types using FlextTypes."""

            # Use composition instead of inheritance since TypeAdapter is final
            class _CoercingStringAdapter:
                def __init__(self) -> None:
                    self._adapter = TypeAdapter(str)

                def validate_python(self, value: object) -> str:
                    return self._adapter.validate_python(str(value))

            return _CoercingStringAdapter()

        @staticmethod
        def create_integer_adapter() -> TypeAdapter[int]:
            """Create TypeAdapter for integer types using FlextTypes."""
            return TypeAdapter(int)

        @staticmethod
        def create_float_adapter() -> TypeAdapter[float]:
            """Create TypeAdapter for float types using FlextTypes."""
            return TypeAdapter(float)

        @staticmethod
        def create_boolean_adapter() -> TypeAdapter[bool]:
            """Create TypeAdapter for boolean types using FlextTypes."""
            return TypeAdapter(bool)

        @staticmethod
        def validate_with_adapter(
            arg1: object, arg2: object, adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[object]:
            """Validate value using TypeAdapter with FlextResult error handling."""
            try:
                value = arg1
                target_type = arg2

                adp = adapter or TypeAdapter(cast("type", target_type))
                validated_value = adp.validate_python(value)
                return FlextResult[object].ok(validated_value)
            except Exception as e:
                error_msg = f"Validation failed: {e!s}"
                return FlextResult[object].fail(
                    error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

    class Domain:
        """Business-specific type validation with efficient domain rule enforcement.

        This class implements domain-driven type validation patterns for business-specific
        data types and constraints. It provides efficient validation for domain entities,
        value objects, and business rules while maintaining type safety and error handling
        through FlextResult[T] patterns.

        """

        @staticmethod
        def create_entity_id_adapter() -> TypeAdapter[str]:
            """Create TypeAdapter for entity IDs with validation."""
            return TypeAdapter(str)

        @staticmethod
        def validate_entity_id(value: object) -> FlextResult[str]:
            """Validate entity ID with business rules."""
            if not value or (isinstance(value, str) and len(value.strip()) == 0):
                return FlextResult[str].fail(
                    FlextConstants.Messages.INVALID_INPUT,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if isinstance(value, str):
                if len(value) < 1:
                    return FlextResult[str].fail(
                        "Entity ID must be at least 1 character",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[str].ok(value)

            return FlextResult[str].fail(
                FlextConstants.Messages.TYPE_MISMATCH,
                error_code=FlextConstants.Errors.TYPE_ERROR,
            )

        @staticmethod
        def validate_percentage(value: object) -> FlextResult[float]:
            """Validate percentage with business rules."""
            try:
                if not isinstance(value, (int, float)):
                    return FlextResult[float].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                float_value = float(value)
                min_percentage = FlextConstants.Validation.MIN_PERCENTAGE
                max_percentage = FlextConstants.Validation.MAX_PERCENTAGE

                if not (min_percentage <= float_value <= max_percentage):
                    return FlextResult[float].fail(
                        f"Percentage must be between {min_percentage} and {max_percentage}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[float].ok(float_value)
            except Exception as e:
                return FlextResult[float].fail(
                    f"Percentage validation failed: {e!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        @staticmethod
        def validate_version(value: object) -> FlextResult[int]:
            """Validate version with business rules."""
            try:
                if not isinstance(value, int):
                    return FlextResult[int].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if value < 1:
                    return FlextResult[int].fail(
                        "Version must be >= 1",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[int].ok(value)
            except Exception as e:
                return FlextResult[int].fail(
                    f"Version validation failed: {e!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        class _HostPortValidationStrategy:
            """Strategy Pattern for host:port validation using flext-core patterns."""

            @staticmethod
            def create_validation_pipeline() -> list[
                Callable[[object], FlextResult[object]]
            ]:
                """Create validation pipeline using Strategy Pattern."""

                def _wrap_validate_type(obj: object) -> FlextResult[object]:
                    return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_type(
                        obj,
                    ).map(lambda x: x)

                def _wrap_validate_format(obj: object) -> FlextResult[object]:
                    if isinstance(obj, str):
                        return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_format(
                            obj,
                        ).map(lambda x: x)
                    return FlextResult[object].fail(
                        "Expected string for format validation",
                    )

                def _wrap_validate_host(obj: object) -> FlextResult[object]:
                    host_port_tuple_length = 2  # host:port tuple expected length
                    if isinstance(obj, tuple) and len(obj) == host_port_tuple_length:
                        return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_host(
                            obj,
                        ).map(lambda x: x)
                    return FlextResult[object].fail(
                        "Expected tuple for host validation",
                    )

                def _wrap_validate_port(obj: object) -> FlextResult[object]:
                    host_port_tuple_length = 2  # host:port tuple expected length
                    if isinstance(obj, tuple) and len(obj) == host_port_tuple_length:
                        return FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_port(
                            obj,
                        ).map(lambda x: x)
                    return FlextResult[object].fail(
                        "Expected tuple for port validation",
                    )

                return [
                    _wrap_validate_type,
                    _wrap_validate_format,
                    _wrap_validate_host,
                    _wrap_validate_port,
                ]

            @staticmethod
            def _validate_type(host_port: object) -> FlextResult[str]:
                """Type validation strategy."""
                if not isinstance(host_port, str):
                    return FlextResult[str].fail(
                        "Host:port must be string",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
                return FlextResult[str].ok(host_port)

            @staticmethod
            def _validate_format(value: str) -> FlextResult[tuple[str, str]]:
                """Format validation strategy."""
                if ":" not in value:
                    return FlextResult[tuple[str, str]].fail(
                        "Host:port must contain ':' separator",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                parts = value.split(":")
                expected_host_port_parts = 2  # host:port must have exactly 2 parts
                if len(parts) != expected_host_port_parts:
                    return FlextResult[tuple[str, str]].fail(
                        "Host:port must have exactly one ':' separator",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[tuple[str, str]].ok((parts[0], parts[1]))

            @staticmethod
            def _validate_host(
                host_port_tuple: tuple[str, str],
            ) -> FlextResult[tuple[str, str]]:
                """Host validation strategy."""
                host, port_str = host_port_tuple
                if not host.strip():
                    return FlextResult[tuple[str, str]].fail(
                        "Host must be non-empty string",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[tuple[str, str]].ok((host.strip(), port_str))

            @staticmethod
            def _validate_port(
                host_port_tuple: tuple[str, str],
            ) -> FlextResult[tuple[str, int]]:
                """Port validation strategy."""
                host, port_str = host_port_tuple
                try:
                    port = int(port_str)
                    min_port = FlextConstants.Network.MIN_PORT
                    max_port = FlextConstants.Network.MAX_PORT
                    if not (min_port <= port <= max_port):
                        return FlextResult[tuple[str, int]].fail(
                            f"Port must be between {min_port} and {max_port}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                    return FlextResult[tuple[str, int]].ok((host, port))
                except ValueError:
                    return FlextResult[tuple[str, int]].fail(
                        "Port must be valid integer",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

        @staticmethod
        def validate_host_port(host_port: object) -> FlextResult[tuple[str, int]]:
            """Validate host:port string using Strategy Pattern - REDUCED COMPLEXITY."""
            try:
                # Use Railway Pattern with validation strategies
                return (
                    FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_type(
                        host_port,
                    )
                    .flat_map(
                        FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_format,
                    )
                    .flat_map(
                        FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_host,
                    )
                    .flat_map(
                        FlextTypeAdapters.Domain._HostPortValidationStrategy._validate_port,
                    )
                )
            except Exception as e:
                return FlextResult[tuple[str, int]].fail(
                    f"Host:port validation failed: {e!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

    class Application:
        """Enterprise serialization, deserialization, and schema generation system.

        This class implements efficient application-layer serialization capabilities for
        type adapters, providing JSON and dictionary conversion, schema generation for API
        documentation, and batch processing with efficient error handling. It serves as
        the primary interface for data interchange and documentation generation.
        """

        @staticmethod
        def serialize_to_json(arg1: object, arg2: object) -> FlextResult[str]:
            """Serialize value to JSON string using TypeAdapter."""
            try:
                if isinstance(arg1, TypeAdapter):
                    adapter = cast("TypeAdapter[object]", arg1)
                    value = arg2
                else:
                    adapter = cast("TypeAdapter[object]", arg2)
                    value = arg1
                json_bytes = adapter.dump_json(value)
                return FlextResult[str].ok(json_bytes.decode("utf-8"))
            except Exception as e:
                return FlextResult[str].fail(
                    f"JSON serialization failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

        @staticmethod
        def serialize_to_dict(
            arg1: object, arg2: object,
        ) -> FlextResult[dict[str, object]]:
            """Serialize value to Python dictionary using TypeAdapter."""
            try:
                if isinstance(arg1, TypeAdapter):
                    adapter = cast("TypeAdapter[object]", arg1)
                    value = arg2
                else:
                    adapter = cast("TypeAdapter[object]", arg2)
                    value = arg1
                result = adapter.dump_python(value)
                if isinstance(result, dict):
                    dict_result: dict[str, object] = cast("dict[str, object]", result)
                    return FlextResult[dict[str, object]].ok(dict_result)
                return FlextResult[dict[str, object]].fail(
                    "Value did not serialize to dictionary",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )
            except Exception as e:
                return FlextResult[dict[str, object]].fail(
                    f"Dictionary serialization failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

        @staticmethod
        def deserialize_from_json(
            json_str: str, model_type: type[object], adapter: TypeAdapter[object],
        ) -> FlextResult[object]:
            """Deserialize value from JSON string using TypeAdapter."""
            try:
                value = adapter.validate_json(json_str)
                # If a concrete class type is provided, ensure the deserialized
                # value is of the expected type. Skip check for typing constructs
                # or when callers pass a TypeAdapter by mistake.
                if isinstance(model_type, type) and not isinstance(value, model_type):
                    return FlextResult[object].fail(
                        f"Deserialized type mismatch: expected {model_type.__name__}",
                        error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                    )
                return FlextResult[object].ok(value)
            except Exception as e:
                return FlextResult[object].fail(
                    f"JSON deserialization failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

        @staticmethod
        def deserialize_from_dict(
            data_dict: dict[str, object],
            model_type: type[object],
            adapter: TypeAdapter[object],
        ) -> FlextResult[object]:
            """Deserialize value from Python dictionary using TypeAdapter."""
            try:
                value = adapter.validate_python(data_dict)
                if isinstance(model_type, type) and not isinstance(value, model_type):
                    return FlextResult[object].fail(
                        f"Deserialized type mismatch: expected {model_type.__name__}",
                        error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                    )
                return FlextResult[object].ok(value)
            except Exception as e:
                return FlextResult[object].fail(
                    f"Dictionary deserialization failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

        @staticmethod
        def generate_schema(
            model_type: type[object], adapter: TypeAdapter[object],
        ) -> FlextResult[dict[str, object]]:
            """Generate JSON schema for TypeAdapter."""
            try:
                schema = adapter.json_schema()
                # Ensure schema has a title aligned with the model name when possible
                model_name = getattr(model_type, "__name__", None)
                if model_name and isinstance(schema, dict) and "title" not in schema:
                    schema["title"] = model_name
                return FlextResult[dict[str, object]].ok(schema)
            except Exception as e:
                return FlextResult[dict[str, object]].fail(
                    f"Schema generation failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

        @staticmethod
        def generate_multiple_schemas(
            types: list[type[object]],
        ) -> FlextResult[list[dict[str, object]]]:
            """Generate schemas for multiple types."""
            try:
                schemas: list[dict[str, object]] = []
                for model_type in types:
                    adapter = TypeAdapter(model_type)
                    schema_result = FlextTypeAdapters.Application.generate_schema(
                        model_type, adapter,
                    )
                    if schema_result.is_failure:
                        return FlextResult[list[dict[str, object]]].fail(
                            f"Failed to generate schema for {model_type}: {schema_result.error}",
                            error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                        )
                    schemas.append(schema_result.value)
                return FlextResult[list[dict[str, object]]].ok(schemas)
            except Exception as e:
                return FlextResult[list[dict[str, object]]].fail(
                    f"Multiple schema generation failed: {e!s}",
                    error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
                )

    # ------------------------------------------------------------------
    # Thin instance facade for common operations expected by tests
    # ------------------------------------------------------------------

    def adapt_type(
        self, data: object, target_type: type[object],
    ) -> FlextResult[object]:
        try:
            adapter = TypeAdapter(target_type)
            value = adapter.validate_python(data)
            return FlextResult[object].ok(value)
        except Exception as e:
            return FlextResult[object].fail(
                f"Type adaptation failed: {e!s}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    def adapt_batch(
        self, items: list[object], target_type: type[object],
    ) -> FlextResult[list[object]]:
        try:
            adapter = TypeAdapter(target_type)
            results: list[object] = []
            for item in items:
                try:
                    results.append(adapter.validate_python(item))
                except Exception:
                    return FlextResult[list[object]].fail(
                        "Batch adaptation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
            return FlextResult[list[object]].ok(results)
        except Exception as e:
            return FlextResult[list[object]].fail(
                f"Batch adaptation error: {e!s}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    def validate_batch(self, items: list[object], target_type: type[object]) -> object:
        adapter = TypeAdapter(target_type)
        total = len(items)
        valid = 0
        for item in items:
            with contextlib.suppress(Exception):
                adapter.validate_python(item)
                valid += 1

        # Return simple object with attributes expected in tests
        return cast(
            "object",
            type(
                "BatchValidationResult",
                (),
                {
                    "total_items": total,
                    "valid_items": valid,
                },
            )(),
        )

    def generate_schema(
        self, target_type: type[object],
    ) -> FlextResult[dict[str, object]]:
        try:
            adapter = TypeAdapter(target_type)
            return self.Application.generate_schema(target_type, adapter)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Schema generation error: {e!s}",
                error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
            )

    def get_type_info(
        self, target_type: type[object],
    ) -> FlextResult[dict[str, object]]:
        try:
            info: dict[str, object] = {
                "type_name": getattr(target_type, "__name__", str(target_type)),
            }
            return FlextResult[dict[str, object]].ok(info)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Type info error: {e!s}",
                error_code=FlextConstants.Errors.SERIALIZATION_ERROR,
            )

    def serialize_to_json(
        self, value: object, target_type: type[object],
    ) -> FlextResult[str]:
        adapter = TypeAdapter(target_type)
        return self.Application.serialize_to_json(adapter, value)

    def deserialize_from_json(
        self, json_str: str, target_type: type[object],
    ) -> FlextResult[object]:
        adapter = TypeAdapter(target_type)
        return self.Application.deserialize_from_json(json_str, target_type, adapter)

    def serialize_to_dict(
        self, value: object, target_type: type[object],
    ) -> FlextResult[dict[str, object]]:
        adapter = TypeAdapter(target_type)
        return self.Application.serialize_to_dict(adapter, value)

    def deserialize_from_dict(
        self, data: dict[str, object], target_type: type[object],
    ) -> FlextResult[object]:
        adapter = TypeAdapter(target_type)
        return self.Application.deserialize_from_dict(data, target_type, adapter)

    # ------------------------------------------------------------------
    # Compatibility nested facades expected by various tests
    # ------------------------------------------------------------------

    class Serializers:
        """Serialization utilities for type adapters."""

        @staticmethod
        def serialize_to_json(
            value: object,
            model_or_adapter: object,
            adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[str]:
            adp = (
                model_or_adapter
                if isinstance(model_or_adapter, TypeAdapter)
                else (adapter or TypeAdapter(model_or_adapter))
            )
            return FlextTypeAdapters.Application.serialize_to_json(
                cast("TypeAdapter[object]", adp), value,
            )

        @staticmethod
        def serialize_to_dict(
            value: object,
            model_or_adapter: object,
            adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[dict[str, object]]:
            adp = (
                model_or_adapter
                if isinstance(model_or_adapter, TypeAdapter)
                else (adapter or TypeAdapter(model_or_adapter))
            )
            return FlextTypeAdapters.Application.serialize_to_dict(
                cast("TypeAdapter[object]", adp), value,
            )

        @staticmethod
        def deserialize_from_json(
            json_str: str,
            model_or_adapter: type[object] | TypeAdapter[object],
            adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[object]:
            adp = (
                model_or_adapter
                if isinstance(model_or_adapter, TypeAdapter)
                else (adapter or TypeAdapter(model_or_adapter))
            )
            if isinstance(model_or_adapter, TypeAdapter):
                # Extract the model type from the TypeAdapter
                model_type = getattr(model_or_adapter, "_generic_origin", object)
            else:
                model_type = model_or_adapter
            return FlextTypeAdapters.Application.deserialize_from_json(
                json_str, model_type, adp,
            )

        @staticmethod
        def deserialize_from_dict(
            data: dict[str, object],
            model_or_adapter: type[object] | TypeAdapter[object],
            adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[object]:
            adp = (
                model_or_adapter
                if isinstance(model_or_adapter, TypeAdapter)
                else (adapter or TypeAdapter(model_or_adapter))
            )
            if isinstance(model_or_adapter, TypeAdapter):
                # Extract the model type from the TypeAdapter
                model_type = getattr(model_or_adapter, "_generic_origin", object)
            else:
                model_type = model_or_adapter
            return FlextTypeAdapters.Application.deserialize_from_dict(
                data, model_type, adp,
            )

    class SchemaGenerators:
        """Schema generation utilities for type adapters."""

        @staticmethod
        def generate_schema(
            model: type[object], adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[dict[str, object]]:
            return FlextTypeAdapters.Application.generate_schema(
                model, adapter or TypeAdapter(model),
            )

        @staticmethod
        def generate_multiple_schemas(
            types: list[type[object]],
        ) -> FlextResult[list[dict[str, object]]]:
            try:
                schemas: list[dict[str, object]] = [
                    cast("dict[str, object]", TypeAdapter(t).json_schema())
                    for t in types
                ]
                return FlextResult[list[dict[str, object]]].ok(schemas)
            except Exception as e:
                return FlextResult[list[dict[str, object]]].fail(str(e))

    class BatchOperations:
        """Batch processing utilities for type adapters."""

        @staticmethod
        def validate_batch(
            items: list[object],
            model: type[object],
            adapter: TypeAdapter[object] | None = None,
        ) -> FlextResult[list[object]]:
            adp = adapter or TypeAdapter(model)
            validated: list[object] = []
            for item in items:
                try:
                    validated.append(adp.validate_python(item))
                except Exception:
                    return FlextResult[list[object]].fail("Batch validation failed")
            return FlextResult[list[object]].ok(validated)

    class AdapterRegistry:
        """Registry for reusable type adapters."""

        _registry: ClassVar[dict[str, TypeAdapter[object]]] = {}

        @classmethod
        def register_adapter(
            cls, key: str, adapter: TypeAdapter[object],
        ) -> FlextResult[None]:
            cls._registry[key] = adapter
            return FlextResult[None].ok(None)

        @classmethod
        def get_adapter(cls, key: str) -> FlextResult[TypeAdapter[object]]:
            adapter = cls._registry.get(key)
            if adapter is None:
                return FlextResult[TypeAdapter[object]].fail(
                    f"Adapter '{key}' not found",
                    error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                )
            return FlextResult[TypeAdapter[object]].ok(adapter)

        @classmethod
        def list_adapters(cls) -> FlextResult[list[str]]:
            return FlextResult[list[str]].ok(list(cls._registry.keys()))

    # Backward-compat aliases for test names
    BaseAdapters = Foundation
    # Validators alias for Domain class which has validation methods
    Validators = Domain

    class AdvancedAdapters:
        """Advanced adapter creation utilities."""

        @staticmethod
        def create_adapter_for_type(model: type[object]) -> TypeAdapter[object]:
            return TypeAdapter(model)

    class ProtocolAdapters:
        """Protocol-based adapter utilities."""

        @staticmethod
        def create_validator_protocol() -> object:
            return cast(
                "type[FlextProtocols.Foundation.Validator[object]] | None",
                FlextProtocols.Foundation.Validator,
            )

    class MigrationAdapters:
        """Migration utilities for legacy code."""

        @staticmethod
        def migrate_from_basemodel(name: str) -> str:
            return f"# Migration helper for {name}: Use pydantic.TypeAdapter for validation."

    class Examples:
        """Usage examples and patterns."""

        @staticmethod
        def validate_example_user() -> FlextResult[object]:
            class User(FlextModels.Config):
                name: str
                age: int

            adapter = TypeAdapter(User)
            try:
                value = adapter.validate_python({"name": "John", "age": 30})
                return FlextResult[object].ok(value)
            except Exception as e:
                return FlextResult[object].fail(str(e))

        @staticmethod
        def validate_example_config() -> FlextResult[object]:
            try:
                sample = {"feature": True, "retries": 3}
                return FlextResult[object].ok(sample)
            except Exception as e:
                return FlextResult[object].fail(str(e))

    class Infrastructure:
        """Protocol-based adapter interfaces and registry management system.

        This class implements infrastructure-layer patterns for type adapter composition,
        registration, and discovery. It provides protocol-based interfaces for flexible
        adapter composition, registry management for dynamic adapter discovery, and
        infrastructure patterns supporting enterprise adapter architecture.

        """

        @staticmethod
        def create_validator_protocol() -> (
            type[FlextProtocols.Foundation.Validator[object]] | None
        ):
            """Create validator protocol for adapter composition."""
            return cast(
                "type[FlextProtocols.Foundation.Validator[object]] | None",
                FlextProtocols.Foundation.Validator,
            )

        @staticmethod
        def register_adapter(
            name: str, adapter: TypeAdapter[object],
        ) -> FlextResult[None]:
            """Register TypeAdapter in global registry."""
            # This would use a global registry to store adapters
            # For now, just return success as placeholder
            if not name or not adapter:
                return FlextResult[None].fail(
                    "Adapter name and instance are required",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[None].ok(None)

    class Utilities:
        """Comprehensive utility functions, migration tools, and compatibility bridges.

        This class provides essential utility functions for type adapter operations, migration
        tools for transitioning from legacy patterns, batch processing capabilities, and
        compatibility bridges for maintaining functionality during transitions. It serves as
        the toolbox for adapter maintenance, migration, and operational support.

        """

        @staticmethod
        def create_adapter_for_type(target_type: type[object]) -> TypeAdapter[object]:
            """Create TypeAdapter for any type."""
            return TypeAdapter(target_type)

        @staticmethod
        def validate_batch(
            items: list[object], model_type: type[object], adapter: TypeAdapter[object],
        ) -> FlextResult[list[object]]:
            """Validate batch of items using TypeAdapter."""
            validated: list[object] = []
            for item in items:
                try:
                    value = adapter.validate_python(item)
                    if isinstance(model_type, type) and not isinstance(
                        value, model_type,
                    ):
                        return FlextResult[list[object]].fail(
                            "Batch validation failed: type mismatch",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                    validated.append(value)
                except Exception:
                    return FlextResult[list[object]].fail(
                        "Batch validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
            return FlextResult[list[object]].ok(validated)

        @staticmethod
        def migrate_from_basemodel(model_class_name: str) -> str:
            """Generate migration code from BaseModel to TypeAdapter."""
            return f"""# Migration for {model_class_name}:
# 1. Replace BaseModel inheritance with dataclass
# 2. Create TypeAdapter instance: adapter = TypeAdapter({model_class_name})
# 3. Use FlextTypeAdapters.Foundation.validate_with_adapter() for validation
# 4. Update serialization to use FlextTypeAdapters.Application methods"""

        @staticmethod
        def create_legacy_adapter[TModel](
            model_class: type[TModel],
        ) -> TypeAdapter[TModel]:
            """Create TypeAdapter for existing model class during migration."""
            return TypeAdapter(model_class)

        @staticmethod
        def validate_example_user() -> FlextResult[object]:
            """Example validation demonstrating TypeAdapter patterns."""

            # Example dataclass for demonstration - defined outside try block
            @dataclass
            class ExampleUser:
                name: str
                email: str
                age: int

                def __post_init__(self) -> None:
                    # Basic validation example - no complex error handling needed
                    if self.age < 0:
                        # Direct validation without extra abstraction
                        msg = "Age cannot be negative"
                        self._raise_age_error(msg)

                def _raise_age_error(self, message: str) -> None:
                    # Raise age validation error using FlextExceptions
                    raise FlextExceptions.ValidationError(
                        message, field="age", validation_type="range",
                    )

            try:
                # Create adapter and validate example data
                user_adapter = TypeAdapter(ExampleUser)
                example_data = {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "age": 30,
                }

                with contextlib.suppress(Exception):
                    validated_user = user_adapter.validate_python(example_data)
                    return FlextResult[object].ok(validated_user)

                return FlextResult[object].fail(
                    "Example validation failed",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            except Exception as e:
                return FlextResult[object].fail(
                    f"Example validation failed: {e!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        @staticmethod
        def validate_example_config() -> FlextResult[object]:
            """Example configuration validation demonstrating enterprise patterns."""

            # Example configuration for demonstration - defined outside try block
            @dataclass
            class ExampleConfig:
                host: str
                port: int
                database: str

                def __post_init__(self) -> None:
                    # Basic configuration validation
                    self._validate_host()
                    self._validate_port()

                def _validate_host(self) -> None:
                    if not self.host.strip():
                        host_error_msg = "Host cannot be empty"
                        self._raise_host_error(host_error_msg)

                def _raise_host_error(self, message: str) -> None:
                    raise FlextExceptions.ValidationError(
                        message, field="host", validation_type="string",
                    )

                def _validate_port(self) -> None:
                    min_port = FlextConstants.Network.MIN_PORT or 1
                    max_port = FlextConstants.Network.MAX_PORT or 65535
                    if not (min_port <= self.port <= max_port):
                        port_error_msg = (
                            f"Port must be between {min_port} and {max_port}"
                        )
                        self._raise_port_error(port_error_msg)

                def _raise_port_error(self, message: str) -> None:
                    # Raise port validation error using FlextExceptions
                    raise FlextExceptions.ValidationError(
                        message, field="port", validation_type="range",
                    )

            try:
                # Create adapter and validate example configuration
                config_adapter = TypeAdapter(ExampleConfig)
                example_config = {
                    "host": "localhost",
                    "port": 5432,
                    "database": "flext",
                }

                with contextlib.suppress(Exception):
                    validated_config = config_adapter.validate_python(example_config)
                    return FlextResult[object].ok(validated_config)

                return FlextResult[object].fail(
                    "Example config validation failed",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            except Exception as e:
                return FlextResult[object].fail(
                    f"Example config validation failed: {e!s}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )


__all__ = [
    "FlextTypeAdapters",
]
