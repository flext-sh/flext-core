"""Layer 2: Exception hierarchy aligned with the FLEXT 1.0.0 modernization charter.

This module provides structured exception classes with error codes and correlation
tracking for the entire FLEXT ecosystem. Use FlextException hierarchy for all
error handling throughout FLEXT applications.

Dependency Layer: 2 (Early Foundation)
Dependencies: FlextConstants, FlextTypes
Used by: All FlextCore modules requiring structured error handling

Error codes, correlation tracking, and structured payloads match the guidance
in ``README.md`` and ``docs/architecture.md`` so diagnostics remain uniform
across packages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import ClassVar, cast

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes


class FlextExceptions:
    """Hierarchical exception system with modernization-ready diagnostics.

    Factory helpers create structured errors and record metrics so dispatcher
    flows, domain services, and configuration loaders surface consistent
    failures throughout the 1.0.0 rollout.
    """

    def __call__(
        self,
        message: str,
        *,
        operation: str | None = None,  # Operation context for error tracking
        field: str | None = None,  # Field name for validation errors
        config_key: str | None = None,  # Configuration key for config errors
        error_code: str | None = None,  # Error code for categorization
        **kwargs: object,
    ) -> FlextExceptions.BaseError:
        """Allow FlextExceptions() to be called directly."""
        # Custom __call__ method to make exceptions callable
        return self.create(
            message,
            operation=operation,
            field=field,
            config_key=config_key,
            error_code=error_code,
            **kwargs,
        )

    # =============================================================================
    # Metrics Domain: Exception metrics and monitoring functionality
    # =============================================================================

    _metrics: ClassVar[FlextTypes.Core.CounterDict] = {}

    # Type conversion mapping for cleaner type resolution
    _TYPE_MAP: ClassVar[dict[str, type]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    @classmethod
    def record_exception(cls, exception_type: str) -> None:
        """Record exception occurrence."""
        cls._metrics[exception_type] = cls._metrics.get(exception_type, 0) + 1

    @classmethod
    def get_metrics(cls) -> FlextTypes.Core.CounterDict:
        """Get exception counts."""
        return dict(cls._metrics)

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._metrics.clear()

    # =============================================================================
    # BASE EXCEPTION CLASS - Clean hierarchical approach
    # =============================================================================

    class BaseError(Exception):
        """Base exception with structured error handling."""

        def __init__(
            self,
            message: str,
            *,
            code: str | None = None,
            context: Mapping[str, object] | None = None,
            correlation_id: str | None = None,
        ) -> None:
            """Initialize structured exception."""
            super().__init__(message)
            self.message = message
            self.code = code or FlextConstants.Errors.GENERIC_ERROR
            self.context = dict(context or {})
            self.correlation_id = correlation_id or f"flext_{int(time.time() * 1000)}"
            self.timestamp = time.time()
            FlextExceptions.record_exception(self.__class__.__name__)

        def __str__(self) -> str:
            """Return string representation with error code and message."""
            return f"[{self.code}] {self.message}"

        @property
        def error_code(self) -> str:
            """Get error code as string."""
            return str(self.code)

        @staticmethod
        def _build_context(
            base_context: Mapping[str, object] | None,
            **additional_context: object,
        ) -> dict[str, object]:
            """Build context dictionary from base context and additional items."""
            context_dict = dict(base_context) if isinstance(base_context, dict) else {}
            context_dict.update(additional_context)
            return context_dict

        @staticmethod
        def _extract_common_kwargs(
            kwargs: dict[str, object],
        ) -> tuple[
            dict[str, object] | None,
            str | None,
            str | None,
        ]:
            """Extract common kwargs (context, correlation_id, error_code) from kwargs."""
            context_raw = kwargs.get("context")
            context = (
                cast("dict[str, object]", context_raw)
                if isinstance(context_raw, dict)
                else None
            )

            correlation_id_raw = kwargs.get("correlation_id")
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )

            error_code_raw = kwargs.get("error_code")
            error_code = str(error_code_raw) if error_code_raw is not None else None

            return context, correlation_id, error_code

        @staticmethod
        def _convert_type_name(type_name: str | None) -> type | str:
            """Convert type name string to actual type object."""
            if not type_name:
                return ""
            return FlextExceptions._TYPE_MAP.get(type_name, type_name)

    @staticmethod
    def _extract_common_kwargs(
        kwargs: dict[str, object],
    ) -> tuple[
        dict[str, object] | None,
        str | None,
        str | None,
    ]:
        """Extract common kwargs (context, correlation_id, error_code) from kwargs."""
        context_raw = kwargs.get("context")
        context = (
            cast("dict[str, object]", context_raw)
            if isinstance(context_raw, dict)
            else None
        )

        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id = (
            str(correlation_id_raw) if correlation_id_raw is not None else None
        )

        error_code_raw = kwargs.get("error_code")
        error_code = str(error_code_raw) if error_code_raw is not None else None

        return context, correlation_id, error_code

    # =============================================================================
    # SPECIFIC EXCEPTION CLASSES - Clean subclass hierarchy
    # =============================================================================

    class _AttributeError(BaseError, AttributeError):
        """Attribute access failure with attribute context."""

        def __init__(
            self,
            message: str,
            *,
            attribute_name: str | None = None,
            attribute_context: Mapping[str, object] | None = None,
            **kwargs: object,
        ) -> None:
            self.attribute_name = attribute_name

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context_data: dict[str, object] = {"attribute_name": attribute_name}
            if attribute_context:
                context_data["attribute_context"] = dict(attribute_context)

            context = self._build_context(base_context, **context_data)

            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _OperationError(BaseError, RuntimeError):
        """Generic operation failure."""

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.operation = operation

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(base_context, operation=operation)

            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _ValidationError(BaseError, ValueError):
        """Data validation failure."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: object = None,
            validation_details: object = None,
            **kwargs: object,
        ) -> None:
            self.field = field
            self.value = value
            self.validation_details = validation_details

            # Extract common parameters using helper
            base_context, correlation_id, error_code = self._extract_common_kwargs(
                kwargs
            )

            # Build context with specific fields
            context = self._build_context(
                base_context,
                field=field,
                value=value,
                validation_details=validation_details,
            )

            super().__init__(
                message,
                code=error_code or FlextConstants.Errors.VALIDATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _ConfigurationError(BaseError, ValueError):
        """System configuration error."""

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_file: str | None = None,
            **kwargs: object,
        ) -> None:
            self.config_key = config_key
            self.config_file = config_file
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({"config_key": config_key, "config_file": config_file})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONFIGURATION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _ConnectionError(BaseError, ConnectionError):
        """Network or service connection failure."""

        def __init__(
            self,
            message: str,
            *,
            service: str | None = None,
            endpoint: str | None = None,
            **kwargs: object,
        ) -> None:
            self.service = service
            self.endpoint = endpoint
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({"service": service, "endpoint": endpoint})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONNECTION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _ProcessingError(BaseError, RuntimeError):
        """Business logic or data processing failure."""

        def __init__(
            self,
            message: str,
            *,
            business_rule: str | None = None,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.business_rule = business_rule
            self.operation = operation
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({
                "business_rule": business_rule,
                "operation": operation,
            })
            super().__init__(
                message,
                code=FlextConstants.Errors.PROCESSING_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _TimeoutError(BaseError, TimeoutError):
        """Operation timeout with timing context."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            **kwargs: object,
        ) -> None:
            self.timeout_seconds = timeout_seconds
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict["timeout_seconds"] = timeout_seconds
            super().__init__(
                message,
                code=FlextConstants.Errors.TIMEOUT_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _NotFoundError(BaseError, FileNotFoundError):
        """Resource not found."""

        def __init__(
            self,
            message: str,
            *,
            resource_id: str | None = None,
            resource_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.resource_id = resource_id
            self.resource_type = resource_type
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({
                "resource_id": resource_id,
                "resource_type": resource_type,
            })
            super().__init__(
                message,
                code=FlextConstants.Errors.NOT_FOUND,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _AlreadyExistsError(BaseError, FileExistsError):
        """Resource already exists."""

        def __init__(
            self,
            message: str,
            *,
            resource_id: str | None = None,
            resource_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.resource_id = resource_id
            self.resource_type = resource_type
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict["resource_id"] = resource_id
            context_dict["resource_type"] = resource_type
            super().__init__(
                message,
                code=FlextConstants.Errors.ALREADY_EXISTS,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _PermissionError(BaseError, PermissionError):
        """Insufficient permissions."""

        def __init__(
            self,
            message: str,
            *,
            required_permission: str | None = None,
            **kwargs: object,
        ) -> None:
            self.required_permission = required_permission
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict["required_permission"] = required_permission
            super().__init__(
                message,
                code=FlextConstants.Errors.PERMISSION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _AuthenticationError(BaseError, PermissionError):
        """Authentication failure."""

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            **kwargs: object,
        ) -> None:
            self.auth_method = auth_method
            context_raw = kwargs.get("context", {})
            context_dict: dict[str, object]
            if isinstance(context_raw, dict):
                context_dict = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict["auth_method"] = auth_method
            super().__init__(
                message,
                code=FlextConstants.Errors.AUTHENTICATION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _TypeError(BaseError, TypeError):
        """Type validation failure."""

        def __init__(
            self,
            message: str,
            *,
            expected_type: str | None = None,
            actual_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.expected_type = expected_type
            self.actual_type = actual_type

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Convert type names to actual types using helper
            expected_type_obj = self._convert_type_name(expected_type)
            actual_type_obj = self._convert_type_name(actual_type)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                expected_type=expected_type_obj,
                actual_type=actual_type_obj,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _CriticalError(BaseError, SystemError):
        """Critical system error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            # Extract special parameters
            context_raw = kwargs.pop("context", None)
            if isinstance(context_raw, dict):
                context_dict: dict[str, object] | None = cast(
                    "dict[str, object]", context_raw
                )
            else:
                context_dict = None
            correlation_id_raw = kwargs.pop("correlation_id", None)
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )

            # Add remaining kwargs to context for full functionality
            if context_dict is not None:
                full_context: dict[str, object] = dict(context_dict)
                full_context.update(kwargs)
                context_dict = full_context
            elif kwargs:
                context_dict = dict(kwargs)

            super().__init__(
                message,
                code=FlextConstants.Errors.CRITICAL_ERROR,
                context=context_dict,
                correlation_id=str(correlation_id)
                if correlation_id is not None
                else None,
            )

    class _Error(BaseError, RuntimeError):
        """Generic FLEXT error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            # Extract special parameters
            context_raw = kwargs.pop("context", None)
            if isinstance(context_raw, dict):
                context_dict: dict[str, object] | None = cast(
                    "dict[str, object]", context_raw
                )
            else:
                context_dict = None
            correlation_id_raw = kwargs.pop("correlation_id", None)
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )
            error_code_raw = kwargs.pop("error_code", None)
            error_code = str(error_code_raw) if error_code_raw is not None else None

            # Add remaining kwargs to context for full functionality
            if context_dict is not None:
                full_context: dict[str, object] = dict(context_dict)
                full_context.update(kwargs)
                context_dict = full_context
            elif kwargs:
                context_dict = dict(kwargs)

            super().__init__(
                message,
                code=error_code or FlextConstants.Errors.GENERIC_ERROR,
                context=context_dict,
                correlation_id=correlation_id,
            )

    class _UserError(BaseError, TypeError):
        """User input or API usage error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            context_raw = kwargs.get("context", {})
            if isinstance(context_raw, dict):
                context_dict: dict[str, object] | None = cast(
                    "dict[str, object]", context_raw
                )
            else:
                context_dict = None
            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    # =============================================================================
    # PUBLIC API ALIASES - Real exception classes with clean names
    # =============================================================================

    AttributeError = _AttributeError
    OperationError = _OperationError
    ValidationError = _ValidationError
    ConfigurationError = _ConfigurationError
    ConnectionError = _ConnectionError
    ProcessingError = _ProcessingError
    TimeoutError = _TimeoutError
    NotFoundError = _NotFoundError
    AlreadyExistsError = _AlreadyExistsError
    PermissionError = _PermissionError
    AuthenticationError = _AuthenticationError
    TypeError = _TypeError
    CriticalError = _CriticalError
    Error = _Error
    UserError = _UserError

    # =============================================================================
    # DIRECT CALLABLE INTERFACE - For general usage
    # =============================================================================

    @classmethod
    def create(
        cls,
        message: str,
        *,
        operation: str | None = None,
        field: str | None = None,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> BaseError:
        """Create exception with automatic type selection."""
        # Extract common parameters using helper
        base_context, correlation_id, extracted_error_code = cls._extract_common_kwargs(
            kwargs
        )
        final_error_code = error_code or extracted_error_code

        # Exception type mapping for cleaner selection
        if operation is not None:
            return cls._OperationError(
                message,
                operation=operation,
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        if field is not None:
            return cls._ValidationError(
                message,
                field=field,
                value=kwargs.get("value"),
                validation_details=kwargs.get("validation_details"),
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        if config_key is not None:
            config_file = (
                str(kwargs.get("config_file"))
                if isinstance(kwargs.get("config_file"), str)
                else None
            )
            return cls._ConfigurationError(
                message,
                config_key=config_key,
                config_file=config_file,
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        # Default to general error
        return cls._Error(
            message,
            context=base_context,
            correlation_id=correlation_id,
            error_code=final_error_code,
        )

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    @staticmethod
    def create_module_exception_classes(
        module_name: str,
    ) -> dict[str, type[FlextExceptions.BaseError]]:
        """Create module-specific exception classes.

        Creates a dictionary of exception classes tailored for a specific module,
        following the FLEXT ecosystem naming conventions.

        Args:
            module_name: Name of the module (e.g., "flext_grpc")

        Returns:
            Dictionary mapping exception names to exception classes

        """
        # Normalize module name for class naming
        normalized_name = module_name.upper().replace("-", "_").replace(".", "_")

        # Create base exception class for the module
        class ModuleBaseError(FlextExceptions.BaseError):
            """Base exception for module-specific errors."""

        # Create configuration error class
        class ModuleConfigurationError(ModuleBaseError):
            """Configuration-related errors for the module."""

        # Create connection error class
        class ModuleConnectionError(ModuleBaseError):
            """Connection-related errors for the module."""

        # Create validation error class
        class ModuleValidationError(ModuleBaseError):
            """Validation-related errors for the module."""

        # Create authentication error class
        class ModuleAuthenticationError(ModuleBaseError):
            """Authentication-related errors for the module."""

        # Create processing error class
        class ModuleProcessingError(ModuleBaseError):
            """Processing-related errors for the module."""

        # Create timeout error class
        class ModuleTimeoutError(ModuleBaseError):
            """Timeout-related errors for the module."""

        # Return dictionary with module-specific naming
        return {
            f"{normalized_name}BaseError": ModuleBaseError,
            f"{normalized_name}Error": ModuleBaseError,  # General error alias
            f"{normalized_name}ConfigurationError": ModuleConfigurationError,
            f"{normalized_name}ConnectionError": ModuleConnectionError,
            f"{normalized_name}ValidationError": ModuleValidationError,
            f"{normalized_name}AuthenticationError": ModuleAuthenticationError,
            f"{normalized_name}ProcessingError": ModuleProcessingError,
            f"{normalized_name}TimeoutError": ModuleTimeoutError,
        }


__all__: FlextTypes.Core.StringList = [
    "FlextExceptions",
]
