"""Message and routing constants for FlextConstantsErrors."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, override


class FlextConstantsErrorsMessages:
    """Template, domain, handler, and context error constants."""

    ERR_TEMPLATE_FAILED_WITH_ERROR: Final[str] = "Failed to {operation}: {error}"
    ERR_TEMPLATE_KEY_NOT_FOUND: Final[str] = "Key '{key}' not found"
    ERR_TEMPLATE_KEY_NOT_FOUND_AT_PATH: Final[str] = "Key '{key}' not found at '{path}'"
    ERR_TEMPLATE_FOUND_NONE: Final[str] = "found_none:{key}"
    ERR_TEMPLATE_INDEX_OUT_OF_RANGE: Final[str] = "Index {index} out of range"
    ERR_TEMPLATE_INVALID_INDEX: Final[str] = "Invalid index {index}"
    ERR_TEMPLATE_MISSING_VALUE: Final[str] = (
        "Template value '{key}' is required for template '{template}'"
    )
    ERR_TEMPLATE_VALIDATION_FAILED_FOR_FIELD: Final[str] = (
        "Validation failed for {field}"
    )
    ERR_TEMPLATE_MESSAGE_AND_DEFAULT_IS_NONE: Final[str] = (
        "{message} and default is None"
    )
    ERR_TEMPLATE_ARRAY_ERROR_AT_KEY: Final[str] = "Array error at '{key}': {error}"
    ERR_TEMPLATE_PATH_IS_NONE: Final[str] = "Path '{path}' is None"
    ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE: Final[str] = "Extracted value is None"
    ERR_TEMPLATE_EXTRACT_FAILED: Final[str] = "Extract failed: {error}"
    ERR_TEMPLATE_FAILED_TO_MAP_DICT_KEYS: Final[str] = (
        "Failed to map dict keys: {error}"
    )
    ERR_TEMPLATE_TRANSFORM_FAILED: Final[str] = "Transform failed: {error}"

    @unique
    class ErrorDomain(StrEnum):
        """Standard error domain categories for structured error routing.

        Enables consistent error handling across FLEXT projects by categorizing
        errors into domains. Each domain has standard error codes that can be
        routed to specific handlers.
        """

        #: Validation errors (input validation, schema validation, constraints)
        VALIDATION = "VALIDATION"

        #: Network errors (connection, timeout, DNS, protocol)
        NETWORK = "NETWORK"

        #: Authentication/Authorization errors (invalid credentials, access denied)
        AUTH = "AUTH"

        #: Resource not found errors (missing user, missing file, missing record)
        NOT_FOUND = "NOT_FOUND"

        #: Operation timeout errors (request timeout, operation timeout)
        TIMEOUT = "TIMEOUT"

        #: Internal errors (unexpected state, invariant violation, internal bug)
        INTERNAL = "INTERNAL"

        #: Unknown error category (when error doesn't fit other domains)
        UNKNOWN = "UNKNOWN"

        @override
        def __str__(self) -> str:
            """Return the domain value (not the enum name)."""
            return self.value

    ERR_HANDLER_MUST_BE_CALLABLE: Final[str] = "Handler must be callable"
    ERR_HANDLER_FAILED: Final[str] = "Handler failed"
    ERR_HANDLER_RETURNED_NON_CONTAINER_SUCCESS_RESULT: Final[str] = (
        "Handler returned non-container value in success result"
    )
    ERR_HANDLER_RETURNED_NONE: Final[str] = "Handler returned None"
    ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE: Final[str] = (
        "Handler returned non-container value"
    )
    ERR_HANDLER_EXECUTION_FAILED: Final[str] = "Handler execution failed: {error}"
    ERR_HANDLER_INVALID_MODE: Final[str] = "Invalid handler mode: {mode}"
    ERR_HANDLER_MISSING_HANDLE_IMPLEMENTATION: Final[str] = (
        "{qualname} must implement a handle() method"
    )
    ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE: Final[str] = (
        "Handler with mode '{handler_mode}' cannot execute {operation} pipelines"
    )
    ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE: Final[str] = (
        "Handler cannot handle message type {type_name}"
    )
    ERR_HANDLER_MESSAGE_VALIDATION_FAILED: Final[str] = (
        "Message validation failed: {error}"
    )
    ERR_HANDLER_CRITICAL_FAILURE: Final[str] = "Critical handler failure: {error}"
    ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED: Final[str] = (
        "Handler must expose message_type, event_type, or can_handle"
    )
    ERR_DISPATCHER_NOT_CONFIGURED: Final[str] = "Dispatcher not configured"
    ERR_REGISTRY_CATEGORY_NAME_CANNOT_BE_EMPTY: Final[str] = (
        "{category} name cannot be empty"
    )
    ERR_REGISTRY_VALIDATION_ERROR: Final[str] = "Validation error: {error}"
    ERR_REGISTRY_PLUGIN_NOT_REGISTERED: Final[str] = (
        "{category} '{name}' not registered"
    )
    ERR_UNEXPECTED_MESSAGE_TYPE: Final[str] = "Unexpected message type"
    ERR_SERVICE_TYPE_MISMATCH: Final[str] = "Service is not of type {type_name}"
    ERR_SERVICE_NOT_FOUND: Final[str] = "{resource_type} '{name}' not found"
    ERR_RESOURCE_UNSUPPORTED_RUNTIME_TYPE: Final[str] = (
        "Resource '{name}' returned unsupported runtime type"
    )
    ERR_RESULT_NOT_SCALAR_COMPATIBLE: Final[str] = (
        "Result must be compatible with Scalar"
    )
    ERR_RESULT_FILTER_PREDICATE_FAILED: Final[str] = (
        "Value did not pass filter predicate"
    )
    ERR_RESULT_CANNOT_ACCESS_VALUE: Final[str] = (
        "Cannot access value of failed result: {error}"
    )
    ERR_RESULT_CANNOT_UNWRAP: Final[str] = "Cannot unwrap failed result: {error}"
    ERR_MESSAGE_CANNOT_BE_NONE: Final[str] = "Message cannot be None"
    ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED: Final[str] = (
        "Key must be a non-empty string"
    )
    ERR_CONTEXT_VALUE_CANNOT_BE_NONE: Final[str] = "Value cannot be None"
    ERR_CONTEXT_VALUE_NOT_SERIALIZABLE: Final[str] = "Value must be serializable"
    ERR_CONTEXT_NOT_ACTIVE: Final[str] = "Context is not active"
    ERR_CONTEXT_INVALID_KEY_FOUND: Final[str] = "Invalid key found in context"
    ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED: Final[str] = (
        "Value is required for single-key set"
    )
    ERR_CONTEXT_METADATA_KEY_NOT_FOUND: Final[str] = "Metadata key '{key}' not found"
    ERR_VALIDATION_FAILED: Final[str] = "Validation failed"
    ERR_VALIDATION_FAILED_WITH_ERROR: Final[str] = "Validation failed: {error}"
    ERR_GENERATOR_KIND_MISSING: Final[str] = "No kind provided for prefix resolution"


__all__ = ["FlextConstantsErrorsMessages"]
