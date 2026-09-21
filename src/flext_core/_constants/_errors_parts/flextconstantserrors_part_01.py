"""Template, domain, handler, and context error constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, override


class FlextConstantsErrorsMessages:
    """Template, domain, handler, and context error constants."""

    ERR_TEMPLATE_FAILED_WITH_ERROR: ClassVar[str] = "Failed to {operation}: {error}"
    ERR_TEMPLATE_KEY_NOT_FOUND: ClassVar[str] = "Key '{key}' not found"
    ERR_TEMPLATE_INDEX_OUT_OF_RANGE: ClassVar[str] = "Index {index} out of range"
    ERR_TEMPLATE_INVALID_INDEX: ClassVar[str] = "Invalid index {index}"
    ERR_TEMPLATE_MISSING_VALUE: ClassVar[str] = (
        "Template value '{key}' is required for template '{template}'"
    )
    ERR_TEMPLATE_VALIDATION_FAILED_FOR_FIELD: ClassVar[str] = (
        "Validation failed for {field}"
    )
    ERR_TEMPLATE_PATH_IS_NONE: ClassVar[str] = "Path '{path}' is None"
    ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE: ClassVar[str] = "Extracted value is None"
    ERR_TEMPLATE_EXTRACT_FAILED: ClassVar[str] = "Extract failed: {error}"
    ERR_TEMPLATE_FAILED_TO_MAP_DICT_KEYS: ClassVar[str] = (
        "Failed to map dict keys: {error}"
    )
    ERR_TEMPLATE_TRANSFORM_FAILED: ClassVar[str] = "Transform failed: {error}"

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

    ERR_HANDLER_MUST_BE_CALLABLE: ClassVar[str] = "Handler must be callable"
    ERR_HANDLER_FAILED: ClassVar[str] = "Handler failed"
    ERR_HANDLER_RETURNED_NON_CONTAINER_SUCCESS_RESULT: ClassVar[str] = (
        "Handler returned non-container value in success result"
    )
    ERR_HANDLER_RETURNED_NONE: ClassVar[str] = "Handler returned None"
    ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE: ClassVar[str] = (
        "Handler returned non-container value"
    )
    ERR_HANDLER_EXECUTION_FAILED: ClassVar[str] = "Handler execution failed: {error}"
    ERR_HANDLER_INVALID_MODE: ClassVar[str] = "Invalid handler mode: {mode}"
    ERR_HANDLER_MISSING_HANDLE_IMPLEMENTATION: ClassVar[str] = (
        "{qualname} must implement a handle() method"
    )
    ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE: ClassVar[str] = (
        "Handler with mode '{handler_mode}' cannot execute {operation} pipelines"
    )
    ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE: ClassVar[str] = (
        "Handler cannot handle message type {type_name}"
    )
    ERR_HANDLER_MESSAGE_VALIDATION_FAILED: ClassVar[str] = (
        "Message validation failed: {error}"
    )
    ERR_HANDLER_CRITICAL_FAILURE: ClassVar[str] = "Critical handler failure: {error}"
    ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED: ClassVar[str] = (
        "Handler must expose message_type, event_type, or can_handle"
    )
    ERR_DISPATCHER_NOT_CONFIGURED: ClassVar[str] = "Dispatcher not configured"
    ERR_REGISTRY_CATEGORY_NAME_CANNOT_BE_EMPTY: ClassVar[str] = (
        "{category} name cannot be empty"
    )
    ERR_REGISTRY_VALIDATION_ERROR: ClassVar[str] = "Validation error: {error}"
    ERR_REGISTRY_PLUGIN_NOT_REGISTERED: ClassVar[str] = (
        "{category} '{name}' not registered"
    )
    ERR_UNEXPECTED_MESSAGE_TYPE: ClassVar[str] = "Unexpected message type"
    ERR_SERVICE_TYPE_MISMATCH: ClassVar[str] = "Service is not of type {type_name}"
    ERR_SERVICE_NOT_FOUND: ClassVar[str] = "{resource_type} '{name}' not found"
    ERR_RESOURCE_UNSUPPORTED_RUNTIME_TYPE: ClassVar[str] = (
        "Resource '{name}' returned unsupported runtime type"
    )
    ERR_RESULT_NOT_SCALAR_COMPATIBLE: ClassVar[str] = (
        "Result must be compatible with Scalar"
    )
    ERR_RESULT_FILTER_PREDICATE_FAILED: ClassVar[str] = (
        "Value did not pass filter predicate"
    )
    ERR_RESULT_FAILURE_REQUIRED: ClassVar[str] = (
        "Cannot propagate a successful result as a failure"
    )
    ERR_RESULT_CANNOT_ACCESS_VALUE: ClassVar[str] = (
        "Cannot access value of failed result: {error}"
    )
    ERR_RESULT_CANNOT_UNWRAP: ClassVar[str] = "Cannot unwrap failed result: {error}"
    ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_NONE: ClassVar[str] = (
        "Success result payload cannot be None"
    )
    ERR_RESULT_TYPE_PARAM_NONE_FORBIDDEN: ClassVar[str] = (
        "FlextResult cannot be parameterized with None; use r[bool].ok(True) "
        "or a concrete payload type"
    )
    ERR_RESULT_TYPE_PARAM_OBJECT_FORBIDDEN: ClassVar[str] = (
        "FlextResult cannot be parameterized with object; use a concrete payload type"
    )
    ERR_RESULT_SUCCESS_PAYLOAD_CANNOT_BE_OBJECT: ClassVar[str] = (
        "Success result payload cannot be a bare object instance"
    )
    ERR_MESSAGE_CANNOT_BE_NONE: ClassVar[str] = "Message cannot be None"
    ERR_CONTEXT_KEY_NON_EMPTY_STRING_REQUIRED: ClassVar[str] = (
        "Key must be a non-empty string"
    )
    ERR_CONTEXT_VALUE_CANNOT_BE_NONE: ClassVar[str] = "Value cannot be None"
    ERR_CONTEXT_VALUE_NOT_SERIALIZABLE: ClassVar[str] = "Value must be serializable"
    ERR_CONTEXT_NOT_ACTIVE: ClassVar[str] = "Context is not active"
    ERR_CONTEXT_INVALID_KEY_FOUND: ClassVar[str] = "Invalid key found in context"
    ERR_CONTEXT_SINGLE_KEY_VALUE_REQUIRED: ClassVar[str] = (
        "Value is required for single-key set"
    )
    ERR_CONTEXT_METADATA_KEY_NOT_FOUND: ClassVar[str] = "Metadata key '{key}' not found"
    ERR_VALIDATION_FAILED: ClassVar[str] = "Validation failed"
    ERR_VALIDATION_FAILED_WITH_ERROR: ClassVar[str] = "Validation failed: {error}"
    ERR_GENERATOR_KIND_MISSING: ClassVar[str] = "No kind provided for prefix resolution"
