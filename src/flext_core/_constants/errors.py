"""FlextConstantsErrors - error domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, override


class FlextConstantsErrors:
    """Error domain constants for structured error routing."""

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
    ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED: Final[str] = (
        "Handler must expose message_type, event_type, or can_handle"
    )
    ERR_DISPATCHER_NOT_CONFIGURED: Final[str] = "Dispatcher not configured"
    ERR_UNEXPECTED_MESSAGE_TYPE: Final[str] = "Unexpected message type"
    ERR_RESULT_NOT_SCALAR_COMPATIBLE: Final[str] = (
        "Result must be compatible with Scalar"
    )
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
    ERR_VALIDATION_FAILED: Final[str] = "Validation failed"
    ERR_GENERATOR_KIND_MISSING: Final[str] = "No kind provided for prefix resolution"
    ERR_DOMAIN_EVENT_NAME_REQUIRED: Final[str] = (
        "Domain event name must be a non-empty string"
    )
    ERR_EVENTS_LIST_OR_TUPLE_REQUIRED: Final[str] = "Events must be a list or tuple"
    ERR_EVENT_NAME_REQUIRED: Final[str] = "Event name must be non-empty string"
    ERR_CHECKER_HANDLER_NO_HANDLE_METHOD: Final[str] = "Handler has no handle method"
    ERR_CHECKER_HANDLER_HANDLE_NOT_CALLABLE: Final[str] = (
        "Handler handle attribute is not callable"
    )
    ERR_CHECKER_NO_MESSAGE_PARAMETER: Final[str] = (
        "No message parameter found in handle"
    )
    ERR_CHECKER_TYPE_HINT_NONE: Final[str] = "Type hint is None"
    ERR_CHECKER_NO_ANNOTATION_OR_TYPE_HINT: Final[str] = (
        "No annotation or type hint for parameter"
    )
    ERR_CHECKER_INVALID_HANDLE_METHOD_SIGNATURE: Final[str] = (
        "Invalid handle method signature"
    )
    ERR_COLLECTION_NO_MATCHING_ITEM_FOUND: Final[str] = "No matching item found"
    ERR_MAPPER_NOT_A_SEQUENCE: Final[str] = "Not a sequence"
    ERR_MAPPER_FOUND_NONE_INDEX: Final[str] = "found_none:index"
    ERR_INFRA_INVALID_HANDLER_MODE: Final[str] = (
        "handler_mode must be 'command' or 'query'"
    )
    ERR_INFRA_HANDLER_REQUIRED: Final[str] = "handler cannot be None"
    ERR_INFRA_MESSAGE_REQUIRED: Final[str] = "message cannot be None"
    ERR_INFRA_TIMEOUT_POSITIVE: Final[str] = "timeout must be positive"
    ERR_INFRA_INVALID_REGISTRATION_ID: Final[str] = (
        "registration_id must be non-empty string"
    )
    ERR_INFRA_INVALID_REQUEST_ID: Final[str] = "request_id must be non-empty string"
    ERR_MIXINS_EMPTY_OPERATION: Final[str] = "Operation name cannot be empty"
    ERR_MIXINS_EMPTY_STATE: Final[str] = "State value cannot be empty"
    ERR_MIXINS_EMPTY_FIELD_NAME: Final[str] = "Field name cannot be empty"
    ERR_MIXINS_INVALID_ENCODING: Final[str] = "Invalid character encoding"
    ERR_MIXINS_MISSING_TIMESTAMP: Final[str] = "Required timestamp fields missing"
    ERR_MIXINS_INVALID_LOG_LEVEL: Final[str] = "Invalid log level"
    ERR_SERVICE_REGISTRATION_FAILED: Final[str] = "Service registration failed"
