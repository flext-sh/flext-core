"""FlextConstantsErrors - error domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from enum import StrEnum, unique
from typing import ClassVar, Final, override

from pydantic import ValidationError as _PydanticValidationError


class FlextConstantsErrors:
    """Error domain constants for structured error routing."""

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
            """Return the domain value (not the enum name).

            Returns:
                The domain value (not the enum name).

            """
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
    ERR_REGISTRY_PLUGIN_KEY_INVALID: ClassVar[str] = (
        "Plugin category and name must be nonempty and cannot contain '{separator}'"
    )
    ERR_HANDLER_METADATA_INVALID: ClassVar[str] = "Invalid handler metadata for {name}"
    ERR_HANDLER_POLICY_UNSUPPORTED: ClassVar[str] = (
        "Handler timeout and middleware are unsupported"
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
        "FlextResult cannot be parameterized with None; use r[bool].ok(value=True) "
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

    _ARITH: ClassVar[tuple[type[Exception], ...]] = (ArithmeticError,)
    _ATTR: ClassVar[tuple[type[Exception], ...]] = (AttributeError,)
    _CONN: ClassVar[tuple[type[Exception], ...]] = (ConnectionError,)
    _FS: ClassVar[tuple[type[Exception], ...]] = (FileNotFoundError,)
    _KEY: ClassVar[tuple[type[Exception], ...]] = (KeyError,)
    _OS: ClassVar[tuple[type[Exception], ...]] = (OSError,)
    _RUNTIME: ClassVar[tuple[type[Exception], ...]] = (RuntimeError,)
    _TYPE: ClassVar[tuple[type[Exception], ...]] = (TypeError,)
    _UNICODE_DECODE: ClassVar[tuple[type[Exception], ...]] = (UnicodeDecodeError,)
    _VALUE: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    EXC_CANCELLATION: ClassVar[tuple[type[Exception], ...]] = (CancelledError,)
    """Ordinary-exception cancellation signals that must propagate unchanged."""

    EXC_TYPE_VALIDATION: ClassVar[tuple[type[Exception], ...]] = _TYPE + _VALUE
    """Minimal type-validation catch for value-coercion boundaries."""

    EXC_ATTR_TYPE: ClassVar[tuple[type[Exception], ...]] = _ATTR + _TYPE
    """Minimal attribute-access + type catch for object-shape boundaries."""

    EXC_OS_TYPE: ClassVar[tuple[type[Exception], ...]] = _OS + _TYPE
    """Filesystem + type-validation catch for path-handler boundaries."""

    EXC_OS_VALUE: ClassVar[tuple[type[Exception], ...]] = _OS + _VALUE
    """Filesystem + value-validation catch for path/IO boundaries."""

    EXC_OS_DECODING: ClassVar[tuple[type[Exception], ...]] = _OS + _UNICODE_DECODE
    """Filesystem read + unicode decoding catch for text-file boundaries."""

    EXC_OS_RUNTIME_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        _OS + _RUNTIME + _VALUE
    )
    """Filesystem + runtime + value-validation catch for IO-bound flows."""

    EXC_FS_DECODING: ClassVar[tuple[type[Exception], ...]] = (
        _FS + _OS + (PermissionError,) + _UNICODE_DECODE
    )
    """Filesystem read + decoding catch for file-handler boundaries."""

    EXC_NETWORK_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        *_CONN,
        TimeoutError,
        *_VALUE,
    )
    """Network connectivity + type-validation catch for HTTP/RPC boundaries."""

    EXC_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Runtime + type-validation catch for basic adapter boundaries."""

    EXC_BASIC_TYPE: ClassVar[tuple[type[Exception], ...]] = _ATTR + EXC_TYPE_VALIDATION
    """Attribute + type-validation catch for object-shape adapter boundaries."""

    EXC_MAPPING_TYPE: ClassVar[tuple[type[Exception], ...]] = _KEY + EXC_TYPE_VALIDATION
    """Mapping access + type-validation catch for dict-shape boundaries."""

    EXC_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = _OS + EXC_TYPE_VALIDATION
    """Filesystem + typing catch for path/IO + value-validation boundaries."""

    EXC_ATTR_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _ATTR + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Attribute access + runtime + typing catch for object-state boundaries."""

    EXC_OS_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Filesystem + runtime + typing catch for IO-bound boundary code."""

    EXC_BROAD_RUNTIME: ClassVar[tuple[type[Exception], ...]] = (
        _ARITH + _ATTR + _KEY + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Broad runtime catch for adapter-internal flows (no IO, no import)."""

    EXC_BROAD_RUNTIME_OS: ClassVar[tuple[type[Exception], ...]] = (
        _ATTR + _KEY + _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Broad runtime + filesystem boundary catch (no ImportError)."""

    EXC_BROAD_IO_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        *_ATTR,
        ImportError,
        *_KEY,
        *_OS,
        *_RUNTIME,
        *EXC_TYPE_VALIDATION,
    )
    """Broad-spectrum boundary catch: filesystem, import, type, runtime errors."""

    CATCHABLE_RUNTIME_EXCEPTIONS: ClassVar[tuple[type[Exception], ...]] = (
        _ARITH + _ATTR + _KEY + (LookupError,) + _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )

    EXC_HTTP_PROCESSING: ClassVar[tuple[type[Exception], ...]] = (
        _CONN + EXC_MAPPING_TYPE
    )
    """HTTP-shape boundary: connection + parsing + typing for request handlers."""

    _PYDANTIC: ClassVar[tuple[type[Exception], ...]] = (_PydanticValidationError,)

    EXC_VALIDATION_TYPE: ClassVar[tuple[type[Exception], ...]] = (TypeError, *_PYDANTIC)
    """Pydantic validation + type-validation catch for model-coercion boundaries."""

    EXC_VALIDATION_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        ValueError,
        *_PYDANTIC,
    )
    """Pydantic validation + value-validation catch for input-coercion boundaries."""

    EXC_VALIDATION_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        _PYDANTIC + EXC_TYPE_VALIDATION
    )
    """Pydantic validation + typing + value catch for full validation boundaries."""

    EXC_ATTR_KEY_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        AttributeError,
        *EXC_MAPPING_TYPE,
    )
    """Attribute + mapping + typing catch for object-state + dict boundaries."""

    EXC_FS_KEY_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        KeyError,
        *EXC_OS_VALUE,
    )
    """Filesystem read + mapping + value-validation catch for config-file flows."""

    EXC_FS_FULL_DECODE: ClassVar[tuple[type[Exception], ...]] = (
        *EXC_FS_DECODING,
        ValueError,
    )
    """Filesystem + permissions + decoding + value catch for full-text-file flows."""

    EXC_KEY_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        KeyError,
        *EXC_OS_TYPE_VALUE,
    )
    """Mapping + filesystem + typing catch for IO-bound config flows."""

    EXC_OS_SYNTAX: ClassVar[tuple[type[Exception], ...]] = (OSError, SyntaxError)
    """Filesystem + syntax catch for source-parsing boundaries."""

    EXC_ATTR_RUNTIME_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        EXC_ATTR_RUNTIME_TYPE + _PYDANTIC
    )
    """Pydantic validation + runtime + attr catch for full model boundaries."""

    EXC_OS_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        EXC_OS_TYPE_VALUE + _PYDANTIC
    )
    """IO + validation + typing catch for config-file model boundaries."""

    EXC_ATTR_KEY_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        AttributeError,
        *EXC_KEY_OS_TYPE_VALUE,
    )
    """Object + mapping + filesystem + typing catch for full IO+state flows."""

    EXC_FS_TYPE_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        *EXC_TYPE_VALIDATION,
        *_PYDANTIC,
    )
    """Filesystem + typing + Pydantic validation catch for config-load flows."""

    ERR_DOMAIN_EVENT_NAME_REQUIRED: ClassVar[str] = (
        "Domain event name must be a non-empty string"
    )
    ERR_DOMAIN_EVENT_DATA_MUST_BE_DICT_OR_NONE: ClassVar[str] = (
        "Domain event data must be a dictionary or None"
    )
    ERR_EVENTS_LIST_OR_TUPLE_REQUIRED: ClassVar[str] = "Events must be a list or tuple"
    ERR_EVENT_NAME_REQUIRED: ClassVar[str] = "Event name must be non-empty string"
    ERR_CHECKER_HANDLER_NO_HANDLE_METHOD: ClassVar[str] = "Handler has no handle method"
    ERR_CHECKER_HANDLER_HANDLE_NOT_CALLABLE: ClassVar[str] = (
        "Handler handle attribute is not callable"
    )
    ERR_CHECKER_NO_MESSAGE_PARAMETER: ClassVar[str] = (
        "No message parameter found in handle"
    )
    ERR_CHECKER_TYPE_HINT_NONE: ClassVar[str] = "Type hint is None"
    ERR_CHECKER_NO_ANNOTATION_OR_TYPE_HINT: ClassVar[str] = (
        "No annotation or type hint for parameter"
    )
    ERR_CHECKER_INVALID_HANDLE_METHOD_SIGNATURE: ClassVar[str] = (
        "Invalid handle method signature"
    )
    ERR_COLLECTION_NO_MATCHING_ITEM_FOUND: ClassVar[str] = "No matching item found"
    ERR_MAPPER_NOT_A_SEQUENCE: ClassVar[str] = "Not a sequence"
    ERR_INFRA_INVALID_HANDLER_MODE: ClassVar[str] = (
        "handler_mode must be 'command' or 'query'"
    )
    ERR_INFRA_HANDLER_REQUIRED: ClassVar[str] = "handler cannot be None"
    ERR_INFRA_MESSAGE_REQUIRED: ClassVar[str] = "message cannot be None"
    ERR_INFRA_TIMEOUT_POSITIVE: ClassVar[str] = "timeout must be positive"
    ERR_INFRA_INVALID_REGISTRATION_ID: ClassVar[str] = (
        "registration_id must be non-empty string"
    )
    ERR_INFRA_INVALID_REQUEST_ID: ClassVar[str] = "request_id must be non-empty string"
    ERR_MIXINS_EMPTY_OPERATION: ClassVar[str] = "Operation name cannot be empty"
    ERR_MIXINS_EMPTY_STATE: ClassVar[str] = "State value cannot be empty"
    ERR_MIXINS_EMPTY_FIELD_NAME: ClassVar[str] = "Field name cannot be empty"
    ERR_MIXINS_INVALID_ENCODING: ClassVar[str] = "Invalid character encoding"
    ERR_MIXINS_MISSING_TIMESTAMP: ClassVar[str] = "Required timestamp fields missing"
    ERR_MIXINS_INVALID_LOG_LEVEL: ClassVar[str] = "Invalid log level"
    ERR_SERVICE_REGISTRATION_FAILED: ClassVar[str] = "Service registration failed"
    ERR_SERVICE_EXECUTION_FAILED: ClassVar[str] = "Service execution failed"
    ERR_TEXT_NONE_NOT_ALLOWED: ClassVar[str] = (
        "Text cannot be None. Use explicit empty string '' or handle None "
        "in calling code."
    )
    ERR_TEXT_EMPTY_NOT_ALLOWED: ClassVar[str] = (
        "Text cannot be empty or whitespace-only. Use explicit non-empty string."
    )
    ERR_PARSER_COERCE_BOOL_FAILED: ClassVar[str] = "Cannot coerce '{value}' to bool"
    ERR_PARSER_COERCE_FLOAT_FAILED: ClassVar[str] = "Cannot coerce {type_name} to float"
    ERR_PARSER_COERCE_INT_FAILED: ClassVar[str] = "Cannot coerce {type_name} to int"
    ERR_PARSER_TARGET_NOT_STRENUM: ClassVar[str] = (
        "{field_prefix}Target is not a StrEnum. Enum mode cannot be used here."
    )
    ERR_PARSER_CANNOT_PARSE_ENUM: ClassVar[str] = (
        "{field_prefix}Cannot parse '{value}' as {target_name} [options: {options}]"
    )
    ERR_PARSER_TARGET_NOT_BASEMODEL: ClassVar[str] = (
        "{field_prefix}Target is not a BaseModel"
    )
    ERR_PARSER_CANNOT_PARSE_SCALAR_TO_MODEL: ClassVar[str] = (
        "{field_prefix}Cannot parse scalar '{value}' into {target_name}"
    )
    ERR_PARSER_TYPEADAPTER_RETURN_MISMATCH: ClassVar[str] = (
        "{field_prefix}TypeAdapter returned {actual_type}, expected {target_type}"
    )
    ERR_PARSER_EXPECTED_DICT_FOR_MODEL: ClassVar[str] = (
        "{field_prefix}Expected dict for model, got {type_name}"
    )
    ERR_PARSER_CANNOT_PARSE_TO_TARGET: ClassVar[str] = (
        "{field_prefix}Cannot parse {source_type} to {target_name}: {error}"
    )
    ERR_PARSER_MODEL_PARSE_FAILED: ClassVar[str] = "Model parse failed: {error}"
    ERR_PARSER_PARSE_FAILED_FOR_TARGET: ClassVar[str] = (
        "{field_prefix}Failed to parse '{value}' as {target_name}"
    )
    ERR_PARSER_VALUE_IS_NONE: ClassVar[str] = "{field_prefix}Value is None"

    # --- Models ---
    ERR_MODEL_UPDATED_AT_BEFORE_CREATED_AT: ClassVar[str] = (
        "updated_at cannot be before created_at"
    )
    ERR_MODEL_VERSION_BELOW_MINIMUM: ClassVar[str] = (
        "Version {version} is below minimum {minimum}"
    )
    ERR_MODEL_MAX_DELAY_LESS_THAN_INITIAL: ClassVar[str] = (
        "max_delay_seconds must be >= initial_delay_seconds"
    )
    ERR_ENTITY_INVARIANT_VIOLATED: ClassVar[str] = (
        "Invariant violated: {invariant_name}"
    )
    ERR_ENTITY_AGGREGATE_INVARIANT_FAILURE: ClassVar[str] = (
        "Aggregate invariant violation: {error}"
    )
    ERR_ENTITY_TOO_MANY_DOMAIN_EVENTS: ClassVar[str] = (
        "Too many uncommitted domain events: {count} (max: {max})"
    )
    ERR_CQRS_PARSE_MESSAGE_NOT_IMPLEMENTED: ClassVar[str] = (
        "parse_message must be implemented by subclasses"
    )
    ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED: ClassVar[str] = (
        "_build_product() must be implemented by builder subclasses"
    )

    # --- Context ---
    ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING: ClassVar[str] = (
        "Cannot normalize {type_name} to Mapping"
    )
    ERR_CONTEXT_FIELD_MUST_HAVE_GET_SET: ClassVar[str] = (
        "Context must have get() and set() methods"
    )

    # --- Utilities ---
    ERR_ENUM_INVALID_VALUE: ClassVar[str] = "Invalid {enum_name}: {value}"
    ERR_COLLECTION_INVALID_ENUM_VALUE: ClassVar[str] = (
        "Invalid {enum_name} value: '{value}'"
    )
    ERR_COLLECTION_EXPECTED_SEQUENCE: ClassVar[str] = (
        "Expected sequence, got {type_name}"
    )
    ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM: ClassVar[str] = (
        "Processing failed for item {item}: {error}"
    )
    ERR_COLLECTION_EXPECTED_STR_FOR_ENUM: ClassVar[str] = (
        "Expected str for enum conversion, got {type_name}"
    )
    ERR_CONFIG_INVALID_DB_URL_SCHEME: ClassVar[str] = "Invalid database URL scheme"
    ERR_CONFIG_TRACE_REQUIRES_DEBUG: ClassVar[str] = "Trace mode requires debug mode"
    ERR_CONFIG_FACTORY_REGISTRATION_FAILED: ClassVar[str] = (
        "Factory registration failed"
    )
    ERR_CONFIG_FACTORY_REGISTRATION_FAILED_FOR_NAME: ClassVar[str] = (
        "Factory registration failed for {name}: {error}"
    )

    # --- Container / Runtime ---
    ERR_CONTAINER_REGISTRATION_FAILED: Final[str] = (
        "Container registration of '{name}' failed: {reason}"
    )
    ERR_CONTAINER_NAME_EMPTY: Final[str] = (
        "Container registration requires a non-empty name"
    )
    ERR_CONTAINER_NAME_DUPLICATE: Final[str] = (
        "Container name '{name}' is already registered; drop it before rebinding"
    )
    ERR_CONTAINER_NAME_RESERVED: Final[str] = (
        "Container name '{name}' is reserved for the core runtime services"
    )
    ERR_CONTAINER_CALLER_UNRESOLVED: Final[str] = (
        "auto_register_factories requires a caller module imported in sys.modules"
    )
    ERR_RUNTIME_METADATA_MODEL_NOT_BOUND: Final[str] = (
        "u.Metadata is not bound to a concrete model"
    )
    ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE: Final[str] = (
        "attributes must be dict-like"
    )
    ERR_RUNTIME_MAPPING_INVALID_TYPE: Final[str] = (
        "Invalid type in Mapping: {type_name}"
    )
    ERR_RUNTIME_SEQUENCE_INVALID_TYPE: Final[str] = (
        "Invalid type in Sequence: {type_name}"
    )
    ERR_RUNTIME_CONTAINER_INVALID_TYPE: Final[str] = (
        "Unsupported runtime container value: {type_name}"
    )
    ERR_RUNTIME_BATCH_VALIDATION_FAILED: Final[str] = (
        "Batch validation failed: {errors}"
    )
    ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED: Final[str] = (
        "Keys starting with '_' are reserved: {key}"
    )
    ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE: Final[str] = (
        "Service must be a RegisterableService type, got {type_name}"
    )
    ERR_RUNTIME_RETRY_LOOP_ENDED_WITHOUT_RESULT: Final[str] = (
        "Retry loop completed without success or exception"
    )
    ERR_RUNTIME_UNSUPPORTED_GENERATOR_KIND: Final[str] = (
        "Unsupported generator kind: {kind}"
    )
    ERR_RUNTIME_CONTAINER_NOT_INITIALIZED: Final[str] = (
        "Container not initialized. Call FlextContext.configure_container(container) "
        "before using resolve_container()."
    )

    # --- Exceptions / Error handling ---
    ERR_EXCEPTIONS_PARAMS_CLS_MISSING: Final[str] = "{class_name} is missing params_cls"
    ERR_EXCEPTIONS_UNKNOWN_ERROR_TYPE: Final[str] = "Unknown error type: {message}"

    # --- Handlers ---
    ERR_HANDLER_UNSUPPORTED_TYPE: Final[str] = (
        "Unsupported handler type: {handler_type}"
    )

    # --- Services ---
    ERR_SERVICE_PORT_TYPE: Final[str] = (
        "{service}.{field}: port type {port_type!r} must be a plain "
        "@runtime_checkable Protocol class; isinstance cannot validate a "
        "subscripted generic, and a concrete class is not a port"
    )
    ERR_SERVICE_OPERATION: Final[str] = (
        "Service operation {service}.{operation} (module {module}, annotation "
        "{annotation}) {defect}"
    )
    ERR_SERVICE_OPERATION_ASYNC: Final[str] = (
        "must be synchronous. Fix: declare it with a plain def."
    )
    ERR_SERVICE_OPERATION_GENERIC: Final[str] = (
        "must not be generic. Fix: remove its type parameters."
    )
    ERR_SERVICE_OPERATION_SIGNATURE: Final[str] = (
        "must take self and at most one positional request parameter without "
        "default. Fix: accept one request Protocol or nothing."
    )
    ERR_SERVICE_OPERATION_DOCSTRING: Final[str] = (
        "must have a docstring. Fix: add a one-line summary docstring."
    )
    ERR_SERVICE_OPERATION_NAME: Final[str] = (
        "must annotate with a dotted name. Fix: use a module-level name such as "
        "p.Ns.Request, or p.Result[...] for the return."
    )
    ERR_SERVICE_OPERATION_EVALUATED: Final[str] = (
        "is evaluated instead of a string annotation. Fix: add "
        "`from __future__ import annotations` to the module."
    )
    ERR_SERVICE_OPERATION_UNBOUND: Final[str] = (
        "names a symbol unbound in the module at runtime (imported only under "
        "TYPE_CHECKING, or undefined). Fix: import it at module runtime."
    )
    ERR_SERVICE_OPERATION_REQUEST: Final[str] = (
        "must take a request Protocol with a Pydantic model declared in operation "
        "metadata. Fix: annotate with p.Request and declare "
        "h.handler(command=m.Request)."
    )
    ERR_SERVICE_OPERATION_RESULT: Final[str] = (
        "must return p.Result. Fix: annotate the return as p.Result[...]."
    )
    ERR_SERVICE_OPERATION_COLLISION: Final[str] = (
        "is declared by sibling classes {owners}. Fix: rename one of them."
    )
    ERR_SERVICE_NO_OPERATIONS: Final[str] = (
        "{service} (module {module}) declares no operation. Fix: declare a "
        "public method below FlextService that returns p.Result."
    )
    ERR_SERVICE_NOT_A_SERVICE: Final[str] = (
        "{service} (module {module}) is not a service: no class of its MRO "
        "declares the service members {members}. Fix: derive it from FlextService."
    )
    ERR_SERVICE_OPERATION_INPUTLESS_COMMAND: Final[str] = (
        "takes no request and cannot declare a command model. Fix: remove the "
        "command from its handler metadata or accept a request Protocol."
    )
    ERR_SERVICE_OPERATION_METADATA: Final[str] = (
        "takes a request but declares no handler metadata. Fix: declare "
        "h.handler(command=m.Request)."
    )
    ERR_SERVICE_OPERATION_COMMAND_MODEL: Final[str] = (
        "declares a handler command that is not a Pydantic request model. Fix: "
        "declare h.handler(command=m.Request) with an m.* request model."
    )
    ERR_SERVICE_OPERATION_PROTOCOL: Final[str] = (
        "takes a request model that does not satisfy its Protocol members "
        "{missing}. Fix: declare those members on the request model."
    )

    # --- Lazy loading ---
    ERR_LAZY_RELATIVE_PATH_REQUIRES_MODULE: Final[str] = (
        "relative child module paths require module_name"
    )

    # --- Settings ---
    ERR_SETTINGS_NAMESPACE_NOT_REGISTERED: Final[str] = (
        "Namespace '{namespace}' not registered"
    )
    ERR_SETTINGS_CLASS_REQUIRED_FOR_NON_DECORATOR: Final[str] = (
        "settings_class is required when decorator=False"
    )
    ERR_SETTINGS_NAMESPACE_TYPE_MISMATCH: Final[str] = (
        "Namespace '{namespace}' settings instance {instance_class} is not instance "
        "of {expected_type}"
    )


__all__ = ["FlextConstantsErrors"]
