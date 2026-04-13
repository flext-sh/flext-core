"""FlextConstantsErrors - error domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, override


class FlextConstantsErrors:
    """Error domain constants for structured error routing."""

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
    ERR_SERVICE_NOT_FOUND: Final[str] = "Service '{name}' not found"
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
    ERR_DOMAIN_EVENT_NAME_REQUIRED: Final[str] = (
        "Domain event name must be a non-empty string"
    )
    ERR_DOMAIN_EVENT_DATA_MUST_BE_DICT_OR_NONE: Final[str] = (
        "Domain event data must be a dictionary or None"
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
    ERR_MAPPER_FOUND_NONE_INDEX: Final[str] = ERR_TEMPLATE_FOUND_NONE.format(
        key="index"
    )
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
    ERR_SERVICE_EXECUTION_FAILED: Final[str] = "Service execution failed"
    ERR_TEXT_NONE_NOT_ALLOWED: Final[str] = (
        "Text cannot be None. Use explicit empty string '' or handle None in calling code."
    )
    ERR_TEXT_EMPTY_NOT_ALLOWED: Final[str] = (
        "Text cannot be empty or whitespace-only. Use explicit non-empty string."
    )
    ERR_PARSER_COERCE_BOOL_FAILED: Final[str] = "Cannot coerce '{value}' to bool"
    ERR_PARSER_COERCE_FLOAT_FAILED: Final[str] = "Cannot coerce {type_name} to float"
    ERR_PARSER_COERCE_INT_FAILED: Final[str] = "Cannot coerce {type_name} to int"
    ERR_PARSER_TARGET_NOT_STRENUM: Final[str] = (
        "{field_prefix}Target is not a StrEnum. Enum mode cannot be used here."
    )
    ERR_PARSER_CANNOT_PARSE_ENUM: Final[str] = (
        "{field_prefix}Cannot parse '{value}' as {target_name} [options: {options}]"
    )
    ERR_PARSER_TARGET_NOT_BASEMODEL: Final[str] = (
        "{field_prefix}Target is not a BaseModel"
    )
    ERR_PARSER_CANNOT_PARSE_SCALAR_TO_MODEL: Final[str] = (
        "{field_prefix}Cannot parse scalar '{value}' into {target_name}"
    )
    ERR_PARSER_TYPEADAPTER_RETURN_MISMATCH: Final[str] = (
        "{field_prefix}TypeAdapter returned {actual_type}, expected {target_type}"
    )
    ERR_PARSER_EXPECTED_DICT_FOR_MODEL: Final[str] = (
        "{field_prefix}Expected dict for model, got {type_name}"
    )
    ERR_PARSER_CANNOT_PARSE_TO_TARGET: Final[str] = (
        "{field_prefix}Cannot parse {source_type} to {target_name}: {error}"
    )
    ERR_PARSER_MODEL_PARSE_FAILED: Final[str] = "Model parse failed: {error}"
    ERR_PARSER_PARSE_FAILED_FOR_TARGET: Final[str] = (
        "{field_prefix}Failed to parse '{value}' as {target_name}"
    )
    ERR_PARSER_VALUE_IS_NONE: Final[str] = "{field_prefix}Value is None"

    # --- Models ---
    ERR_MODEL_UPDATED_AT_BEFORE_CREATED_AT: Final[str] = (
        "updated_at cannot be before created_at"
    )
    ERR_MODEL_VERSION_BELOW_MINIMUM: Final[str] = (
        "Version {version} is below minimum {minimum}"
    )
    ERR_MODEL_MAX_DELAY_LESS_THAN_INITIAL: Final[str] = (
        "max_delay_seconds must be >= initial_delay_seconds"
    )
    ERR_ENTITY_INVARIANT_VIOLATED: Final[str] = "Invariant violated: {invariant_name}"
    ERR_ENTITY_AGGREGATE_INVARIANT_FAILURE: Final[str] = (
        "Aggregate invariant violation: {error}"
    )
    ERR_ENTITY_TOO_MANY_DOMAIN_EVENTS: Final[str] = (
        "Too many uncommitted domain events: {count} (max: {max})"
    )
    ERR_CQRS_PARSE_MESSAGE_NOT_IMPLEMENTED: Final[str] = (
        "parse_message must be implemented by subclasses"
    )
    ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED: Final[str] = (
        "_build_product() must be implemented by builder subclasses"
    )

    # --- Context ---
    ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING: Final[str] = (
        "Cannot normalize {type_name} to Mapping"
    )
    ERR_CONTEXT_FIELD_MUST_HAVE_GET_SET: Final[str] = (
        "Context must have get() and set() methods"
    )

    # --- Utilities ---
    ERR_ENUM_INVALID_VALUE: Final[str] = "Invalid {enum_name}: {value}"
    ERR_COLLECTION_INVALID_ENUM_VALUE: Final[str] = (
        "Invalid {enum_name} value: '{value}'"
    )
    ERR_COLLECTION_EXPECTED_SEQUENCE: Final[str] = "Expected sequence, got {type_name}"
    ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM: Final[str] = (
        "Processing failed for item: {item}"
    )
    ERR_COLLECTION_EXPECTED_STR_FOR_ENUM: Final[str] = (
        "Expected str for enum conversion, got {type_name}"
    )
    ERR_CONFIG_INVALID_DB_URL_SCHEME: Final[str] = "Invalid database URL scheme"
    ERR_CONFIG_TRACE_REQUIRES_DEBUG: Final[str] = "Trace mode requires debug mode"
    ERR_CONFIG_FACTORY_REGISTRATION_FAILED: Final[str] = "Factory registration failed"
    ERR_CONFIG_FACTORY_REGISTRATION_FAILED_FOR_NAME: Final[str] = (
        "Factory registration failed for {name}: {error}"
    )

    # --- Container / Runtime ---
    ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE: Final[str] = (
        "Factory '{name}' returned value that does not satisfy RegisterableService"
        " protocol. Expected a canonical registerable service, protocol, or callable."
    )
    ERR_CONTAINER_CONFIG_NOT_INITIALIZED: Final[str] = (
        "Configuration must be initialized via initialize_registrations"
    )
    ERR_CONTAINER_CONTEXT_NOT_INITIALIZED: Final[str] = (
        "Context not initialized. Provide context during container creation via "
        "FlextContainer(registration=m.ServiceRegistrationSpec(context=...)) or "
        "FlextContainer.shared(context=...)"
    )
    ERR_CONTAINER_PROVIDE_HELPER_NOT_INITIALIZED: Final[str] = (
        "DI bridge Provide helper not initialized"
    )
    ERR_CONTAINER_PROVIDE_HELPER_UNSUPPORTED_TYPE: Final[str] = (
        "DI bridge Provide helper returned unsupported type"
    )
    ERR_CONTAINER_BRIDGE_MUST_HAVE_CONFIG_PROVIDER: Final[str] = (
        "Bridge must have settings provider"
    )
    ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_CANNOT_BE_NONE: Final[str] = (
        "Bridge settings provider cannot be None"
    )
    ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_MUST_SUPPORT_OVERRIDE: Final[str] = (
        "Bridge settings provider must support override()"
    )
    ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED: Final[str] = (
        "Provider '{name}' is already registered"
    )
    ERR_RUNTIME_METADATA_MODEL_NOT_BOUND: Final[str] = (
        "FlextRuntime.Metadata is not bound to a concrete model"
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
        "Container not initialized. Call FlextContext.configure_container(container) before using resolve_container()."
    )

    # --- Exceptions / Error handling ---
    ERR_EXCEPTIONS_PARAMS_CLS_MISSING: Final[str] = (
        "{class_name} is missing _params_cls"
    )
    ERR_EXCEPTIONS_UNKNOWN_ERROR_TYPE: Final[str] = "Unknown error type: {message}"

    # --- Handlers ---
    ERR_HANDLER_UNSUPPORTED_TYPE: Final[str] = (
        "Unsupported handler type: {handler_type}"
    )

    # --- Lazy loading ---
    ERR_LAZY_RELATIVE_PATH_REQUIRES_MODULE: Final[str] = (
        "relative child module paths require module_name"
    )

    # --- Settings ---
    ERR_SETTINGS_NAMESPACE_NOT_REGISTERED: Final[str] = (
        "Namespace '{namespace}' not registered"
    )
    ERR_SETTINGS_DI_PROVIDER_NOT_INITIALIZED: Final[str] = "DI provider not initialized"
    ERR_SETTINGS_CLASS_REQUIRED_FOR_NON_DECORATOR: Final[str] = (
        "settings_class is required when decorator=False"
    )
    ERR_SETTINGS_NAMESPACE_TYPE_MISMATCH: Final[str] = (
        "Namespace '{namespace}' settings instance {instance_class} is not instance of {expected_type}"
    )
