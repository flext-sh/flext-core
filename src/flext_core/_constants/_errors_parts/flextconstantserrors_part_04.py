"""Domain, checker, parser, model, context, and utility errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class FlextConstantsErrorsDomainParser:
    """Domain, checker, parser, model, context, and utility errors."""

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
        "Processing failed for item: {item}"
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
