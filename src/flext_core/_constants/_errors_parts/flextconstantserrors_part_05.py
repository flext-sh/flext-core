"""Container, runtime, exceptions, lazy, and settings errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsErrorsRuntimeSettings:
    """Container, runtime, exceptions, lazy, and settings errors."""

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
