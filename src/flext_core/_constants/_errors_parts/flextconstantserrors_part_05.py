"""Container, runtime, exceptions, lazy, and settings errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsErrorsRuntimeSettings:
    """Container, runtime, exceptions, lazy, and settings errors."""

    # --- Container / Runtime ---
    ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE: Final[str] = (
        "Factory '{name}' returned value that does not satisfy RegisterableService"
        " protocol. Expected a canonical registerable service, protocol, or callable."
    )
    ERR_CONTAINER_REGISTRATION_FAILED: Final[str] = (
        "Container registration of '{name}' failed: {reason}"
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
        "default. Fix: accept one Pydantic request model or nothing."
    )
    ERR_SERVICE_OPERATION_DOCSTRING: Final[str] = (
        "must have a docstring. Fix: add a one-line summary docstring."
    )
    ERR_SERVICE_OPERATION_NAME: Final[str] = (
        "must annotate with a dotted name. Fix: use a module-level name such as "
        "m.Ns.Request, or p.Result[...] for the return."
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
        "must take a Pydantic model class as the request. Fix: annotate the "
        "request with a Pydantic model subclass."
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
        "Namespace '{namespace}' settings instance {instance_class} is not instance "
        "of {expected_type}"
    )
