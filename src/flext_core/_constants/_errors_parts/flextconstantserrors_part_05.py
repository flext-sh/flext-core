"""Container, runtime, exceptions, lazy, and settings errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, Self

from ..._settings import FlextSettings


class FlextConstantsErrorsRuntimeSettings(FlextSettings):
    """Container, runtime, exceptions, lazy, and settings errors.

    MRO carries ``FlextSettings`` (ENFORCE-042); the class is a namespace
    holder, never instantiated — class-attribute access resolves via the MRO.
    """

    # ENFORCE-042 namespace-holder contract: ``FlextSettings`` contributes
    # namespacing only — instance machinery stays plain object semantics so the
    # settings singleton/validation machinery cannot leak into instantiated
    # facade composites (e.g. the ``u`` logging facade).
    def __new__(cls, *args: object, **kwargs: object) -> Self:
        return object.__new__(cls)

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = self, args, kwargs

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    __eq__ = object.__eq__

    __hash__ = object.__hash__

    # --- Container / Runtime ---
    ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE: ClassVar[str] = (
        "Factory '{name}' returned value that does not satisfy RegisterableService"
        " protocol. Expected a canonical registerable service, protocol, or callable."
    )
    ERR_CONTAINER_CONFIG_NOT_INITIALIZED: ClassVar[str] = (
        "Configuration must be initialized via initialize_registrations"
    )
    ERR_CONTAINER_CONTEXT_NOT_INITIALIZED: ClassVar[str] = (
        "Context not initialized. Provide context during container creation via "
        "FlextContainer(registration=m.ServiceRegistrationSpec(context=...)) or "
        "FlextContainer.shared(context=...)"
    )
    ERR_CONTAINER_PROVIDE_HELPER_NOT_INITIALIZED: ClassVar[str] = (
        "DI bridge Provide helper not initialized"
    )
    ERR_CONTAINER_PROVIDE_HELPER_UNSUPPORTED_TYPE: ClassVar[str] = (
        "DI bridge Provide helper returned unsupported type"
    )
    ERR_CONTAINER_BRIDGE_MUST_HAVE_CONFIG_PROVIDER: ClassVar[str] = (
        "Bridge must have settings provider"
    )
    ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_CANNOT_BE_NONE: ClassVar[str] = (
        "Bridge settings provider cannot be None"
    )
    ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_MUST_SUPPORT_OVERRIDE: ClassVar[str] = (
        "Bridge settings provider must support override()"
    )
    ERR_RUNTIME_PROVIDER_ALREADY_REGISTERED: ClassVar[str] = (
        "Provider '{name}' is already registered"
    )
    ERR_RUNTIME_METADATA_MODEL_NOT_BOUND: ClassVar[str] = (
        "FlextRuntime.Metadata is not bound to a concrete model"
    )
    ERR_RUNTIME_ATTRIBUTES_MUST_BE_DICT_LIKE: ClassVar[str] = (
        "attributes must be dict-like"
    )
    ERR_RUNTIME_MAPPING_INVALID_TYPE: ClassVar[str] = (
        "Invalid type in Mapping: {type_name}"
    )
    ERR_RUNTIME_SEQUENCE_INVALID_TYPE: ClassVar[str] = (
        "Invalid type in Sequence: {type_name}"
    )
    ERR_RUNTIME_BATCH_VALIDATION_FAILED: ClassVar[str] = (
        "Batch validation failed: {errors}"
    )
    ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED: ClassVar[str] = (
        "Keys starting with '_' are reserved: {key}"
    )
    ERR_RUNTIME_SERVICE_MUST_BE_REGISTERABLE: ClassVar[str] = (
        "Service must be a RegisterableService type, got {type_name}"
    )
    ERR_RUNTIME_RETRY_LOOP_ENDED_WITHOUT_RESULT: ClassVar[str] = (
        "Retry loop completed without success or exception"
    )
    ERR_RUNTIME_UNSUPPORTED_GENERATOR_KIND: ClassVar[str] = (
        "Unsupported generator kind: {kind}"
    )
    ERR_RUNTIME_CONTAINER_NOT_INITIALIZED: ClassVar[str] = (
        "Container not initialized. Call FlextContext.configure_container(container) "
        "before using resolve_container()."
    )

    # --- Exceptions / Error handling ---
    ERR_EXCEPTIONS_PARAMS_CLS_MISSING: ClassVar[str] = "{class_name} is missing params_cls"
    ERR_EXCEPTIONS_UNKNOWN_ERROR_TYPE: ClassVar[str] = "Unknown error type: {message}"

    # --- Handlers ---
    ERR_HANDLER_UNSUPPORTED_TYPE: ClassVar[str] = (
        "Unsupported handler type: {handler_type}"
    )

    # --- Lazy loading ---
    ERR_LAZY_RELATIVE_PATH_REQUIRES_MODULE: ClassVar[str] = (
        "relative child module paths require module_name"
    )

    # --- Settings ---
    ERR_SETTINGS_NAMESPACE_NOT_REGISTERED: ClassVar[str] = (
        "Namespace '{namespace}' not registered"
    )
    ERR_SETTINGS_DI_PROVIDER_NOT_INITIALIZED: ClassVar[str] = "DI provider not initialized"
    ERR_SETTINGS_CLASS_REQUIRED_FOR_NON_DECORATOR: ClassVar[str] = (
        "settings_class is required when decorator=False"
    )
    ERR_SETTINGS_NAMESPACE_TYPE_MISMATCH: ClassVar[str] = (
        "Namespace '{namespace}' settings instance {instance_class} is not instance "
        "of {expected_type}"
    )
