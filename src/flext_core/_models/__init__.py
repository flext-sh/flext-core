# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models.base import FlextModelsBase
    from flext_core._models.builder import FlextModelsBuilder
    from flext_core._models.collection_models import FlextModelsCollections
    from flext_core._models.config import FlextModelsConfig
    from flext_core._models.container import FlextModelsContainer
    from flext_core._models.containers import FlextModelsContainers
    from flext_core._models.context import FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors
    from flext_core._models.exception_params import FlextModelsExceptionParams
    from flext_core._models.flext_context import FlextModelsContextState
    from flext_core._models.flext_mixins import FlextModelsServiceState
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.options import FlextModelsOptions
    from flext_core._models.project_metadata import FlextModelsProjectMetadata
    from flext_core._models.pydantic import FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsSettings


__all__: tuple[str, ...] = (
    "FlextModelsBase",
    "FlextModelsBuilder",
    "FlextModelsCollections",
    "FlextModelsConfig",
    "FlextModelsContainer",
    "FlextModelsContainers",
    "FlextModelsContext",
    "FlextModelsContextState",
    "FlextModelsCqrs",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsNamespace",
    "FlextModelsOptions",
    "FlextModelsProjectMetadata",
    "FlextModelsPydantic",
    "FlextModelsRegistry",
    "FlextModelsService",
    "FlextModelsServiceState",
    "FlextModelsSettings",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextModelsBase": ".base",
        "FlextModelsBuilder": ".builder",
        "FlextModelsCollections": ".collection_models",
        "FlextModelsConfig": ".config",
        "FlextModelsContainer": ".container",
        "FlextModelsContainers": ".containers",
        "FlextModelsContext": ".context",
        "FlextModelsContextState": ".flext_context",
        "FlextModelsCqrs": ".cqrs",
        "FlextModelsDomainEvent": ".domain_event",
        "FlextModelsEntity": ".entity",
        "FlextModelsErrors": ".errors",
        "FlextModelsExceptionParams": ".exception_params",
        "FlextModelsHandler": ".handler",
        "FlextModelsNamespace": ".namespace",
        "FlextModelsOptions": ".options",
        "FlextModelsProjectMetadata": ".project_metadata",
        "FlextModelsPydantic": ".pydantic",
        "FlextModelsRegistry": ".registry",
        "FlextModelsService": ".service",
        "FlextModelsServiceState": ".flext_mixins",
        "FlextModelsSettings": ".settings",
    },
    public_exports=__all__,
    deferred_members={
        "FlextModelsBase": (
            "EnforcedModel",
            "ManagedModel",
            "EnumManagedModel",
            "NormalizedModel",
            "StrictManagedModel",
            "StrictModel",
            "FrozenModel",
            "ArbitraryTypesModel",
            "StrictBoundaryModel",
            "FlexibleInternalModel",
            "ImmutableValueModel",
            "TaggedModel",
            "FlexibleModel",
            "DynamicModel",
            "FrozenDynamicModel",
            "Metadata",
            "ContractModel",
            "ExternalContractModel",
            "FrozenValueModel",
            "MutableConfiguredMixin",
            "NormalizedMutableConfiguredMixin",
            "IdentifiableMixin",
            "TimestampableMixin",
            "VersionableMixin",
            "RetryConfigurationMixin",
            "TimestampedModel",
        ),
        "FlextModelsBuilder": (
            "Base",
            "Identity",
        ),
        "FlextModelsCollections": (
            "TransformSpec",
            "GuardCheckSpec",
        ),
        "FlextModelsConfig": (
            "AsyncLogJob",
            "StructlogOptions",
            "ConfigDocument",
            "ModelDumpOptions",
            "ParseOptions",
            "RetryOptions",
            "LoggingRecord",
        ),
        "FlextModelsContainer": (
            "RegistrationEnvelope",
            "ServiceRegistration",
            "FactoryRegistration",
            "ResourceRegistration",
            "ContainerConfig",
            "ServiceRegistrationSpec",
            "FactoryDecoratorConfig",
        ),
        "FlextModelsContainers": (
            "ValidatorCallable",
            "_MappingRootBase",
            "Dict",
            "ConfigMap",
            "ObjectList",
        ),
        "FlextModelsContext": (),
        "FlextModelsContextState": ("ContextState",),
        "FlextModelsCqrs": (
            "Command",
            "Query",
            "HandlerConfig",
            "Event",
            "FlextMessage",
        ),
        "FlextModelsDomainEvent": ("DomainEvent",),
        "FlextModelsEntity": (
            "Entity",
            "Value",
            "AggregateRoot",
            "TransformOptions",
            "ExtractResolvePathPartContext",
        ),
        "FlextModelsErrors": (
            "ResultFailureSpec",
            "LoggedCallSpec",
            "ExceptionMetricsSnapshot",
            "ExceptionMetricsState",
        ),
        "FlextModelsExceptionParams": (),
        "FlextModelsHandler": (
            "HandlerRegistration",
            "HandlerRegistrationReport",
            "RegistrationDetails",
            "ExecutionContext",
            "HandlerRuntimeState",
            "DecoratorConfig",
            "CombinedRailwayOptions",
        ),
        "FlextModelsOptions": ("GenerateOptions",),
        "FlextModelsProjectMetadata": ("PyprojectDocument",),
        "FlextModelsPydantic": (
            "Field",
            "FieldInfo",
            "PrivateAttr",
            "JsonSchemaGenerator",
            "BaseModel",
            "BaseSettings",
            "EnvSettingsSource",
            "RootModel",
            "ConfigDict",
            "SettingsConfigDict",
            "ForceDecode",
            "field_validator",
            "model_validator",
            "field_serializer",
            "model_serializer",
            "computed_field",
            "Discriminator",
            "StringConstraints",
            "AliasChoices",
            "AliasPath",
            "AfterValidator",
            "BeforeValidator",
            "FailFast",
            "InstanceOf",
            "PlainValidator",
            "ValidateAs",
            "WrapValidator",
            "SkipValidation",
            "PlainSerializer",
            "SerializeAsAny",
            "WrapSerializer",
            "FieldSerializationInfo",
            "TypeAdapter",
            "GetPydanticSchema",
            "ValidationError",
        ),
        "FlextModelsRegistry": (
            "RegistryServiceState",
            "RegistryOptions",
            "RegistrySummary",
        ),
        "FlextModelsService": (
            "ServiceRuntime",
            "RuntimeBootstrapOptions",
            "ServiceOperation",
        ),
        "FlextModelsServiceState": ("ServiceState",),
        "FlextModelsSettings": (
            "SettingsValue",
            "RetryConfiguration",
        ),
    },
)

defer_lazy_bases(
    __name__,
    {
        "FlextModelsContext": (
            ("flext_core._models._context", "FlextModelsContextData"),
            ("flext_core._models._context", "FlextModelsContextExport"),
            ("flext_core._models._context", "FlextModelsContextScope"),
            ("flext_core._models._context", "FlextModelsContextTokens"),
        ),
        "FlextModelsCqrs": (("flext_core._models._cqrs", "FlextModelsCqrsPagination"),),
        "FlextModelsExceptionParams": (
            (
                "flext_core._models._exception_params",
                "FlextModelsExceptionParamsOperationErrors",
            ),
        ),
        "FlextModelsProjectMetadata": (
            (
                "flext_core._models._project_metadata",
                "FlextModelsProjectMetadataDocument",
            ),
        ),
    },
)
