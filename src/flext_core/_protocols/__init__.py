# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.collection_models import FlextProtocolsCollectionModels
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.containers import FlextProtocolsContainers
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.cqrs import FlextProtocolsCqrs
    from flext_core._protocols.domain_event import FlextProtocolsDomainEvent
    from flext_core._protocols.entity import FlextProtocolsEntity
    from flext_core._protocols.errors import FlextProtocolsErrors
    from flext_core._protocols.exception_params import FlextProtocolsExceptionParams
    from flext_core._protocols.flext_context import FlextProtocolsFlextContext
    from flext_core._protocols.flext_mixins import FlextProtocolsFlextMixins
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.loggings import FlextProtocolsLogging
    from flext_core._protocols.options import FlextProtocolsOptions
    from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings


__all__: tuple[str, ...] = (
    "FlextProtocolsBase",
    "FlextProtocolsCollectionModels",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContainers",
    "FlextProtocolsContext",
    "FlextProtocolsCqrs",
    "FlextProtocolsDomainEvent",
    "FlextProtocolsEntity",
    "FlextProtocolsErrors",
    "FlextProtocolsExceptionParams",
    "FlextProtocolsFlextContext",
    "FlextProtocolsFlextMixins",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsOptions",
    "FlextProtocolsProjectMetadata",
    "FlextProtocolsPydantic",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextProtocolsSettings",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsBase": ".base",
        "FlextProtocolsCollectionModels": ".collection_models",
        "FlextProtocolsConfig": ".config",
        "FlextProtocolsContainer": ".container",
        "FlextProtocolsContainers": ".containers",
        "FlextProtocolsContext": ".context",
        "FlextProtocolsCqrs": ".cqrs",
        "FlextProtocolsDomainEvent": ".domain_event",
        "FlextProtocolsEntity": ".entity",
        "FlextProtocolsErrors": ".errors",
        "FlextProtocolsExceptionParams": ".exception_params",
        "FlextProtocolsFlextContext": ".flext_context",
        "FlextProtocolsFlextMixins": ".flext_mixins",
        "FlextProtocolsHandler": ".handler",
        "FlextProtocolsLogging": ".loggings",
        "FlextProtocolsOptions": ".options",
        "FlextProtocolsProjectMetadata": ".project_metadata",
        "FlextProtocolsPydantic": ".pydantic",
        "FlextProtocolsRegistry": ".registry",
        "FlextProtocolsResult": ".result",
        "FlextProtocolsService": ".service",
        "FlextProtocolsSettings": ".settings",
    },
    public_exports=__all__,
    deferred_members={
        "FlextProtocolsBase": (
            "HasDomainEvents",
            "FieldSerializerFactory",
            "ModelSerializerFactory",
            "ComputedFieldFactory",
            "Routable",
            "CommandRoutable",
            "EventRoutable",
            "QueryRoutable",
            "ConfigObject",
            "Flushable",
            "Lock",
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
            "RetryConfigurationMixin",
            "TimestampedModel",
        ),
        "FlextProtocolsCollectionModels": (
            "TransformSpec",
            "GuardCheckSpec",
        ),
        "FlextProtocolsConfig": (
            "ConfigLoader",
            "AsyncLogJob",
            "StructlogOptions",
            "ConfigDocument",
            "ModelDumpOptions",
            "ParseOptions",
            "RetryOptions",
            "LoggingRecord",
        ),
        "FlextProtocolsContainer": (
            "RegistryCollaborator",
            "ServiceValue",
            "RegisterableService",
            "FactoryCallable",
            "ResourceCallable",
            "ServiceMap",
            "FactoryMap",
            "ResourceMap",
            "ServiceRegistrationSpec",
            "RegistrationEnvelope",
            "ServiceRegistration",
            "RootDict",
            "MutableRootDict",
            "Container",
            "ContainerLifecycle",
            "ContainerType",
            "ProtocolGuardInput",
            "GuardInput",
            "FactoryRegistration",
            "ResourceRegistration",
            "ContainerConfig",
            "FactoryDecoratorConfig",
        ),
        "FlextProtocolsContainers": (
            "ValidatorCallable",
            "_MappingRootBase",
            "Dict",
            "ConfigMap",
            "ObjectList",
        ),
        "FlextProtocolsContext": (
            "ContextRead",
            "ContextWrite",
            "ContextLifecycle",
            "ContextExporter",
            "ContextMetadataAccess",
            "Context",
            "ContextServiceNamespace",
            "ContextCorrelationNamespace",
            "ContextRequestNamespace",
            "ContextPerformanceNamespace",
            "ContextSerializationNamespace",
            "ContextUtilitiesNamespace",
            "ContextType",
            "ContextProxyToken",
            "ContextProxy",
            "ContextMetadata",
        ),
        "FlextProtocolsCqrs": (
            "Command",
            "Query",
            "HandlerConfig",
            "Event",
            "FlextMessage",
        ),
        "FlextProtocolsDomainEvent": ("DomainEvent",),
        "FlextProtocolsEntity": (
            "Entity",
            "Value",
            "AggregateRoot",
            "TransformOptions",
            "ExtractResolvePathPartContext",
        ),
        "FlextProtocolsErrors": (
            "ResultFailureSpec",
            "LoggedCallSpec",
            "ExceptionMetricsSnapshot",
            "ExceptionMetricsState",
        ),
        "FlextProtocolsExceptionParams": (),
        "FlextProtocolsFlextContext": ("ContextState",),
        "FlextProtocolsFlextMixins": ("ServiceState",),
        "FlextProtocolsHandler": (
            "HandlerCallable",
            "DispatchableHandler",
            "ResolvedHandlerCallable",
            "RoutedHandlerCallable",
            "DispatcherRouteEntry",
            "ExecutionContext",
            "HandlerRuntimeState",
            "DecoratorConfig",
            "Handler",
            "DispatchMessage",
            "Handle",
            "Execute",
            "AutoDiscoverableHandler",
            "HandlerRegistration",
            "HandlerRegistrationReport",
            "Dispatcher",
            "CommandBus",
            "Middleware",
            "RegistrationDetails",
            "CombinedRailwayOptions",
        ),
        "FlextProtocolsLogging": (
            "LogArgument",
            "LogValue",
            "LoggingEventValue",
            "LoggingEvent",
            "LoggingContext",
            "LogResult",
            "Logger",
            "HasLogger",
            "OutputLogger",
            "LoggingStage",
            "OutputLoggerFactory",
            "NativeLoggingWrapper",
            "LoggerWrapperFactory",
            "LoggingSettings",
            "LoggingObserver",
            "LoggingHandler",
            "Connection",
            "ValidatorSpec",
            "Entry",
            "TextStream",
            "AccessibleData",
        ),
        "FlextProtocolsOptions": ("GenerateOptions",),
        "FlextProtocolsProjectMetadata": (
            "DistributionSource",
            "PyprojectDocument",
        ),
        "FlextProtocolsPydantic": (
            "ConfigDict",
            "EncoderProtocol",
            "ModelWrapValidatorHandler",
            "ValidationInfo",
            "ValidatorFunctionWrapHandler",
            "Base",
            "AttributeProbe",
            "ModuleOwned",
            "ModelCopy",
            "BaseModel",
            "ModelField",
            "ModelType",
            "FunctionDecorator",
            "ValidateCall",
            "UnsetPrivateAttribute",
            "PrivateAttrFactory",
            "BaseSettings",
            "RootModel",
        ),
        "FlextProtocolsRegistry": (
            "RegistrySummary",
            "Registry",
            "RegistryServiceState",
            "RegistryOptions",
        ),
        "FlextProtocolsResult": (
            "ModelInput",
            "ConfigModelInput",
            "MetadataInput",
            "ResultView",
            "FailureLike",
            "ResultState",
            "ResultOps",
            "Result",
            "RailwayDecorator",
            "SuccessCheckable",
            "StructuredError",
            "HasModelDump",
        ),
        "FlextProtocolsService": (
            "ServiceOperationHandler",
            "ServiceOperation",
            "RuntimeBootstrapProvider",
            "MixinsInfrastructure",
            "Service",
            "DispatchableService",
            "ServiceRuntime",
            "RuntimeBootstrapOptions",
        ),
        "FlextProtocolsSettings": (
            "SettingsOverride",
            "SettingsOverridesMapping",
            "Configurable",
            "Settings",
            "SettingsType",
            "SettingsValue",
            "RetryConfiguration",
        ),
    },
)

defer_lazy_bases(
    __name__,
    {
        "FlextProtocolsContext": (
            ("flext_core._protocols._context", "FlextProtocolsContextData"),
            ("flext_core._protocols._context", "FlextProtocolsContextExport"),
            ("flext_core._protocols._context", "FlextProtocolsContextScope"),
            ("flext_core._protocols._context", "FlextProtocolsContextTokens"),
        ),
        "FlextProtocolsCqrs": (
            (
                "flext_core._protocols._cqrs",
                "FlextProtocolsCqrsFlextmodelscqrsPagination",
            ),
        ),
        "FlextProtocolsExceptionParams": (
            (
                "flext_core._protocols._exception_params",
                "FlextProtocolsExceptionParamsOperationErrors",
            ),
            (
                "flext_core._protocols._exception_params",
                "FlextProtocolsExceptionParamsServiceErrors",
            ),
            (
                "flext_core._protocols._exception_params",
                "FlextProtocolsExceptionParamsBase",
            ),
        ),
        "FlextProtocolsProjectMetadata": (
            (
                "flext_core._protocols._project_metadata",
                "FlextProtocolsProjectMetadataDocument",
            ),
            (
                "flext_core._protocols._project_metadata",
                "FlextProtocolsProjectMetadataAggregates",
            ),
            (
                "flext_core._protocols._project_metadata",
                "FlextProtocolsProjectMetadataFields",
            ),
        ),
    },
)
