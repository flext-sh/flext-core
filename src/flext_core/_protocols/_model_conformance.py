# AUTO-GENERATED FILE — Regenerate with: make gen
"""Static instance-to-protocol evidence required by candidate admission.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_model_conformance
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import m, p

    def model_protocol_1(value: m.AggregateRoot) -> p.AggregateRoot:
        return value

    def model_protocol_2(value: m.ArbitraryTypesModel) -> p.ArbitraryTypesModel:
        return value

    def model_protocol_3(value: m.AsyncLogJob) -> p.AsyncLogJob:
        return value

    def model_protocol_4(
        value: m.AttributeAccessErrorParams,
    ) -> p.AttributeAccessErrorParams:
        return value

    def model_protocol_5(
        value: m.AuthenticationErrorParams,
    ) -> p.AuthenticationErrorParams:
        return value

    def model_protocol_6(
        value: m.AuthorizationErrorParams,
    ) -> p.AuthorizationErrorParams:
        return value

    def model_protocol_7(value: m.BaseModel) -> p.BaseModel:
        return value

    def model_protocol_8(value: m.BaseSettings) -> p.BaseSettings:
        return value

    def model_protocol_9(
        value: m.CircuitBreakerErrorParams,
    ) -> p.CircuitBreakerErrorParams:
        return value

    def model_protocol_10(value: m.CombinedRailwayOptions) -> p.CombinedRailwayOptions:
        return value

    def model_protocol_11(value: m.Command) -> p.Command:
        return value

    def model_protocol_12(value: m.ConfigDocument) -> p.ConfigDocument:
        return value

    def model_protocol_13(value: m.ConfigMap) -> p.ConfigMap:
        return value

    def model_protocol_14(
        value: m.ConfigurationErrorParams,
    ) -> p.ConfigurationErrorParams:
        return value

    def model_protocol_15(value: m.ConflictErrorParams) -> p.ConflictErrorParams:
        return value

    def model_protocol_16(value: m.ConnectionErrorParams) -> p.ConnectionErrorParams:
        return value

    def model_protocol_17(value: m.ContainerConfig) -> p.ContainerConfig:
        return value

    def model_protocol_18(value: m.ContextContainerState) -> p.ContextContainerState:
        return value

    def model_protocol_19(value: m.ContextData) -> p.ContextData:
        return value

    def model_protocol_20(value: m.ContextExport) -> p.ContextExport:
        return value

    def model_protocol_21(value: m.ContextRuntimeState) -> p.ContextRuntimeState:
        return value

    def model_protocol_22(value: m.ContextState) -> p.ContextState:
        return value

    def model_protocol_23(value: m.ContextStatistics) -> p.ContextStatistics:
        return value

    def model_protocol_24(value: m.ContractModel) -> p.ContractModel:
        return value

    def model_protocol_25(value: m.DecoratorConfig) -> p.DecoratorConfig:
        return value

    def model_protocol_26(value: m.Dict) -> p.Dict:
        return value

    def model_protocol_27(value: m.DomainEvent) -> p.DomainEvent:
        return value

    def model_protocol_28(value: m.DynamicModel) -> p.DynamicModel:
        return value

    def model_protocol_29(value: m.EnforcedModel) -> p.EnforcedModel:
        return value

    def model_protocol_30(value: m.Entity) -> p.Entity:
        return value

    def model_protocol_31(value: m.EnumManagedModel) -> p.EnumManagedModel:
        return value

    def model_protocol_32(value: m.Event) -> p.Event:
        return value

    def model_protocol_33(
        value: m.ExceptionFactoryOptions,
    ) -> p.ExceptionFactoryOptions:
        return value

    def model_protocol_34(value: m.ExceptionInitOptions) -> p.ExceptionInitOptions:
        return value

    def model_protocol_35(
        value: m.ExceptionMetricsSnapshot,
    ) -> p.ExceptionMetricsSnapshot:
        return value

    def model_protocol_36(value: m.ExceptionMetricsState) -> p.ExceptionMetricsState:
        return value

    def model_protocol_37(value: m.ExecutionContext) -> p.ExecutionContext:
        return value

    def model_protocol_38(
        value: m.ExpectedActualTypeParams,
    ) -> p.ExpectedActualTypeParams:
        return value

    def model_protocol_39(value: m.ExternalContractModel) -> p.ExternalContractModel:
        return value

    def model_protocol_40(
        value: m.ExtractResolvePathPartContext,
    ) -> p.ExtractResolvePathPartContext:
        return value

    def model_protocol_41(value: m.FactoryDecoratorConfig) -> p.FactoryDecoratorConfig:
        return value

    def model_protocol_42(value: m.FactoryRegistration) -> p.FactoryRegistration:
        return value

    def model_protocol_43(value: m.FlexibleInternalModel) -> p.FlexibleInternalModel:
        return value

    def model_protocol_44(value: m.FlexibleModel) -> p.FlexibleModel:
        return value

    def model_protocol_45(value: m.FrozenDynamicModel) -> p.FrozenDynamicModel:
        return value

    def model_protocol_46(value: m.FrozenModel) -> p.FrozenModel:
        return value

    def model_protocol_47(value: m.FrozenValueModel) -> p.FrozenValueModel:
        return value

    def model_protocol_48(value: m.GenerateOptions) -> p.GenerateOptions:
        return value

    def model_protocol_49(value: m.GuardCheckSpec) -> p.GuardCheckSpec:
        return value

    def model_protocol_50(value: m.HandlerConfig) -> p.HandlerConfig:
        return value

    def model_protocol_51(value: m.HandlerRegistration) -> p.HandlerRegistration:
        return value

    def model_protocol_52(
        value: m.HandlerRegistrationReport,
    ) -> p.HandlerRegistrationReport:
        return value

    def model_protocol_53(value: m.HandlerRuntimeState) -> p.HandlerRuntimeState:
        return value

    def model_protocol_54(value: m.ImmutableValueModel) -> p.ImmutableValueModel:
        return value

    def model_protocol_55(value: m.LoggedCallSpec) -> p.LoggedCallSpec:
        return value

    def model_protocol_56(value: m.LoggingRecord) -> p.LoggingRecord:
        return value

    def model_protocol_57(value: m.ManagedModel) -> p.ManagedModel:
        return value

    def model_protocol_58(value: m.Metadata) -> p.Metadata:
        return value

    def model_protocol_59(value: m.ModelDumpOptions) -> p.ModelDumpOptions:
        return value

    def model_protocol_60(value: m.NormalizedModel) -> p.NormalizedModel:
        return value

    def model_protocol_61(value: m.NotFoundErrorParams) -> p.NotFoundErrorParams:
        return value

    def model_protocol_62(value: m.ObjectList) -> p.ObjectList:
        return value

    def model_protocol_63(value: m.OperationErrorParams) -> p.OperationErrorParams:
        return value

    def model_protocol_64(value: m.Pagination) -> p.Pagination:
        return value

    def model_protocol_65(value: m.ParamsModel) -> p.ParamsModel:
        return value

    def model_protocol_66[T]() -> type[p.ParseOptions[T]]:
        return m.ParseOptions[T]

    def model_protocol_67(value: m.Project) -> p.Project:
        return value

    def model_protocol_68(value: m.ProjectAuthor) -> p.ProjectAuthor:
        return value

    def model_protocol_69(value: m.ProjectMetadata) -> p.ProjectMetadata:
        return value

    def model_protocol_70(value: m.ProjectToolFlext) -> p.ProjectToolFlext:
        return value

    def model_protocol_71(value: m.ProjectToolFlextDocs) -> p.ProjectToolFlextDocs:
        return value

    def model_protocol_72(
        value: m.ProjectToolFlextNamespace,
    ) -> p.ProjectToolFlextNamespace:
        return value

    def model_protocol_73(
        value: m.ProjectToolFlextProject,
    ) -> p.ProjectToolFlextProject:
        return value

    def model_protocol_74(
        value: m.ProjectToolFlextReadmeSection,
    ) -> p.ProjectToolFlextReadmeSection:
        return value

    def model_protocol_75(
        value: m.ProjectToolFlextWorkspace,
    ) -> p.ProjectToolFlextWorkspace:
        return value

    def model_protocol_76(value: m.ProjectUrls) -> p.ProjectUrls:
        return value

    def model_protocol_77(value: m.PyprojectDocument) -> p.PyprojectDocument:
        return value

    def model_protocol_78(value: m.PyprojectTool) -> p.PyprojectTool:
        return value

    def model_protocol_79(value: m.Query) -> p.Query:
        return value

    def model_protocol_80(value: m.RateLimitErrorParams) -> p.RateLimitErrorParams:
        return value

    def model_protocol_81(value: m.RegistrationDetails) -> p.RegistrationDetails:
        return value

    def model_protocol_82(value: m.RegistrationEnvelope) -> p.RegistrationEnvelope:
        return value

    def model_protocol_83(value: m.RegistryOptions) -> p.RegistryOptions:
        return value

    def model_protocol_84(value: m.RegistryPluginParams) -> p.RegistryPluginParams:
        return value

    def model_protocol_85(value: m.RegistryServiceState) -> p.RegistryServiceState:
        return value

    def model_protocol_86(value: m.RegistrySummary) -> p.RegistrySummary:
        return value

    def model_protocol_87(value: m.ResourceIdentityParams) -> p.ResourceIdentityParams:
        return value

    def model_protocol_88(value: m.ResourceRegistration) -> p.ResourceRegistration:
        return value

    def model_protocol_89(value: m.ResultFailureSpec) -> p.ResultFailureSpec:
        return value

    def model_protocol_90(value: m.RetryConfiguration) -> p.RetryConfiguration:
        return value

    def model_protocol_91(
        value: m.RetryConfigurationMixin,
    ) -> p.RetryConfigurationMixin:
        return value

    def model_protocol_92(value: m.RetryOptions) -> p.RetryOptions:
        return value

    def model_protocol_93[RootValueT]() -> type[p.RootModel[RootValueT]]:
        return m.RootModel[RootValueT]

    def model_protocol_94(
        value: m.RuntimeBootstrapOptions,
    ) -> p.RuntimeBootstrapOptions:
        return value

    def model_protocol_95(value: m.ServiceLookupParams) -> p.ServiceLookupParams:
        return value

    def model_protocol_96(value: m.ServiceOperation) -> p.ServiceOperation:
        return value

    def model_protocol_97(value: m.ServiceRegistration) -> p.ServiceRegistration:
        return value

    def model_protocol_98(
        value: m.ServiceRegistrationSpec,
    ) -> p.ServiceRegistrationSpec:
        return value

    def model_protocol_99(value: m.ServiceRuntime) -> p.ServiceRuntime:
        return value

    def model_protocol_100(value: m.ServiceState) -> p.ServiceState:
        return value

    def model_protocol_101(value: m.SettingsValue) -> p.SettingsValue:
        return value

    def model_protocol_102(value: m.StrictBoundaryModel) -> p.StrictBoundaryModel:
        return value

    def model_protocol_103(value: m.StrictManagedModel) -> p.StrictManagedModel:
        return value

    def model_protocol_104(value: m.StrictModel) -> p.StrictModel:
        return value

    def model_protocol_105(value: m.StructlogOptions) -> p.StructlogOptions:
        return value

    def model_protocol_106(value: m.StructlogProxyToken) -> p.StructlogProxyToken:
        return value

    def model_protocol_107(value: m.TaggedModel) -> p.TaggedModel:
        return value

    def model_protocol_108(value: m.TimeoutErrorParams) -> p.TimeoutErrorParams:
        return value

    def model_protocol_109(value: m.TimestampedModel) -> p.TimestampedModel:
        return value

    def model_protocol_110(value: m.Token) -> p.Token:
        return value

    def model_protocol_111(value: m.TransformOptions) -> p.TransformOptions:
        return value

    def model_protocol_112(value: m.TransformSpec) -> p.TransformSpec:
        return value

    def model_protocol_113(value: m.TypeErrorParams) -> p.TypeErrorParams:
        return value

    def model_protocol_114(value: m.ValidationErrorParams) -> p.ValidationErrorParams:
        return value

    def model_protocol_115(value: m.ValidatorCallable) -> p.ValidatorCallable:
        return value

    def model_protocol_116(value: m.Value) -> p.Value:
        return value
