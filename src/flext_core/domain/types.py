"""FLEXT Domain Types Module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the complete unified typing system for FLEXT framework.
All types here use modern Python 3.13 patterns, eliminate duplication,
and provide the single source of truth for types across all FLEXT projects.
"""

from __future__ import annotations

# Re-export from models for backward compatibility
from flext_core.domain.models import FlextConstants
from flext_core.domain.models import entity_id_field
from flext_core.domain.models import project_name_field
from flext_core.domain.models import validate_entity_id
from flext_core.domain.models import validate_project_name
from flext_core.domain.models import version_field

# Re-export unified types from shared_types and models
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    URL,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    AlertSeverity,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ApiKey,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    BaseConfigDict,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    BaseEntityDict,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    BatchSize,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    CommandServiceProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ConfigMapping,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ConfigurableProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ConfigurationKey,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ConfigurationValue,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    CpuPercent,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    CreatedAt,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DatabaseName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DatabaseURL,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DependencyConfiguratorProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DirPath,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DiskMB,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    DurationSeconds,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EntityCollection,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EntityId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EntityMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EntityProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EntityStatus,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Environment,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EnvironmentLiteral,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    EnvironmentMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    FileName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    FilePath,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    FileSize,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    HandlerFunction,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    IdentifierMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    IPAddress,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Json,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    JsonDict,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    JsonList,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    JsonSchema,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    LogLevel,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    LogLevelLiteral,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    MemoryMB,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    MergeableProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    MetricType,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    NonEmptyStr,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    NonNegativeFloat,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    NonNegativeInt,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    # Oracle database types
    OracleArraySize,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleFetchSize,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleHost,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OraclePassword,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OraclePort,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleQueryTimeout,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleSchema,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleServiceName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleSID,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleUsername,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleWMSAuthMethod,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleWMSEntityType,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleWMSFilterOperator,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleWMSPageMode,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    OracleWMSWriteMode,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Password,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PipelineDict,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PipelineId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PluginDict,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PluginId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PluginType,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Port,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PositiveFloat,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    PositiveInt,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ProjectMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ProjectName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ProjectSettingsProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    QueryServiceProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ResultStatus,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    RetryCount,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    RetryDelay,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ServiceResult,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SessionId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    # Singer protocol types
    SingerBatchSize,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerBookmark,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerCatalog,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerMaxRecords,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerParallelStreams,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerRecordCount,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerReplicationMethod,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerSchemaName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerState,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerStateInterval,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    SingerStreamName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    StatusMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    TimeoutSeconds,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    TimestampISO,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    TimestampMixin,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Token,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    TraceStatus,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    TransactionId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    UpdatedAt,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    UserId,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Username,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    ValidationServiceProtocol,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    Version,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSCompanyCode,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSConnectionInfo,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSEntityInfo,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSEntityName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSFacilityCode,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSFieldMapping,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSFieldName,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSItemID,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSLocationID,
)
from flext_core.domain.shared_types import (  # Typed dicts; Generic types; Entity types; Protocols; Enums; Base types; Result pattern; Singer types; Mixins; WMS types
    WMSOrderNumber,
)

# Backward compatibility aliases
ConfigDict = BaseConfigDict
EntityDict = BaseEntityDict
ConfigKey = ConfigurationKey
ConfigValue = ConfigurationValue
Status = EntityStatus
StrEnum = str  # For backward compatibility

# ISP backward compatibility - deprecated, use specific protocols instead
ConfigProtocol = ConfigurableProtocol
SettingsProtocol = ProjectSettingsProtocol
ServiceProtocol = CommandServiceProtocol

__all__ = [
    "URL",
    "AlertSeverity",
    "ApiKey",
    "BaseConfigDict",
    "BaseEntityDict",
    "BatchSize",
    "CommandServiceProtocol",
    "ConfigDict",
    "ConfigKey",
    "ConfigMapping",
    "ConfigProtocol",  # Deprecated - use ConfigurableProtocol
    "ConfigValue",
    "ConfigurableProtocol",
    "ConfigurationKey",
    "ConfigurationValue",
    "CpuPercent",
    "CreatedAt",
    "DatabaseName",
    "DatabaseURL",
    "DependencyConfiguratorProtocol",
    "DirPath",
    "DiskMB",
    "DurationSeconds",
    "EntityCollection",
    "EntityDict",
    "EntityId",
    "EntityMixin",
    "EntityProtocol",
    "EntityStatus",
    "Environment",
    "EnvironmentLiteral",
    "EnvironmentMixin",
    "FileName",
    "FilePath",
    "FileSize",
    "FlextConstants",
    "HandlerFunction",
    "IPAddress",
    "IdentifierMixin",
    "Json",
    "JsonDict",
    "JsonList",
    "JsonSchema",
    "LogLevel",
    "LogLevelLiteral",
    "MemoryMB",
    "MergeableProtocol",
    "MetricType",
    "NonEmptyStr",
    "NonNegativeFloat",
    "NonNegativeInt",
    # Oracle database types
    "OracleArraySize",
    "OracleFetchSize",
    "OracleHost",
    "OraclePassword",
    "OraclePort",
    "OracleQueryTimeout",
    "OracleSID",
    "OracleSchema",
    "OracleServiceName",
    "OracleUsername",
    "OracleWMSAuthMethod",
    "OracleWMSEntityType",
    "OracleWMSFilterOperator",
    "OracleWMSPageMode",
    "OracleWMSWriteMode",
    "Password",
    "PipelineDict",
    "PipelineId",
    "PluginDict",
    "PluginId",
    "PluginType",
    "Port",
    "PositiveFloat",
    "PositiveInt",
    "ProjectMixin",
    "ProjectName",
    "ProjectSettingsProtocol",
    "QueryServiceProtocol",
    "ResultStatus",
    "RetryCount",
    "RetryDelay",
    "ServiceProtocol",  # Deprecated - use specific service protocols
    "ServiceResult",
    "SessionId",
    "SettingsProtocol",  # Deprecated - use ProjectSettingsProtocol
    # Singer protocol types
    "SingerBatchSize",
    "SingerBookmark",
    "SingerCatalog",
    "SingerMaxRecords",
    "SingerParallelStreams",
    "SingerRecordCount",
    "SingerReplicationMethod",
    "SingerSchemaName",
    "SingerState",
    "SingerStateInterval",
    "SingerStreamName",
    "Status",
    "StatusMixin",
    "StrEnum",
    "TimeoutSeconds",
    "TimestampISO",
    "TimestampMixin",
    "Token",
    "TraceStatus",
    "TransactionId",
    "UpdatedAt",
    "UserId",
    "Username",
    "ValidationServiceProtocol",
    "Version",
    "WMSCompanyCode",
    "WMSConnectionInfo",
    "WMSEntityInfo",
    "WMSEntityName",
    "WMSFacilityCode",
    "WMSFieldMapping",
    "WMSFieldName",
    "WMSItemID",
    "WMSLocationID",
    "WMSOrderNumber",
    "entity_id_field",
    "project_name_field",
    "validate_entity_id",
    "validate_project_name",
    "version_field",
]
