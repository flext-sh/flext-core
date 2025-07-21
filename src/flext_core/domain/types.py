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
# Categories: Typed dicts, Generic types, Entity types, Protocols, Enums,
# Base types, Result pattern, Singer types, Mixins, WMS types
from flext_core.domain.shared_types import URL
from flext_core.domain.shared_types import AlertSeverity
from flext_core.domain.shared_types import ApiKey
from flext_core.domain.shared_types import BaseConfigDict
from flext_core.domain.shared_types import BaseEntityDict
from flext_core.domain.shared_types import BatchSize
from flext_core.domain.shared_types import CommandServiceProtocol
from flext_core.domain.shared_types import ConfigMapping
from flext_core.domain.shared_types import ConfigurableProtocol
from flext_core.domain.shared_types import ConfigurationKey
from flext_core.domain.shared_types import ConfigurationValue
from flext_core.domain.shared_types import CpuPercent
from flext_core.domain.shared_types import CreatedAt
from flext_core.domain.shared_types import DatabaseName
from flext_core.domain.shared_types import DatabaseURL
from flext_core.domain.shared_types import DependencyConfiguratorProtocol
from flext_core.domain.shared_types import DirPath
from flext_core.domain.shared_types import DiskMB
from flext_core.domain.shared_types import DurationSeconds
from flext_core.domain.shared_types import EntityCollection
from flext_core.domain.shared_types import EntityId
from flext_core.domain.shared_types import EntityMixin
from flext_core.domain.shared_types import EntityProtocol
from flext_core.domain.shared_types import EntityStatus
from flext_core.domain.shared_types import Environment
from flext_core.domain.shared_types import EnvironmentLiteral
from flext_core.domain.shared_types import EnvironmentMixin
from flext_core.domain.shared_types import FileName
from flext_core.domain.shared_types import FilePath
from flext_core.domain.shared_types import FileSize
from flext_core.domain.shared_types import HandlerFunction
from flext_core.domain.shared_types import IdentifierMixin
from flext_core.domain.shared_types import IPAddress
from flext_core.domain.shared_types import Json
from flext_core.domain.shared_types import JsonDict
from flext_core.domain.shared_types import JsonList
from flext_core.domain.shared_types import JsonSchema
from flext_core.domain.shared_types import LogLevel
from flext_core.domain.shared_types import LogLevelLiteral
from flext_core.domain.shared_types import MemoryMB
from flext_core.domain.shared_types import MergeableProtocol
from flext_core.domain.shared_types import MetricType
from flext_core.domain.shared_types import NonEmptyStr
from flext_core.domain.shared_types import NonNegativeFloat
from flext_core.domain.shared_types import NonNegativeInt
from flext_core.domain.shared_types import OracleArraySize  # Oracle database types
from flext_core.domain.shared_types import OracleFetchSize
from flext_core.domain.shared_types import OracleHost
from flext_core.domain.shared_types import OraclePassword
from flext_core.domain.shared_types import OraclePort
from flext_core.domain.shared_types import OracleQueryTimeout
from flext_core.domain.shared_types import OracleSchema
from flext_core.domain.shared_types import OracleServiceName
from flext_core.domain.shared_types import OracleSID
from flext_core.domain.shared_types import OracleUsername
from flext_core.domain.shared_types import OracleWMSAuthMethod
from flext_core.domain.shared_types import OracleWMSEntityType
from flext_core.domain.shared_types import OracleWMSFilterOperator
from flext_core.domain.shared_types import OracleWMSPageMode
from flext_core.domain.shared_types import OracleWMSWriteMode
from flext_core.domain.shared_types import Password
from flext_core.domain.shared_types import PipelineDict
from flext_core.domain.shared_types import PipelineId
from flext_core.domain.shared_types import PluginDict
from flext_core.domain.shared_types import PluginId
from flext_core.domain.shared_types import PluginType
from flext_core.domain.shared_types import Port
from flext_core.domain.shared_types import PositiveFloat
from flext_core.domain.shared_types import PositiveInt
from flext_core.domain.shared_types import ProjectMixin
from flext_core.domain.shared_types import ProjectName
from flext_core.domain.shared_types import ProjectSettingsProtocol
from flext_core.domain.shared_types import QueryServiceProtocol
from flext_core.domain.shared_types import ResultStatus
from flext_core.domain.shared_types import RetryCount
from flext_core.domain.shared_types import RetryDelay
from flext_core.domain.shared_types import ServiceResult
from flext_core.domain.shared_types import SessionId
from flext_core.domain.shared_types import SingerBatchSize  # Singer protocol types
from flext_core.domain.shared_types import SingerBookmark
from flext_core.domain.shared_types import SingerCatalog
from flext_core.domain.shared_types import SingerMaxRecords
from flext_core.domain.shared_types import SingerParallelStreams
from flext_core.domain.shared_types import SingerRecordCount
from flext_core.domain.shared_types import SingerReplicationMethod
from flext_core.domain.shared_types import SingerSchemaName
from flext_core.domain.shared_types import SingerState
from flext_core.domain.shared_types import SingerStateInterval
from flext_core.domain.shared_types import SingerStreamName
from flext_core.domain.shared_types import StatusMixin
from flext_core.domain.shared_types import TimeoutSeconds
from flext_core.domain.shared_types import TimestampISO
from flext_core.domain.shared_types import TimestampMixin
from flext_core.domain.shared_types import Token
from flext_core.domain.shared_types import TraceStatus
from flext_core.domain.shared_types import TransactionId
from flext_core.domain.shared_types import UpdatedAt
from flext_core.domain.shared_types import UserId
from flext_core.domain.shared_types import Username
from flext_core.domain.shared_types import ValidationServiceProtocol
from flext_core.domain.shared_types import Version
from flext_core.domain.shared_types import WMSCompanyCode
from flext_core.domain.shared_types import WMSConnectionInfo
from flext_core.domain.shared_types import WMSEntityInfo
from flext_core.domain.shared_types import WMSEntityName
from flext_core.domain.shared_types import WMSFacilityCode
from flext_core.domain.shared_types import WMSFieldMapping
from flext_core.domain.shared_types import WMSFieldName
from flext_core.domain.shared_types import WMSItemID
from flext_core.domain.shared_types import WMSLocationID
from flext_core.domain.shared_types import WMSOrderNumber

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
