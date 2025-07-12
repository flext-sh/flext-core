"""FLext Core - Enterprise Foundation Framework.

Copyright (c) 2024 FLEXT Contributors
SPDX-License-Identifier: MIT

Modern Python 3.13 + Pydantic v2 + Clean Architecture.
Zero tolerance for code duplication and technical debt.

Key Principles:
- SOLID: Single responsibility, Open/closed, Liskov substitution,
  Interface segregation, Dependency inversion
- KISS: Keep it simple, stupid
- DRY: Don't repeat yourself
- Performance: Zero overhead abstractions
- Type Safety: 100% typed with modern Python 3.13 syntax
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings as PydanticBaseSettings

# Import constants first to use FlextFramework.VERSION
from flext_core.domain.constants import FlextFramework

# Version
__version__ = FlextFramework.VERSION

# Application Layer - Command/Query patterns
from flext_core.application import PipelineService

# Configuration System
from flext_core.config import BaseConfig
from flext_core.config import BaseSettings
from flext_core.config import ConfigSection
from flext_core.config import ConfigurationError
from flext_core.config import DIContainer
from flext_core.config import configure_container
from flext_core.config import get_config
from flext_core.config import get_container
from flext_core.config import get_settings
from flext_core.config import injectable
from flext_core.config import singleton

# Configuration Adapters
from flext_core.config.adapters import CLIConfig
from flext_core.config.adapters import CLISettings
from flext_core.config.adapters import DjangoSettings
from flext_core.config.adapters import SingerConfig
from flext_core.config.adapters import cli_config_to_dict
from flext_core.config.adapters import django_settings_adapter
from flext_core.config.adapters import singer_config_adapter

# Domain Base Classes - Foundation for all projects
from flext_core.domain import DomainError
from flext_core.domain import Pipeline
from flext_core.domain import PipelineExecution
from flext_core.domain import PipelineId
from flext_core.domain import PipelineName
from flext_core.domain import ServiceResult

# Import remaining constants
from flext_core.domain.constants import EntityStatuses
from flext_core.domain.constants import Environments
from flext_core.domain.constants import ErrorMessages
from flext_core.domain.constants import ExecutionStatuses
from flext_core.domain.constants import FlextFramework
from flext_core.domain.constants import LogLevels
from flext_core.domain.constants import PluginTypes
from flext_core.domain.constants import RegexPatterns
from flext_core.domain.constants import ResultStatuses
from flext_core.domain.constants import SuccessMessages

# Pydantic Base Models - Foundation for all Pydantic usage
from flext_core.domain.pydantic_base import APIBaseModel
from flext_core.domain.pydantic_base import APIPaginatedResponse
from flext_core.domain.pydantic_base import APIRequest
from flext_core.domain.pydantic_base import APIResponse
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.domain.pydantic_base import Field  # Export Field for all projects

# Shared Domain Models - Common models for all modules
from flext_core.domain.shared_models import AuthToken
from flext_core.domain.shared_models import ComponentHealth
from flext_core.domain.shared_models import DatabaseConfig
from flext_core.domain.shared_models import DataRecord
from flext_core.domain.shared_models import DataSchema
from flext_core.domain.shared_models import ErrorDetail
from flext_core.domain.shared_models import ErrorResponse
from flext_core.domain.shared_models import HealthStatus
from flext_core.domain.shared_models import LDAPEntry
from flext_core.domain.shared_models import LDAPScope
from flext_core.domain.shared_models import LogLevel
from flext_core.domain.shared_models import OperationStatus
from flext_core.domain.shared_models import PipelineConfig
from flext_core.domain.shared_models import PipelineRunStatus
from flext_core.domain.shared_models import PluginMetadata
from flext_core.domain.shared_models import PluginType
from flext_core.domain.shared_models import RedisConfig
from flext_core.domain.shared_models import SystemHealth
from flext_core.domain.shared_models import UserInfo

# Advanced Types - Modern Python 3.13 patterns
from flext_core.domain.types import EntityId
from flext_core.domain.types import EntityStatus
from flext_core.domain.types import Environment
from flext_core.domain.types import PluginId
from flext_core.domain.types import ProjectName
from flext_core.domain.types import ResultStatus
from flext_core.domain.types import Status
from flext_core.domain.types import UserId
from flext_core.domain.types import Version

# Infrastructure - Base implementations
from flext_core.infrastructure import InMemoryRepository

# Public API - Everything other projects need
__all__ = [
    # Configuration System (sorted)
    "APIBaseModel",
    "APIPaginatedResponse",
    "APIRequest",
    "APIResponse",
    "AuthToken",
    "BaseConfig",
    "BaseModel",
    "BaseSettings",
    "CLIConfig",
    "CLISettings",
    "ComponentHealth",
    "ConfigSection",
    "ConfigurationError",
    "DIContainer",
    "DataRecord",
    "DataSchema",
    "DatabaseConfig",
    "DjangoSettings",
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainEntity",
    "DomainError",
    "DomainEvent",
    "DomainValueObject",
    "EntityId",
    "EntityStatus",
    "EntityStatuses",
    "Environment",
    "Environments",
    "ErrorDetail",
    "ErrorMessages",
    "ErrorResponse",
    "ExecutionStatuses",
    "Field",
    "FlextFramework",
    "HealthStatus",
    "InMemoryRepository",
    "LDAPEntry",
    "LDAPScope",
    "LogLevel",
    "LogLevels",
    "OperationStatus",
    "Pipeline",
    "PipelineConfig",
    "PipelineExecution",
    "PipelineId",
    "PipelineName",
    "PipelineRunStatus",
    "PipelineService",
    "PluginId",
    "PluginMetadata",
    "PluginType",
    "PluginTypes",
    "ProjectName",
    "PydanticBaseSettings",
    "RedisConfig",
    "RegexPatterns",
    "ResultStatus",
    "ResultStatuses",
    "ServiceResult",
    "SingerConfig",
    "Status",
    "SuccessMessages",
    "SystemHealth",
    "UserId",
    "UserInfo",
    "Version",
    # Version
    "__version__",
    "cli_config_to_dict",
    "configure_container",
    "django_settings_adapter",
    "get_config",
    "get_container",
    "get_settings",
    "injectable",
    "singer_config_adapter",
    "singleton",
]
