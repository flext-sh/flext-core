"""Domain layer - Pure business logic with zero dependencies.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Pure business logic with zero dependencies.
"""

from __future__ import annotations

from flext_core.domain.constants import ConfigDefaults
from flext_core.domain.constants import EntityStatuses
from flext_core.domain.constants import Environments
from flext_core.domain.constants import ErrorMessages
from flext_core.domain.constants import ExecutionStatuses
from flext_core.domain.constants import FlextFramework
from flext_core.domain.constants import HTTPStatus
from flext_core.domain.constants import LogLevels
from flext_core.domain.constants import MediaTypes
from flext_core.domain.constants import PipelineStatuses
from flext_core.domain.constants import PluginTypes
from flext_core.domain.constants import RegexPatterns
from flext_core.domain.constants import ResultStatuses
from flext_core.domain.constants import SuccessMessages

# Core domain components - centralized imports
from flext_core.domain.core import DomainError
from flext_core.domain.core import NotFoundError
from flext_core.domain.core import Repository
from flext_core.domain.core import RepositoryError
from flext_core.domain.core import ValidationError
from flext_core.domain.pipeline import ExecutionStatus
from flext_core.domain.pipeline import Pipeline
from flext_core.domain.pipeline import PipelineExecution
from flext_core.domain.pipeline import PipelineId
from flext_core.domain.pipeline import PipelineName
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.domain.types import EntityId
from flext_core.domain.types import EntityStatus
from flext_core.domain.types import ProjectName
from flext_core.domain.types import ServiceResult
from flext_core.domain.types import Version

__all__ = [
    "ConfigDefaults",
    "DomainAggregateRoot",
    "DomainEntity",
    # Core abstractions
    "DomainError",
    "DomainEvent",
    "DomainValueObject",
    # Types
    "EntityId",
    "EntityStatus",
    "EntityStatuses",
    "Environments",
    "ErrorMessages",
    # Pipeline domain
    "ExecutionStatus",
    "ExecutionStatuses",
    "FlextFramework",
    "HTTPStatus",
    "LogLevels",
    "MediaTypes",
    "NotFoundError",
    "Pipeline",
    "PipelineExecution",
    "PipelineId",
    "PipelineName",
    "PipelineStatuses",
    "PluginTypes",
    "ProjectName",
    "RegexPatterns",
    "Repository",
    "RepositoryError",
    "ResultStatuses",
    "ServiceResult",
    "SuccessMessages",
    "ValidationError",
    "Version",
]
