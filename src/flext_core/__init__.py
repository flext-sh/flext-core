"""FLEXT Core - Enterprise Foundation Framework with simplified imports.

🎯 SIMPLE IMPORTS - Use these for ALL new code:
# Core entities (direct access)
from flext_core import Entity, ValueObject, AggregateRoot, DomainEvent, DomainModel
# Result pattern (essential for error handling)
from flext_core import DomainError, ValidationError
from flext_core.domain.shared_types import ServiceResult
# Configuration and utilities
from flext_core import BaseSettings, Field, BaseModel
# Foundation patterns
from flext_core import EntityId, UserId, Timestamp, EventBus

🚨 DEPRECATED LONG PATHS (still work, but discouraged):
❌ from flext_core.domain.pydantic_base import DomainEntity
✅ from flext_core import Entity
❌ from flext_core import ✅ from flext_core import ServiceResult
from flext_core.domain.shared_types import ServiceResult
❌ from flext_core.config.base import BaseSettings
✅ from flext_core import BaseSettings

🔄 MIGRATION STRATEGY:
All complex paths show warnings pointing to simple root-level imports.
Use short, direct imports for maximum productivity and clarity.
Copyright (c) 2025 FLEXT Contributors | MIT License
Python 3.13 + Pydantic v2 + Zero Technical Debt.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Any

# Essential Pydantic re-exports for common usage
from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings

# ORACLE INTERFACES REMOVED - MAJOR ARCHITECTURAL VIOLATION
# ❌ Oracle-specific interfaces violated Clean Architecture
# ✅ Oracle interfaces moved to respective Oracle projects for proper DI
# CLI interfaces REMOVED - ARCHITECTURAL VIOLATION
# ❌ CLI concepts are concrete concerns, not abstract foundation
# ✅ These belong in flext-cli project, not flext-core
from flext_core.application.interfaces.plugin import PluginHealthResult

# Plugin interfaces for projects to implement
from flext_core.application.interfaces.plugin import PluginInfo
from flext_core.application.interfaces.plugin import PluginInstallationRequest
from flext_core.application.interfaces.plugin import PluginInstallationResult
from flext_core.application.interfaces.plugin import PluginManagerProvider
from flext_core.application.interfaces.plugin import PluginUninstallRequest
from flext_core.application.interfaces.plugin import PluginUpdateRequest
from flext_core.application.interfaces.plugin import PluginUpdateResult

# All imports from flext_core modules - NO FALLBACKS (use real implementation)
from flext_core.config.base import BaseConfig as RealBaseConfig
from flext_core.config.base import BaseSettings
from flext_core.config.base import get_settings
from flext_core.config.base import singleton as real_singleton
from flext_core.config.unified_config import APIConfigMixin
from flext_core.config.unified_config import BaseConfigMixin
from flext_core.config.unified_config import DatabaseConfigMixin
from flext_core.config.unified_config import LoggingConfigMixin
from flext_core.config.unified_config import MonitoringConfigMixin
from flext_core.config.unified_config import PerformanceConfigMixin
from flext_core.domain.constants import ConfigDefaults
from flext_core.domain.constants import FlextFramework
from flext_core.domain.core import DomainError
from flext_core.domain.core import ValidationError
from flext_core.domain.mixins import TimestampMixin
from flext_core.domain.pipeline import Pipeline
from flext_core.domain.pipeline import PipelineExecution
from flext_core.domain.pydantic_base import APIRequest
from flext_core.domain.pydantic_base import APIResponse
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.domain.shared_models import ComponentHealth
from flext_core.domain.shared_models import HealthStatus
from flext_core.domain.shared_models import PluginMetadata
from flext_core.domain.shared_types import AlertSeverity
from flext_core.domain.shared_types import DurationSeconds
from flext_core.domain.shared_types import EntityStatus
from flext_core.domain.shared_types import EnvironmentLiteral
from flext_core.domain.shared_types import LogLevel
from flext_core.domain.shared_types import MemoryMB
from flext_core.domain.shared_types import MetricType
from flext_core.domain.shared_types import NonEmptyStr
from flext_core.domain.shared_types import PipelineId
from flext_core.domain.shared_types import PluginId
from flext_core.domain.shared_types import PluginType
from flext_core.domain.shared_types import PositiveInt
from flext_core.domain.shared_types import ServiceResult
from flext_core.domain.shared_types import TimeoutSeconds
from flext_core.domain.shared_types import TraceStatus
from flext_core.foundation import AbstractEntity
from flext_core.foundation import AbstractRepository
from flext_core.foundation import AbstractService
from flext_core.foundation import AbstractValueObject
from flext_core.foundation import EntityId
from flext_core.foundation import EventBus
from flext_core.foundation import ResultPattern
from flext_core.foundation import Serializable
from flext_core.foundation import SpecificationPattern
from flext_core.foundation import Timestamp
from flext_core.foundation import UserId
from flext_core.foundation import Validatable
from flext_core.infrastructure.protocols import ConnectionProtocol
from flext_core.version import __version_info__
from flext_core.version import get_version
from flext_core.version import get_version_info

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for FLEXT Core components."""


__version__ = FlextFramework.VERSION
# ============================================================================
# 🎯 SIMPLIFIED PUBLIC API - Direct imports without complex paths
# ============================================================================
# Simple aliases for direct access
Entity = DomainEntity
ValueObject = DomainValueObject
AggregateRoot = DomainAggregateRoot
DomainModel = DomainBaseModel
Environment = EnvironmentLiteral  # Alias for backward compatibility


# DomainEvent already imported above
def _warn_deprecated_import(
    old_path: str,
    new_path: str,
    simple_import: str = "",
) -> None:
    """Issue helpful deprecation warnings."""
    message_parts = [
        f"❌ '{old_path}' is deprecated.",
    ]
    if simple_import:
        message_parts.append(f"✅ SIMPLE: from flext_core import {simple_import}")
    message_parts.extend([f"🏗️ CLEAN ARCH: from {new_path}", "⏰ Removed in v1.0.0"])
    warnings.warn("\n".join(message_parts), FlextDeprecationWarning, stacklevel=3)


# ============================================================================
# ⚠️ DEPRECATED COMPATIBILITY - Will show helpful warnings
# ============================================================================
# Legacy compatibility - provide deprecated functions that show warnings
# Expose the real BaseConfig class instead of the deprecated function
BaseConfig = RealBaseConfig


def di_container(*_args: Any, **_kwargs: Any) -> Any:
    """Use injectable and singleton decorators instead.

    Args:
        _args: Deprecated arguments (ignored for compatibility)
        _kwargs: Deprecated keyword arguments (ignored for compatibility)

    Returns:
        Empty DIContainer instance for compatibility

    """
    _warn_deprecated_import(
        "flext_core.config.base.DIContainer",
        "flext_core.foundation",
        "injectable, singleton",
    )
    return type("DIContainer", (), {})()


# Uppercase alias for backward compatibility
DIContainer = di_container


def configure_container(*args: Any, **_kwargs: Any) -> Any:
    """Use injectable and singleton decorators instead.

    Args:
        args: Container configuration arguments (used for compatibility)
        _kwargs: Deprecated keyword arguments (ignored for compatibility)

    Returns:
        First argument if provided, None otherwise for compatibility

    """
    _warn_deprecated_import(
        "flext_core.config.base.configure_container",
        "flext_core.foundation",
        "injectable, singleton",
    )
    return args[0] if args else None


def get_container() -> Any:
    """Use injectable and singleton decorators instead."""
    _warn_deprecated_import(
        "flext_core.config.base.get_container",
        "flext_core.foundation",
        "injectable, singleton",
    )
    return type("DIContainer", (), {})()


def injectable(service_type: Any = None) -> Any:
    """Use foundation patterns instead.

    Args:
        service_type: Service type for injection (ignored for compatibility)

    Returns:
        Identity function for deprecated compatibility

    """
    _ = service_type  # Acknowledge parameter for deprecated compatibility
    _warn_deprecated_import(
        "flext_core.config.base.injectable",
        "flext_core.foundation",
        "injectable (from foundation)",
    )
    return lambda y: y


# Export real singleton (not deprecated one)
singleton = real_singleton
# LogLevel is already imported from flext_core.domain.shared_types above
# ============================================================================
# 📦 PUBLIC API EXPORTS
# ============================================================================
__all__ = [
    # Configuration Mixins
    "APIConfigMixin",  # from flext_core import APIConfigMixin
    # API Base Classes
    "APIRequest",  # from flext_core import APIRequest
    "APIResponse",  # from flext_core import APIResponse
    # Foundation Patterns
    "AbstractEntity",  # from flext_core import AbstractEntity
    "AbstractRepository",  # from flext_core import AbstractRepository
    "AbstractService",  # from flext_core import AbstractService
    "AbstractValueObject",  # from flext_core import AbstractValueObject
    "AggregateRoot",  # from flext_core import AggregateRoot
    # Types and Enums
    "AlertSeverity",  # from flext_core import AlertSeverity
    # ⚠️ DEPRECATED - LEGACY COMPATIBILITY (will show warnings)
    "BaseConfig",  # → Use BaseSettings
    "BaseConfigMixin",  # from flext_core import BaseConfigMixin
    "BaseModel",  # from flext_core import BaseModel
    # Configuration
    "BaseSettings",  # from flext_core import BaseSettings
    # CLI Interfaces REMOVED - ARCHITECTURAL VIOLATION
    # ❌ CLI concepts moved to flext-cli where they belong
    # Types and Identifiers
    "ComponentHealth",  # from flext_core import ComponentHealth
    "ConfigDefaults",  # from flext_core import ConfigDefaults
    "ConnectionProtocol",  # from flext_core import ConnectionProtocol
    "DIContainer",  # → Use injectable, singleton
    "DatabaseConfigMixin",  # from flext_core import DatabaseConfigMixin
    "DomainAggregateRoot",  # → Use AggregateRoot
    "DomainBaseModel",  # → Use DomainModel
    "DomainEntity",  # → Use Entity
    "DomainError",  # from flext_core import DomainError
    "DomainEvent",  # from flext_core import DomainEvent
    "DomainModel",  # from flext_core import DomainModel
    "DomainValueObject",  # → Use ValueObject
    "DurationSeconds",  # from flext_core import DurationSeconds
    # ✅ RECOMMENDED - SIMPLE DIRECT IMPORTS
    # ✅ RECOMMENDED - SIMPLE DIRECT IMPORTS
    # Core Domain Models (use these for new code)
    "Entity",  # from flext_core import Entity
    "EntityId",  # from flext_core import EntityId
    "EntityStatus",  # from flext_core import EntityStatus
    "Environment",  # from flext_core import Environment
    "EnvironmentLiteral",  # from flext_core import EnvironmentLiteral
    "EventBus",  # from flext_core import EventBus
    "Field",  # from flext_core import Field
    "FlextFramework",  # from flext_core import FlextFramework
    "HealthStatus",  # from flext_core import HealthStatus
    "LogLevel",  # from flext_core import LogLevel
    "LoggingConfigMixin",  # from flext_core import LoggingConfigMixin
    "MemoryMB",  # from flext_core import MemoryMB
    "MetricType",  # from flext_core import MetricType
    "MonitoringConfigMixin",  # from flext_core import MonitoringConfigMixin
    "NonEmptyStr",  # from flext_core import NonEmptyStr
    # ORACLE INTERFACES REMOVED - ARCHITECTURAL VIOLATION
    # ❌ Oracle-specific interfaces moved to respective Oracle projects
    "PerformanceConfigMixin",  # from flext_core import PerformanceConfigMixin
    "Pipeline",  # from flext_core import Pipeline
    "PipelineExecution",  # from flext_core import PipelineExecution
    "PipelineId",  # from flext_core import PipelineId
    "PluginHealthResult",  # from flext_core import PluginHealthResult
    "PluginId",  # from flext_core import PluginId
    # Plugin Interfaces
    "PluginInfo",  # from flext_core import PluginInfo
    "PluginInstallationRequest",  # from flext_core import PluginInstallationRequest
    "PluginInstallationResult",  # from flext_core import PluginInstallationResult
    "PluginManagerProvider",  # from flext_core import PluginManagerProvider
    "PluginMetadata",  # from flext_core import PluginMetadata
    "PluginType",  # from flext_core import PluginType
    "PluginUninstallRequest",  # from flext_core import PluginUninstallRequest
    "PluginUpdateRequest",  # from flext_core import PluginUpdateRequest
    "PluginUpdateResult",  # from flext_core import PluginUpdateResult
    "PositiveInt",  # from flext_core import PositiveInt
    "PydanticBaseSettings",  # → Use BaseSettings
    "ResultPattern",  # from flext_core import ResultPattern
    "Serializable",  # from flext_core import Serializable
    # Essential Patterns
    "ServiceResult",  # from flext_core import ServiceResult
    "SpecificationPattern",  # from flext_core import SpecificationPattern
    "TimeoutSeconds",  # from flext_core import TimeoutSeconds
    "Timestamp",  # from flext_core import Timestamp
    "TimestampMixin",  # from flext_core import TimestampMixin
    "TraceStatus",  # from flext_core import TraceStatus
    "UserId",  # from flext_core import UserId
    "Validatable",  # from flext_core import Validatable
    "ValidationError",  # from flext_core import ValidationError
    "ValueObject",  # from flext_core import ValueObject
    # Metadata
    "__version__",
    "__version_info__",
    "configure_container",  # → Use injectable, singleton
    "get_container",  # → Use injectable, singleton
    "get_settings",  # from flext_core import get_settings
    "get_version",  # from flext_core import get_version
    "get_version_info",  # from flext_core import get_version_info
    "injectable",  # → Use from foundation
    "singleton",  # → Use from foundation
]
