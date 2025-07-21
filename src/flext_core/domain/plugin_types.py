"""Unified plugin types for FLEXT components.

This module consolidates all plugin-related types to eliminate duplication
across different projects.
"""

from __future__ import annotations

from enum import StrEnum


class PluginCapability(StrEnum):
    """Plugin capability enumeration using StrEnum for type safety."""

    # Data processing capabilities
    DATA_EXTRACTION = "data_extraction"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"

    # Integration capabilities
    API_INTEGRATION = "api_integration"
    DATABASE_INTEGRATION = "database_integration"
    FILE_INTEGRATION = "file_integration"
    MESSAGE_QUEUE_INTEGRATION = "message_queue_integration"

    # Authentication capabilities
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"

    # Monitoring capabilities
    MONITORING = "monitoring"
    LOGGING = "logging"
    METRICS = "metrics"
    ALERTING = "alerting"

    # Utility capabilities
    UTILITY = "utility"
    TESTING = "testing"
    DEBUGGING = "debugging"


class PluginLifecycle(StrEnum):
    """Plugin lifecycle states using StrEnum for type safety."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNLOADING = "unloading"


class PluginStatus(StrEnum):
    """Plugin operational status using StrEnum for type safety."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PluginType(StrEnum):
    """Plugin type enumeration."""

    TAP = "tap"
    TARGET = "target"
    TRANSFORM = "transform"
    UTILITY = "utility"
    AUTH = "auth"
    MONITORING = "monitoring"


class PluginSource(StrEnum):
    """Plugin source enumeration."""

    HUB = "hub"
    PYPI = "pypi"
    GIT = "git"
    LOCAL = "local"
    CUSTOM = "custom"
