"""🔥 ULTIMATE UNIFIED CONFIGURATION - ZERO TOLERANCE Architecture.

SINGLE SOURCE OF TRUTH FOR ALL CONFIGURATION + CONSTANTS

This package ELIMINATES ALL configuration duplication across the project:
- DDD + Pydantic + Pydantic-Settings + Python 3.13 type system integration
- Multi-interface (CLI, Web, API, gRPC) unified configuration
- Distributed/parallel/microservices processing configuration
- Environment-specific validation with ZERO fallback patterns
- Enterprise-grade secret management with production validation
"""

from __future__ import annotations

from flx_core.domain_config import (  # Unified Configuration Classes
    HTTP_BAD_REQUEST,
    HTTP_CREATED,
    HTTP_FORBIDDEN,
    HTTP_INTERNAL_ERROR,
    HTTP_NOT_FOUND,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
    MAXIMUM_PORT_NUMBER,
    MIN_PASSWORD_LENGTH,
    MINIMUM_PORT_NUMBER,
    PERFECT_SUCCESS_PERCENTAGE,
    DatabaseConfiguration,
    EnvironmentType,
    FileSizeMB,
    FlxConfiguration,
    FlxSecretConfiguration,
    LogLevel,
    MeltanoBackend,
    MeltanoConfiguration,
    MonitoringConfiguration,
    NetworkConfiguration,
    NonNegativeInt,
    PercentageValue,
    PortNumber,
    PositiveInt,
    ProcessingMode,
    RetryCount,
    SecurityConfiguration,
    ServiceProtocol,
    ThreadCount,
    TimeoutSeconds,
    get_config,
    reset_config,
)

# SINGLE SOURCE OF TRUTH - with strict validation
# Lazy config initialization to avoid circular dependencies

# ZERO TOLERANCE ENFORCEMENT: Legacy aliases ELIMINATED
# Use get_config() and FlxConfiguration directly
# Backward compatibility maintained only for essential interfaces
Settings = FlxConfiguration
UniversalFlxConfig = FlxConfiguration
get_global_config = get_config
reset_global_config = reset_config


# Legacy global instance - lazy initialization
def _get_legacy_settings() -> object:
    return get_config()


settings = _get_legacy_settings()  # Legacy global instance

__all__ = [
    "HTTP_BAD_REQUEST",
    "HTTP_CREATED",
    "HTTP_FORBIDDEN",
    "HTTP_INTERNAL_ERROR",
    "HTTP_NOT_FOUND",
    # Constants
    "HTTP_OK",
    "HTTP_UNAUTHORIZED",
    "MAXIMUM_PORT_NUMBER",
    "MINIMUM_PORT_NUMBER",
    "MIN_PASSWORD_LENGTH",
    "PERFECT_SUCCESS_PERCENTAGE",
    "DatabaseConfiguration",
    # Type System
    "EnvironmentType",
    "FileSizeMB",
    # Main Configuration
    "FlxConfiguration",
    "FlxSecretConfiguration",
    "LogLevel",
    "MeltanoBackend",
    "MeltanoConfiguration",
    "MonitoringConfiguration",
    # Domain Components
    "NetworkConfiguration",
    "NonNegativeInt",
    "PercentageValue",
    "PortNumber",
    "PositiveInt",
    "ProcessingMode",
    "RetryCount",
    "SecurityConfiguration",
    "ServiceProtocol",
    # Legacy Compatibility - with strict validation
    "Settings",
    "ThreadCount",
    "TimeoutSeconds",
    "UniversalFlxConfig",
    "get_config",
    "get_global_config",
    "reset_config",
    "reset_global_config",
    "settings",
]
