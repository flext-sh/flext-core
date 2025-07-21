"""Domain constants for FLEXT framework with modern Python 3.13 type system.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

All constants used across FLEXT projects should be defined here using modern
Python 3.13 type aliases and Final constants for maximum type safety.
"""

from __future__ import annotations

from typing import Final
from typing import Literal

# ==============================================================================
# MODERN TYPE DEFINITIONS - PYTHON 3.13 TYPE ALIASES
# ==============================================================================

# Modern type aliases using Python 3.13 syntax
type EnvironmentType = Literal["development", "staging", "production", "test"]
type LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type EntityStatusType = Literal[
    "active",
    "inactive",
    "error",
    "pending",
    "draft",
    "archived",
]
type ResultStatusType = Literal["success", "error", "pending", "timeout", "cancelled"]
type PluginTypeAlias = Literal[
    "tap",
    "target",
    "transform",
    "utility",
    "dbt",
    "orchestrator",
]
type PipelineStatusType = Literal[
    "draft",
    "active",
    "inactive",
    "running",
    "completed",
    "failed",
    "cancelled",
]
type ExecutionStatusType = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timeout",
]

# ==============================================================================
# FRAMEWORK CONSTANTS
# ==============================================================================


class FlextFramework:
    """Core framework constants."""

    NAME: Final = "FLEXT"
    VERSION: Final = "0.7.0"
    PYTHON_VERSION: Final = "3.13"
    DESCRIPTION: Final = "Enterprise Data Integration Framework"
    HOMEPAGE: Final = "https://github.com/flext-sh/flext"
    AUTHOR: Final = "FLEXT Team"
    AUTHOR_EMAIL: Final = "team@flext.sh"
    LICENSE: Final = "MIT"


# ==============================================================================
# ENVIRONMENT CONSTANTS
# ==============================================================================


class Environments:
    """Environment constants."""

    DEVELOPMENT: Final = "development"
    STAGING: Final = "staging"
    PRODUCTION: Final = "production"
    TEST: Final = "test"

    ALL: Final = [DEVELOPMENT, STAGING, PRODUCTION, TEST]
    DEFAULT: Final = DEVELOPMENT


# ==============================================================================
# LOG LEVELS CONSTANTS
# ==============================================================================


class LogLevels:
    """Log level constants."""

    DEBUG: Final = "DEBUG"
    INFO: Final = "INFO"
    WARNING: Final = "WARNING"
    ERROR: Final = "ERROR"
    CRITICAL: Final = "CRITICAL"

    ALL: Final = [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    DEFAULT: Final = INFO


# ==============================================================================
# STATUS CONSTANTS
# ==============================================================================


class EntityStatuses:
    """Entity status constants."""

    ACTIVE: Final = "active"
    INACTIVE: Final = "inactive"
    ERROR: Final = "error"
    PENDING: Final = "pending"
    DRAFT: Final = "draft"
    ARCHIVED: Final = "archived"

    ALL: Final = [ACTIVE, INACTIVE, ERROR, PENDING, DRAFT, ARCHIVED]
    DEFAULT: Final = ACTIVE


class ResultStatuses:
    """Result status constants."""

    SUCCESS: Final = "success"
    ERROR: Final = "error"
    PENDING: Final = "pending"
    TIMEOUT: Final = "timeout"
    CANCELLED: Final = "cancelled"

    ALL: Final = [SUCCESS, ERROR, PENDING, TIMEOUT, CANCELLED]
    DEFAULT: Final = SUCCESS


# ==============================================================================
# PLUGIN CONSTANTS
# ==============================================================================


class PluginTypes:
    """Plugin type constants."""

    TAP: Final = "tap"
    TARGET: Final = "target"
    TRANSFORM: Final = "transform"
    UTILITY: Final = "utility"
    DBT: Final = "dbt"
    ORCHESTRATOR: Final = "orchestrator"

    ALL: Final = [TAP, TARGET, TRANSFORM, UTILITY, DBT, ORCHESTRATOR]
    DEFAULT: Final = UTILITY


# ==============================================================================
# PIPELINE CONSTANTS
# ==============================================================================


class PipelineStatuses:
    """Pipeline status constants."""

    DRAFT: Final = "draft"
    ACTIVE: Final = "active"
    INACTIVE: Final = "inactive"
    RUNNING: Final = "running"
    COMPLETED: Final = "completed"
    FAILED: Final = "failed"
    CANCELLED: Final = "cancelled"

    ALL: Final = [DRAFT, ACTIVE, INACTIVE, RUNNING, COMPLETED, FAILED, CANCELLED]
    DEFAULT: Final = DRAFT


class ExecutionStatuses:
    """Execution status constants."""

    PENDING: Final = "pending"
    RUNNING: Final = "running"
    COMPLETED: Final = "completed"
    FAILED: Final = "failed"
    CANCELLED: Final = "cancelled"
    TIMEOUT: Final = "timeout"

    ALL: Final = [PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT]
    DEFAULT: Final = PENDING


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================


class ConfigDefaults:
    """Default configuration values."""

    # Timeouts
    DEFAULT_TIMEOUT: Final = 30
    DEFAULT_RETRY_COUNT: Final = 3
    DEFAULT_BATCH_SIZE: Final = 1000

    # Limits
    MAX_ENTITY_NAME_LENGTH: Final = 100
    MAX_DESCRIPTION_LENGTH: Final = 500
    MAX_CONFIG_KEY_LENGTH: Final = 100
    MAX_ERROR_MESSAGE_LENGTH: Final = 1000

    # Pagination
    DEFAULT_PAGE_SIZE: Final = 20
    MAX_PAGE_SIZE: Final = 100

    # Environment variables
    ENV_PREFIX: Final = "FLEXT_"
    ENV_DELIMITER: Final = "__"

    # File extensions
    CONFIG_FILE_EXTENSION: Final = ".toml"
    LOG_FILE_EXTENSION: Final = ".log"
    DATA_FILE_EXTENSION: Final = ".json"

    # Encoding
    DEFAULT_ENCODING: Final = "utf-8"

    # Network
    DEFAULT_HTTP_TIMEOUT: Final = 30
    DEFAULT_HTTP_RETRIES: Final = 3

    # Cache
    DEFAULT_CACHE_TTL: Final = 3600  # 1 hour
    DEFAULT_CACHE_SIZE: Final = 1000


# ==============================================================================
# HTTP CONSTANTS
# ==============================================================================


class HTTPStatus:
    """HTTP status code constants."""

    CONTENT_TYPE: Final = "Content-Type"
    AUTHORIZATION: Final = "Authorization"
    ACCEPT: Final = "Accept"
    USER_AGENT: Final = "User-Agent"
    X_API_KEY: Final = "X-API-Key"
    X_REQUEST_ID: Final = "X-Request-Id"
    X_CORRELATION_ID: Final = "X-Correlation-Id"


class MediaTypes:
    """Media type constants."""

    JSON: Final = "application/json"
    XML: Final = "application/xml"
    HTML: Final = "text/html"
    PLAIN_TEXT: Final = "text/plain"
    FORM_URL_ENCODED: Final = "application/x-www-form-urlencoded"
    MULTIPART_FORM_DATA: Final = "multipart/form-data"


# ==============================================================================
# REGEX PATTERNS
# ==============================================================================


class RegexPatterns:
    """Common regex patterns."""

    # Naming conventions
    PROJECT_NAME: Final = r"^[a-zA-Z][a-zA-Z0-9-_]*$"
    ENTITY_NAME: Final = r"^[a-zA-Z][a-zA-Z0-9-_\s]*$"
    CONFIG_KEY: Final = r"^[a-z][a-z0-9_]*$"

    # Versions
    SEMANTIC_VERSION: Final = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"

    # Identifiers
    UUID: Final = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    # URLs
    HTTP_URL: Final = r"^https?://[^\s/$.?#].[^\s]*$"
    REDIS_URL: Final = r"^redis://[^\s/$.?#].[^\s]*$"

    # Email
    EMAIL: Final = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"


# ==============================================================================
# ERROR MESSAGES
# ==============================================================================


class ErrorMessages:
    """Standard error messages."""

    # Validation errors
    REQUIRED_FIELD: Final = "This field is required"
    INVALID_FORMAT: Final = "Invalid format"
    TOO_LONG: Final = "Value is too long"
    TOO_SHORT: Final = "Value is too short"
    INVALID_TYPE: Final = "Invalid type"

    # Entity errors
    ENTITY_NOT_FOUND: Final = "Entity not found"
    ENTITY_ALREADY_EXISTS: Final = "Entity already exists"
    ENTITY_INVALID_STATE: Final = "Entity is in invalid state"

    # Configuration errors
    INVALID_CONFIG: Final = "Invalid configuration"
    MISSING_CONFIG: Final = "Missing configuration"
    CONFIG_PARSE_ERROR: Final = "Configuration parse error"

    # Pipeline errors
    PIPELINE_NOT_FOUND: Final = "Pipeline not found"
    PIPELINE_EXECUTION_FAILED: Final = "Pipeline execution failed"
    PIPELINE_INVALID_CONFIG: Final = "Invalid pipeline configuration"

    # Plugin errors
    PLUGIN_NOT_FOUND: Final = "Plugin not found"
    PLUGIN_EXECUTION_FAILED: Final = "Plugin execution failed"
    PLUGIN_INVALID_CONFIG: Final = "Invalid plugin configuration"

    # Network errors
    CONNECTION_FAILED: Final = "Connection failed"
    TIMEOUT_ERROR: Final = "Operation timed out"
    HTTP_ERROR: Final = "HTTP error"

    # Generic errors
    UNEXPECTED_ERROR: Final = "An unexpected error occurred"
    OPERATION_FAILED: Final = "Operation failed"
    ACCESS_DENIED: Final = "Access denied"


# ==============================================================================
# SUCCESS MESSAGES
# ==============================================================================


class SuccessMessages:
    """Standard success messages."""

    # Generic operations
    OPERATION_SUCCESS: Final = "Operation completed successfully"
    CREATED_SUCCESS: Final = "Created successfully"
    UPDATED_SUCCESS: Final = "Updated successfully"
    DELETED_SUCCESS: Final = "Deleted successfully"

    # Pipeline operations
    PIPELINE_CREATED: Final = "Pipeline created successfully"
    PIPELINE_EXECUTED: Final = "Pipeline executed successfully"
    PIPELINE_STOPPED: Final = "Pipeline stopped successfully"

    # Plugin operations
    PLUGIN_INSTALLED: Final = "Plugin installed successfully"
    PLUGIN_EXECUTED: Final = "Plugin executed successfully"
    PLUGIN_REMOVED: Final = "Plugin removed successfully"

    # Configuration operations
    CONFIG_LOADED: Final = "Configuration loaded successfully"
    CONFIG_SAVED: Final = "Configuration saved successfully"
    CONFIG_VALIDATED: Final = "Configuration validated successfully"


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ConfigDefaults",
    "EntityStatusType",
    "EntityStatuses",
    "EnvironmentType",
    "Environments",
    "ErrorMessages",
    "ExecutionStatusType",
    "ExecutionStatuses",
    "FlextFramework",
    "HTTPStatus",
    "LogLevelType",
    "LogLevels",
    "MediaTypes",
    "PipelineStatusType",
    "PipelineStatuses",
    "PluginTypeAlias",
    "PluginTypes",
    "RegexPatterns",
    "ResultStatusType",
    "ResultStatuses",
    "SuccessMessages",
]
