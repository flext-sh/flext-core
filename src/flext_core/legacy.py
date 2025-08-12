"""Legacy compatibility layer for deprecated patterns.

⚠️  CRITICAL NOTICE: COMPREHENSIVE DEPRECATED MODULE - REMOVAL IN v2.0.0

This module provides systematic backward compatibility for 100+ deprecated functions
during migration to modern Flext* prefixed architecture. Every export is deprecated.

ORGANIZED LEGACY ARCHITECTURE:
    The 1300+ lines are organized into logical sections for maintainability:

    Section 1: Deprecation Warning System - Centralized warning management
    Section 2: Core Utility Functions - Text, IDs, type conversion (8 functions)
    Section 3: Handler System - CQRS handlers, chains, registries (8 functions)
    Section 4: Configuration System - Database, Redis, Oracle, LDAP (4 configs)
    Section 5: Domain Model System - Business objects (6 model factories)
    Section 6: Event/Message System - Cross-service communication (5 functions)
    Section 7: Validation System - Data validation (2 functions)
    Section 8: Decorator System - Caching, timing, safety (8 decorators)
    Section 9: Logging System - Structured logging (4 functions)
    Section 10: Field System - Form and data fields (3 field types)
    Section 11: Result System - Railway-oriented programming (3 functions)
    Section 12: Performance System - Metrics and monitoring (5 functions)
    Section 13: Constants System - Legacy constants and codes (10+ constants)
    Section 14: Compatibility Classes - Legacy base classes (7 classes)
    Section 15: Compatibility Mixins - Test-compatible mixins (5 mixins)

COMPREHENSIVE MIGRATION STRATEGY:
    Each legacy function provides direct migration path to modern equivalents:

    OLD: from flext_core.legacy import truncate
    NEW: from flext_core import FlextUtilities; FlextUtilities.truncate()

    OLD: from flext_core.legacy import create_base_handler
    NEW: from flext_core import FlextBaseHandler; FlextBaseHandler()

    OLD: from flext_core.legacy import ERROR_CODES
    NEW: from flext_core import FlextConstants; FlextConstants.Errors

REMOVAL TIMELINE:
    - v1.9.x: All functions emit deprecation warnings
    - v2.0.0: Complete module removal

For migration assistance: docs/migration/compatibility-guide.md

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import json as _json
import re
import sys
import warnings
from typing import TYPE_CHECKING, cast

from flext_core.config import (
    FlextConfigFactory,
    FlextDatabaseConfig,
    FlextLDAPConfig,
    FlextOracleConfig,
    FlextRedisConfig,
)
from flext_core.constants import FlextConstants, FlextOperationStatus
from flext_core.decorators import (
    _flext_cache_decorator,  # pyright: ignore[reportPrivateUsage]
    _flext_safe_call_decorator,  # pyright: ignore[reportPrivateUsage]
    _flext_timing_decorator,
    _flext_validate_input_decorator,
)
from flext_core.fields import FlextFieldCore, FlextFields
from flext_core.handlers import (
    FlextAuthorizingHandler,
    FlextBaseHandler,
    FlextEventHandler,
    FlextHandlerChain as _Chain,
    FlextHandlerRegistry,
    FlextMetricsHandler,
    FlextValidatingHandler,
)
from flext_core.loggings import (
    FlextLoggerFactory,
    create_log_context as modern_create_log_context,
    get_logger,
    setup_custom_trace_level,
)
from flext_core.mixins import (
    LegacyCompatibleCacheableMixin,
    LegacyCompatibleCommandMixin,
    LegacyCompatibleComparableMixin,
    LegacyCompatibleDataMixin,
    LegacyCompatibleEntityMixin,
    LegacyCompatibleFullMixin,
    LegacyCompatibleIdentifiableMixin,
    LegacyCompatibleLoggableMixin,
    LegacyCompatibleSerializableMixin,
    LegacyCompatibleServiceMixin,
    LegacyCompatibleTimestampMixin,
    LegacyCompatibleTimingMixin,
    LegacyCompatibleValidatableMixin,
    LegacyCompatibleValueObjectMixin,
)
from flext_core.models import (
    FlextDatabaseModel,
    FlextOperationModel,
    FlextOracleModel,
    FlextServiceModel,
    FlextSingerStreamModel,
)
from flext_core.payload import FlextEvent, FlextMessage
from flext_core.result import FlextResult, FlextResult as _FResult
from flext_core.typings import TAnyDict
from flext_core.utilities import FlextPerformance, FlextUtilities
from flext_core.validation import FlextValidation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from flext_core.protocols import FlextDecoratedFunction

logger = get_logger(__name__)


# Chain function implementation for legacy compatibility
def _chain(*results: _FResult[object]) -> _FResult[list[object]]:
    data: list[object] = []
    for result in results:
        if result.is_failure:
            return _FResult.fail(result.error or "Chain failed")
        if result.data is not None:
            data.append(result.data)
    return _FResult.ok(data)


# =============================================================================
# DEPRECATION WARNING SYSTEM
# =============================================================================


def _emit_legacy_warning(module: str = "legacy") -> None:
    """Emit deprecation warning for legacy imports."""
    warnings.warn(
        f"Importing from flext_core.{module} is deprecated. "
        "Use proper Flext* prefixed classes and methods instead. "
        "This compatibility layer will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def get_legacy_usage_warning() -> str:
    """Get warning message for legacy usage."""
    return (
        "Legacy functions and classes are deprecated. "
        "Use FlextUtilities, FlextHandlers, etc. with proper Flext* prefixes. "
        "All legacy imports will be removed in v2.0.0."
    )


# Emit warning when this module is imported
_emit_legacy_warning("legacy")

# =============================================================================
# LEGACY UTILITY FUNCTIONS - Deprecated helper functions
# =============================================================================


def truncate(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length (DEPRECATED)."""
    _emit_legacy_warning("legacy.truncate")
    return FlextUtilities.truncate(text, max_length)


def generate_id() -> str:
    """Generate unique ID (DEPRECATED)."""
    _emit_legacy_warning("legacy.generate_id")
    return FlextUtilities.generate_id()


def generate_correlation_id() -> str:
    """Generate correlation ID (DEPRECATED)."""
    _emit_legacy_warning("legacy.generate_correlation_id")
    return FlextUtilities.generate_correlation_id()


def generate_uuid() -> str:
    """Generate UUID (DEPRECATED)."""
    _emit_legacy_warning("legacy.generate_uuid")
    return FlextUtilities.generate_uuid()


def generate_iso_timestamp() -> str:
    """Generate ISO timestamp (DEPRECATED)."""
    _emit_legacy_warning("legacy.generate_iso_timestamp")
    return FlextUtilities.generate_iso_timestamp()


def is_not_none(value: object) -> bool:
    """Check if value is not None (DEPRECATED)."""
    _emit_legacy_warning("legacy.is_not_none")
    return FlextUtilities.is_not_none_guard(value)


def safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Safely convert value to integer (DEPRECATED)."""
    _emit_legacy_warning("legacy.safe_int_conversion")
    return FlextUtilities.safe_int_conversion(value, default)


def safe_int_conversion_with_default(value: object, default: int) -> int:
    """Safely convert value to integer with default (DEPRECATED)."""
    _emit_legacy_warning("legacy.safe_int_conversion_with_default")
    return FlextUtilities.safe_int_conversion_with_default(value, default)


# =============================================================================
# LEGACY HANDLER CREATION FUNCTIONS - Deprecated factory functions
# =============================================================================


def create_base_handler(name: str | None = None) -> FlextValidatingHandler:
    """Create base handler (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_base_handler")
    # Return a concrete implementation, not the abstract base
    return FlextValidatingHandler(name or "legacy_handler")


def create_validating_handler(name: str | None = None) -> FlextValidatingHandler:
    """Create validating handler (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_validating_handler")
    return FlextValidatingHandler(name or "legacy_validating_handler")


def create_authorizing_handler(
    name: str | None = None,
    *,
    _auth_required: bool = True,
) -> FlextAuthorizingHandler:
    """Create authorizing handler (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_authorizing_handler")
    return FlextAuthorizingHandler(name or "legacy_auth_handler")


def create_event_handler(name: str | None = None) -> FlextEventHandler:
    """Create event handler (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_event_handler")
    return FlextEventHandler(name or "legacy_event_handler")


def create_metrics_handler(name: str | None = None) -> FlextMetricsHandler:
    """Create metrics handler (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_metrics_handler")
    return FlextMetricsHandler(name or "legacy_metrics_handler")


# =============================================================================
# LEGACY CONFIG CREATION FUNCTIONS - Deprecated config factory aliases
# =============================================================================


def create_database_config(
    *,
    host: str = "localhost",
    port: int = 5432,
    username: str = "postgres",
    password: str | None = None,
    database: str = "flext",
    **kwargs: object,
) -> FlextDatabaseConfig:
    """Create database config (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_database_config")
    return FlextConfigFactory.create_database_config(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        **kwargs,
    )


def create_redis_config(
    *,
    host: str = "localhost",
    port: int = 6379,
    password: str | None = None,
    database: int = 0,
    **kwargs: object,
) -> FlextRedisConfig:
    """Create redis config (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_redis_config")
    return FlextConfigFactory.create_redis_config(
        host=host,
        port=port,
        password=password,
        database=database,
        **kwargs,
    )


def create_oracle_config(
    *,
    host: str = "localhost",
    username: str = "oracle",
    password: str | None = None,
    service_name: str | None = None,
    **kwargs: object,
) -> FlextOracleConfig:
    """Create oracle config (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_oracle_config")
    return FlextConfigFactory.create_oracle_config(
        host=host,
        username=username,
        password=password,
        service_name=service_name,
        **kwargs,
    )


def create_ldap_config(
    *,
    host: str = "localhost",
    port: int = 389,
    base_dn: str = "dc=example,dc=com",
    bind_dn: str | None = None,
    bind_password: str | None = None,
    **kwargs: object,
) -> FlextLDAPConfig:
    """Create LDAP config (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_ldap_config")
    return FlextConfigFactory.create_ldap_config(
        host=host,
        port=port,
        base_dn=base_dn,
        bind_dn=bind_dn,
        bind_password=bind_password,
        **kwargs,
    )


# =============================================================================
# LEGACY CLASS ALIASES - Deprecated class names without Flext prefix
# =============================================================================


# Utility class aliases
# Console removed - exists in utilities.py


class LegacyBaseEntry:
    """Base entry class (DEPRECATED - use FlextValueObject)."""

    def __init__(self, **_kwargs: object) -> None:
        """Initialize base entry (deprecated)."""
        _emit_legacy_warning("legacy.BaseEntry")


class LegacyBaseProcessor:
    """Base processor class (DEPRECATED - use proper Flext classes)."""

    def __init__(self) -> None:
        """Initialize base processor (deprecated)."""
        _emit_legacy_warning("legacy.BaseProcessor")


# =============================================================================
# LEGACY CONTAINER FUNCTIONS - Deprecated container helpers
# =============================================================================


def get_global_registry() -> object:
    """Get global handler registry (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_global_registry")
    return FlextHandlerRegistry()


def reset_global_registry() -> None:
    """Reset global handler registry (DEPRECATED)."""
    _emit_legacy_warning("legacy.reset_global_registry")


def create_handler_registry() -> object:
    """Create handler registry (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_handler_registry")
    return FlextHandlerRegistry()


def create_handler_chain(
    handlers: Iterable[FlextBaseHandler] | None = None,
    name: str | None = None,
) -> object:
    """Create handler chain (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_handler_chain")
    chain = _Chain(handlers=list(handlers) if handlers else None)
    # 'name' kept for signature compatibility; no effect in new API
    _ = name
    return chain


# =============================================================================
# LEGACY RESULT FUNCTIONS - Deprecated result helpers
# =============================================================================


def chain(*results: _FResult[object]) -> _FResult[list[object]]:
    """Chain multiple results (DEPRECATED - use FlextResult methods)."""
    _emit_legacy_warning("legacy.chain")
    return _chain(*results)


def compose(*results: _FResult[object]) -> _FResult[list[object]]:
    """Compose multiple results (DEPRECATED - use FlextResult methods)."""
    _emit_legacy_warning("legacy.compose")
    return chain(*results)


def safe_call(
    func: Callable[[], object] | Callable[[object], object],
) -> _FResult[object]:
    """Safely call a function (DEPRECATED - use FlextUtilities.safe_call)."""
    _emit_legacy_warning("legacy.safe_call")
    return FlextUtilities.safe_call(func)


# =============================================================================
# LEGACY UTILITIES FUNCTIONS - Deprecated utility helpers from utilities.py
# =============================================================================


def flext_safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Legacy alias for safe_int_conversion (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_safe_int_conversion")
    return FlextUtilities.safe_int_conversion(value, default)


def flext_track_performance(category: str) -> object:
    """Track performance (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_track_performance")
    return FlextPerformance.track_performance(category)


def flext_get_performance_metrics() -> dict[str, dict[str, object]]:
    """Get performance metrics (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_get_performance_metrics")
    return FlextPerformance.get_performance_metrics()


def flext_clear_performance_metrics() -> None:
    """Clear performance metrics (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_clear_performance_metrics")
    FlextPerformance.clear_performance_metrics()


def flext_record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool = True,
) -> None:
    """Record performance (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_record_performance")
    FlextPerformance.record_performance(
        category,
        function_name,
        execution_time,
        _success=success,
    )


# =============================================================================
# LEGACY MODEL ALIASES - Deprecated model classes
# =============================================================================

# Legacy aliases that were in models.py and base_handlers.py
FlextHandlers = FlextBaseHandler  # Legacy alias (moved from handlers.py)
FlextCommandHandler = FlextBaseHandler  # Legacy alias
FlextQueryHandler = FlextBaseHandler  # Legacy alias
FlextBaseModel = object  # Generic base model alias

# =============================================================================
# FLEXT_TYPES.PY COMPATIBILITY - Complete migration from removed module
# =============================================================================

# =============================================================================
# LEGACY CONSTANTS - Deprecated constants from constants.py
# =============================================================================


# Legacy ERROR_CODES dict for backward compatibility
def get_error_codes() -> dict[str, str]:
    """Get legacy ERROR_CODES dict (DEPRECATED)."""
    _emit_legacy_warning("legacy.ERROR_CODES")
    return {
        code_name: getattr(FlextConstants.Errors, code_name)
        for code_name in dir(FlextConstants.Errors)
        if not code_name.startswith("_")
    }


# Legacy MESSAGES dict for backward compatibility
def get_messages() -> object:
    """Get legacy MESSAGES dict (DEPRECATED)."""
    _emit_legacy_warning("legacy.MESSAGES")
    return {
        message_name: getattr(FlextConstants.Messages, message_name)
        for message_name in dir(FlextConstants.Messages)
        if not message_name.startswith("_")
    }


# Legacy STATUS_CODES dict for backward compatibility
def get_status_codes() -> object:
    """Get legacy STATUS_CODES dict (DEPRECATED)."""
    _emit_legacy_warning("legacy.STATUS_CODES")
    return {
        status_name: getattr(FlextConstants.Status, status_name)
        for status_name in dir(FlextConstants.Status)
        if not status_name.startswith("_")
    }


# Legacy VALIDATION_RULES for backward compatibility
VALIDATION_RULES = {
    "REQUIRED": "REQUIRED",
    "OPTIONAL": "OPTIONAL",
    "NULLABLE": "NULLABLE",
    "NON_EMPTY": "NON_EMPTY",
}


# Initialize dict constants
ERROR_CODES: dict[str, str] | None = None
MESSAGES: dict[str, str] | None = None
STATUS_CODES: dict[str, str] | None = None


# Initialize legacy dicts
def _init_legacy_dicts() -> None:
    """Initialize legacy dict constants."""
    global ERROR_CODES, MESSAGES, STATUS_CODES  # noqa: PLW0603

    _emit_legacy_warning("legacy.dict_c onstants")
    ERROR_CODES = cast("dict[str, str] | None", get_error_codes())
    MESSAGES = cast("dict[str, str] | None", get_messages())
    STATUS_CODES = cast("dict[str, str] | None", get_status_codes())


_init_legacy_dicts()


def get_default_timeout() -> int:
    """Get default timeout (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_default_timeout")
    return FlextConstants.Defaults.TIMEOUT


def get_default_retries() -> int:
    """Get default retries (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_default_retries")
    return FlextConstants.Defaults.MAX_RETRIES


def get_default_page_size() -> int:
    """Get default page size (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_default_page_size")
    return FlextConstants.Defaults.PAGE_SIZE


# Legacy individual constants for backward compatibility
def _create_legacy_constant_getter(attr_name: str, path: str) -> Callable[[], object]:
    """Create a getter function for legacy constants."""

    def get_constant() -> object:
        _emit_legacy_warning(f"legacy.{attr_name}")

        # Navigate the nested path
        obj: object = FlextConstants
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    return get_constant


# Legacy individual constants as simple variables (lazy evaluation)
_legacy_constants = {
    "DEFAULT_TIMEOUT": ("Defaults", "TIMEOUT"),
    "DEFAULT_RETRIES": ("Defaults", "MAX_RETRIES"),
    "DEFAULT_PAGE_SIZE": ("Defaults", "PAGE_SIZE"),
    "VERSION": ("Core", "VERSION"),
    "NAME": ("Core", "NAME"),
    "EMAIL_PATTERN": ("Patterns", "EMAIL_PATTERN"),
    "UUID_PATTERN": ("Patterns", "UUID_PATTERN"),
    "URL_PATTERN": ("Patterns", "URL_PATTERN"),
    "IDENTIFIER_PATTERN": ("Patterns", "IDENTIFIER_PATTERN"),
    "SERVICE_NAME_PATTERN": ("Patterns", "SERVICE_NAME_PATTERN"),
}

# Simple direct constant assignments (with lazy loading)
DEFAULT_TIMEOUT: int | None = None
DEFAULT_RETRIES: int | None = None
DEFAULT_PAGE_SIZE: int | None = None
VERSION: str | None = None
NAME: str | None = None
EMAIL_PATTERN: str | None = None
UUID_PATTERN: str | None = None
URL_PATTERN: str | None = None
IDENTIFIER_PATTERN: str | None = None
SERVICE_NAME_PATTERN: str | None = None


def _get_legacy_constant_value(const_name: str) -> object | None:
    """Get legacy constant value with deprecation warning."""
    _emit_legacy_warning(f"legacy.{const_name}")

    if const_name in _legacy_constants:
        path_parts = _legacy_constants[const_name]
        obj = FlextConstants
        for part in path_parts:
            obj = getattr(obj, part)
        return obj
    return None


# Initialize legacy constants on first import
def _init_legacy_constants() -> None:
    """Initialize legacy constants."""
    global DEFAULT_TIMEOUT, DEFAULT_RETRIES, DEFAULT_PAGE_SIZE, VERSION, NAME, EMAIL_PATTERN, UUID_PATTERN, URL_PATTERN, IDENTIFIER_PATTERN, SERVICE_NAME_PATTERN  # noqa: PLW0603

    # Emit warning once for module import
    _emit_legacy_warning("legacy.constants")

    DEFAULT_TIMEOUT = FlextConstants.Defaults.TIMEOUT
    DEFAULT_RETRIES = FlextConstants.Defaults.MAX_RETRIES
    DEFAULT_PAGE_SIZE = FlextConstants.Defaults.PAGE_SIZE
    VERSION = FlextConstants.Core.VERSION
    NAME = FlextConstants.Core.NAME
    EMAIL_PATTERN = FlextConstants.Patterns.EMAIL_PATTERN
    UUID_PATTERN = FlextConstants.Patterns.UUID_PATTERN
    URL_PATTERN = FlextConstants.Patterns.URL_PATTERN
    IDENTIFIER_PATTERN = FlextConstants.Patterns.IDENTIFIER_PATTERN
    SERVICE_NAME_PATTERN = FlextConstants.Patterns.SERVICE_NAME_PATTERN


# Initialize on module import
_init_legacy_constants()


# =============================================================================
# LEGACY MODEL FUNCTIONS - Deprecated model factory functions from models.py
# =============================================================================


def create_database_model(**kwargs: object) -> FlextDatabaseModel:
    """Create database model (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_database_model")
    return FlextDatabaseModel(**kwargs)  # type: ignore[arg-type]


def create_oracle_model(**kwargs: object) -> FlextOracleModel:
    """Create oracle model (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_oracle_model")
    return FlextOracleModel(**kwargs)  # type: ignore[arg-type]


def create_operation_model(
    operation_id: str,
    operation_type: str,
    **kwargs: object,
) -> FlextOperationModel:
    """Create operation model (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_operation_model")

    # Use default pending status
    status_enum = FlextOperationStatus.PENDING

    return FlextOperationModel(
        operation_id=operation_id,
        operation_type=operation_type,
        status=status_enum,
        **kwargs,  # type: ignore[arg-type]
    )


def create_service_model(
    service_name: str,
    service_type: str,
    service_port: int,
    **kwargs: object,
) -> FlextServiceModel:
    """Create service model (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_service_model")
    # NOTE: service_type is mapped to service_id for backward compatibility
    return FlextServiceModel(
        service_name=service_name,
        service_id=service_type,
        port=service_port,
        **kwargs,  # type: ignore[arg-type]
    )


def create_singer_stream_model(
    stream_name: str,
    schema_dict: dict[str, object],
    _key_properties: list[str] | None = None,
    **_kwargs: object,  # Renamed to indicate intentional non-use
) -> object:
    """Create singer stream model (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_singer_stream_model")
    # Only pass supported fields to avoid type issues in legacy path
    return FlextSingerStreamModel(
        stream_name=stream_name,
        schema_definition=schema_dict,
    )


def validate_all_models(*models: object) -> FlextResult[object]:
    """Validate all models (DEPRECATED)."""
    _emit_legacy_warning("legacy.validate_all_models")

    for model in models:
        validate_method = getattr(model, "validate_business_rules", None)
        if callable(validate_method):
            result: FlextResult[object] = validate_method()
            if result.is_failure:
                return result
    return FlextResult.ok(None)


def model_to_dict_safe(model: object) -> dict[str, object]:
    """Convert model to dict safely (DEPRECATED)."""
    _emit_legacy_warning("legacy.model_to_dict_safe")

    try:
        to_dict_method = getattr(model, "to_dict", None)
        return to_dict_method() if callable(to_dict_method) else {}
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to serialize model {type(model).__name__} to dict: {e}")
        return {}


# =============================================================================
# LEGACY PAYLOAD FUNCTIONS - Deprecated payload helpers from payload.py
# =============================================================================


def serialize_payload_for_go_bridge(payload: object) -> FlextResult[object]:
    """Serialize payload for Go bridge (DEPRECATED)."""
    _emit_legacy_warning("legacy.serialize_payload_for_go_bridge")

    try:
        to_json_method = getattr(payload, "to_json", None)
        if callable(to_json_method):
            return FlextResult.ok(to_json_method())
        return FlextResult.ok(str(payload))
    except Exception as e:
        return FlextResult.fail(f"Serialization failed: {e}")


def deserialize_payload_from_go_bridge(json_str: str) -> FlextResult[object]:
    """Deserialize payload from Go bridge (DEPRECATED)."""
    _emit_legacy_warning("legacy.deserialize_payload_from_go_bridge")

    try:
        return FlextResult.ok(json.loads(json_str))
    except Exception as e:
        return FlextResult.fail(f"Deserialization failed: {e}")


def create_cross_service_message(
    message_type: str,
    data: dict[str, object] | str,
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextMessage:
    """Create cross-service message (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_cross_service_message")

    # Create metadata with message_type and correlation_id
    metadata = {
        "message_type": message_type,
        "correlation_id": correlation_id or FlextUtilities.generate_correlation_id(),
        **kwargs,
    }
    return FlextMessage(data=str(data), metadata=metadata)


def create_cross_service_event(
    event_type: str,
    event_data: dict[str, object],
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextEvent:
    """Create cross-service event (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_cross_service_event")

    # Create metadata with event_type and correlation_id
    metadata = {
        "event_type": event_type,
        "correlation_id": correlation_id or FlextUtilities.generate_correlation_id(),
        **kwargs,
    }
    return FlextEvent(data=event_data, metadata=metadata)


def validate_cross_service_protocol(payload: object) -> FlextResult[object]:
    """Validate cross-service protocol (DEPRECATED)."""
    _emit_legacy_warning("legacy.validate_cross_service_protocol")

    # Accept either objects with attributes or JSON strings/dicts
    try:
        if isinstance(payload, str):
            parsed = _json.loads(payload)
            if not isinstance(parsed, dict):
                return FlextResult.fail("Invalid protocol format")
            # Parsed payload must be a dict; treat as valid
            return FlextResult.ok(None)
        if isinstance(payload, dict):
            return FlextResult.ok(None)
        message_type = getattr(payload, "message_type", None)
        event_type = getattr(payload, "event_type", None)
        if message_type is not None or event_type is not None:
            return FlextResult.ok(None)
        return FlextResult.fail("Payload missing type information")
    except Exception as e:
        return FlextResult.fail(f"Protocol validation failed: {e}")


def get_serialization_metrics(payload: object | None = None) -> dict[str, object]:
    """Get serialization metrics (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_serialization_metrics")

    data_obj = getattr(payload, "data", None)
    metrics = {
        "type": type(payload).__name__,
        "size": len(str(payload)) if payload is not None else 0,
        "compressed": False,
    }
    # Provide keys expected by tests via payload wrapper
    metrics.setdefault("payload_type", metrics["type"])
    metrics.setdefault("data_type", type(data_obj).__name__)
    return metrics


# =============================================================================
# LEGACY UTILITIES CONSTANTS AND CLASSES - Deprecated patterns from utilities.py
# =============================================================================


# Legacy constants for backward compatibility
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Legacy performance metrics dictionary
PERFORMANCE_METRICS: dict[str, float] = {}


class LegacyConsole:
    """Legacy console class (DEPRECATED - use FlextUtilities.Console)."""

    def __init__(self) -> None:
        """Initialize legacy console (deprecated)."""
        _emit_legacy_warning("legacy.LegacyConsole")

    def print(self, *args: object, **kwargs: object) -> None:
        """Print to console (DEPRECATED)."""
        _emit_legacy_warning("legacy.Console.print")

        text_parts = []
        for arg in args:
            text = str(arg)
            # Remove rich markup tags for plain text output
            clean_text = re.sub(r"\[/?[^\]]*\]", "", text)
            text_parts.append(clean_text)

        sep = str(kwargs.get("sep", " "))
        end = str(kwargs.get("end", "\n"))
        output = sep.join(text_parts) + end
        sys.stdout.write(output)
        sys.stdout.flush()

    def log(self, *args: object, **kwargs: object) -> None:
        """Log to console (DEPRECATED)."""
        _emit_legacy_warning("legacy.Console.log")
        self.print(*args, **kwargs)


# Legacy protocol for decorated functions
class DecoratedFunction:
    """Legacy decorated function protocol (DEPRECATED)."""

    def __init__(self) -> None:
        """Initialize decorated function (deprecated)."""
        _emit_legacy_warning("legacy.DecoratedFunction")


# =============================================================================
# LEGACY VALIDATION FUNCTIONS - Deprecated validation helpers from validation.py
# =============================================================================


def validate_smart(value: object, **_context: object) -> object:
    """Validate value with type detection (DEPRECATED)."""
    _emit_legacy_warning("legacy.validate_smart")

    return FlextValidation.validate(value)


def is_valid_data(value: object) -> bool:
    """Check if value is valid (DEPRECATED)."""
    _emit_legacy_warning("legacy.is_valid_data")

    return FlextValidation.validate(value).is_success


# =============================================================================
# LEGACY DECORATOR FUNCTIONS - Deprecated decorator helpers from decorators.py
# =============================================================================


def create_cache_decorator(
    max_size: int = 128,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Create cache decorator (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_cache_decorator")
    return _flext_cache_decorator(max_size)


def create_safe_decorator(
    error_handler: Callable[[Exception], str] | None = None,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Create safe call decorator (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_safe_decorator")
    return _flext_safe_call_decorator(error_handler)


def create_timing_decorator() -> Callable[
    [FlextDecoratedFunction],
    FlextDecoratedFunction,
]:
    """Create timing decorator (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_timing_decorator")
    return _flext_timing_decorator


def create_validation_decorator(
    validator: Callable[[object], bool],
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Create input validation decorator (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_validation_decorator")
    return _flext_validate_input_decorator(validator)


def safe_call_decorator(
    error_handler: Callable[[Exception], str] | None = None,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Safe call decorator (DEPRECATED - alias)."""
    _emit_legacy_warning("legacy.safe_call_decorator")
    return create_safe_decorator(error_handler)


def timing_decorator(func: FlextDecoratedFunction) -> FlextDecoratedFunction:
    """Time function execution (DEPRECATED - alias)."""
    _emit_legacy_warning("legacy.timing_decorator")
    return create_timing_decorator()(func)


def cache_decorator(
    max_size: int = 128,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Cache decorator (DEPRECATED - alias)."""
    _emit_legacy_warning("legacy.cache_decorator")
    return create_cache_decorator(max_size)


def validation_decorator(
    validator: Callable[[object], bool],
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Validate function input (DEPRECATED - alias)."""
    _emit_legacy_warning("legacy.validation_decorator")
    return create_validation_decorator(validator)


# =============================================================================
# LEGACY LOGGING FUNCTIONS - Deprecated logging helpers from loggings.py
# =============================================================================


def setup_legacy_trace_level() -> None:
    """Set up custom trace level (DEPRECATED)."""
    _emit_legacy_warning("legacy.setup_legacy_trace_level")

    return setup_custom_trace_level()


def get_legacy_logger(name: str, level: str = "INFO") -> object:
    """Get logger (DEPRECATED)."""
    _emit_legacy_warning("legacy.get_legacy_logger")
    return FlextLoggerFactory.get_logger(name, level)


def create_log_context(
    correlation_id: str | None = None,
    user_id: str | None = None,
    operation: str | None = None,
    **kwargs: object,
) -> object:
    """Create log context (DEPRECATED)."""
    _emit_legacy_warning("legacy.create_log_context")
    # Convert positional args to keyword args for modern function
    context = kwargs.copy()
    if correlation_id is not None:
        context["correlation_id"] = correlation_id
    if user_id is not None:
        context["user_id"] = user_id
    if operation is not None:
        context["operation"] = operation
    return modern_create_log_context(logger=None, **context)


def flext_get_logger(name: str) -> object:
    """Flext get logger (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_get_logger")
    return get_logger(name)


# =============================================================================
# LEGACY FIELD FUNCTIONS - Deprecated field helpers from fields.py
# =============================================================================


def flext_create_string_field(
    name: str,
    *,
    required: bool = True,
    min_length: int = 0,
    max_length: int | None = None,
    description: str = "",
    **kwargs: object,
) -> FlextFieldCore:
    """Create string field (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_create_string_field")
    return FlextFields.create_string_field(
        field_id=f"{name}_id",
        field_name=name,
        required=required,
        min_length=min_length,
        max_length=max_length,
        description=description,
        **kwargs,
    )


def flext_create_integer_field(
    name: str,
    *,
    required: bool = True,
    min_value: int | None = None,
    max_value: int | None = None,
    description: str = "",
    **kwargs: object,
) -> FlextFieldCore:
    """Create integer field (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_create_integer_field")
    return FlextFields.create_integer_field(
        field_id=f"{name}_id",
        field_name=name,
        required=required,
        min_value=min_value,
        max_value=max_value,
        description=description,
        **kwargs,
    )


def flext_create_boolean_field(
    name: str,
    *,
    required: bool = True,
    description: str = "",
    **kwargs: object,
) -> FlextFieldCore:
    """Create boolean field (DEPRECATED)."""
    _emit_legacy_warning("legacy.flext_create_boolean_field")
    return FlextFields.create_boolean_field(
        field_id=f"{name}_id",
        field_name=name,
        required=required,
        description=description,
        **kwargs,
    )


# =============================================================================
# LEGACY RESULT FUNCTIONS - Deprecated functions from result.py
# =============================================================================

# NOTE: Redefined functions removed - using existing chain/compose/safe_call above


# =============================================================================
# MIXIN COMPATIBILITY LAYER - Mixins imported from mixin_compat.py
# =============================================================================

# All LegacyCompatible* mixins are imported at the top of the file
# from flext_core.mixin_compat to eliminate code duplication

# =============================================================================
# DEPRECATED EXPORTS - Legacy compatibility only
# =============================================================================

# ⚠️ WARNING: All exports in this module are DEPRECATED
# They will be REMOVED in flext-core v2.0.0
# Use proper Flext* prefixed imports from flext_core main module instead

__all__: list[str] = [
    "BYTES_PER_GB",
    # === LEGACY UTILITIES CONSTANTS AND CLASSES ===
    "BYTES_PER_KB",
    "BYTES_PER_MB",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "EMAIL_PATTERN",
    # === LEGACY CONSTANTS ===
    "ERROR_CODES",
    "IDENTIFIER_PATTERN",
    "MESSAGES",
    "NAME",
    "PERFORMANCE_METRICS",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "SERVICE_NAME_PATTERN",
    "STATUS_CODES",
    "URL_PATTERN",
    "UUID_PATTERN",
    "VALIDATION_RULES",
    "VERSION",
    "DecoratedFunction",
    "LegacyBaseEntry",
    "LegacyBaseProcessor",
    # === LEGACY COMPATIBLE MIXINS ===
    "LegacyCompatibleCacheableMixin",
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "LegacyCompatibleValueObjectMixin",
    # === FROM FLEXT_TYPES.PY ===
    # === LEGACY CLASS ALIASES ===
    "LegacyConsole",
    # === LEGACY SINGER BASE CLASSES ===
    "SingerBase",
    "SingerTap",
    "SingerTarget",
    # All T* aliases from typings.py (imported via *)
    "TAnyDict",
    "cache_decorator",
    # === LEGACY RESULT FUNCTIONS ===
    "chain",
    "compose",
    "create_authorizing_handler",
    # === LEGACY HANDLER CREATION FUNCTIONS ===
    "create_base_handler",
    # === LEGACY DECORATOR FUNCTIONS ===
    "create_cache_decorator",
    "create_cross_service_event",
    "create_cross_service_message",
    # === LEGACY CONFIG CREATION FUNCTIONS ===
    "create_database_config",
    # === LEGACY MODEL FUNCTIONS ===
    "create_database_model",
    "create_event_handler",
    "create_ldap_config",
    "create_log_context",
    "create_metrics_handler",
    "create_operation_model",
    "create_oracle_config",
    "create_oracle_model",
    "create_redis_config",
    "create_safe_decorator",
    "create_service_model",
    "create_singer_stream_model",
    "create_timing_decorator",
    "create_validating_handler",
    "create_validation_decorator",
    "deserialize_payload_from_go_bridge",
    "flext_clear_performance_metrics",
    "flext_create_boolean_field",
    "flext_create_integer_field",
    # === LEGACY FIELD FUNCTIONS ===
    "flext_create_string_field",
    "flext_get_logger",
    "flext_get_performance_metrics",
    "flext_record_performance",
    # === LEGACY UTILITIES FUNCTIONS ===
    "flext_safe_int_conversion",
    "flext_track_performance",
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "get_legacy_logger",
    # === UTILITY FUNCTIONS ===
    "get_legacy_usage_warning",
    "get_serialization_metrics",
    "is_not_none",
    "is_valid_data",
    "model_to_dict_safe",
    "safe_call",
    "safe_call_decorator",
    "safe_int_conversion",
    "safe_int_conversion_with_default",
    # === LEGACY PAYLOAD FUNCTIONS ===
    "serialize_payload_for_go_bridge",
    # === LEGACY LOGGING FUNCTIONS ===
    "setup_legacy_trace_level",
    "timing_decorator",
    # === LEGACY UTILITY FUNCTIONS ===
    "truncate",
    "validate_all_models",
    "validate_cross_service_protocol",
    # === LEGACY VALIDATION FUNCTIONS ===
    "validate_smart",
    "validation_decorator",
]

# =============================================================================
# LEGACY SINGER BASE CLASSES - From singer_base.py
# =============================================================================


class SingerBase:
    """Base class for Singer-related functionality (DEPRECATED)."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize Singer base (deprecated)."""
        _emit_legacy_warning("legacy.SingerBase")
        # Simplified initialization for legacy compatibility
        self._kwargs = kwargs

    def extract(self) -> _FResult[dict[str, object]]:
        """Extract data using Singer patterns (DEPRECATED)."""
        _emit_legacy_warning("legacy.SingerBase.extract")
        return _FResult.ok({})

    def load(self, _data: dict[str, object]) -> _FResult[bool]:
        """Load data using Singer patterns (DEPRECATED)."""
        _emit_legacy_warning("legacy.SingerBase.load")
        success = True
        return _FResult.ok(success)


class SingerTap(SingerBase):
    """Base class for Singer taps/extractors (DEPRECATED)."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize Singer tap (deprecated)."""
        _emit_legacy_warning("legacy.SingerTap")
        super().__init__(**kwargs)


class SingerTarget(SingerBase):
    """Base class for Singer targets/loaders (DEPRECATED)."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize Singer target (deprecated)."""
        _emit_legacy_warning("legacy.SingerTarget")
        super().__init__(**kwargs)


# =============================================================================
# MODULE NOTICE
# =============================================================================

# IMPORTANT: This module is TEMPORARY and will be REMOVED in v2.0.0
# All new code should import directly from flext_core.typings
#
# Migration guide:
# 1. Replace: from flext_core.typings import X
#    With: from flext_core.typings import X
#
# 2. Replace: from flext_core.legacy import X
#    With: from flext_core.typings import X
#
# 3. For hierarchical types, use:
#    from flext_core.typings import FlextTypes
#    my_type: FlextTypes.Core.EntityId = "123"
