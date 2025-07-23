"""FLEXT Core - Foundation Library.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Foundational library following software engineering principles
with Python 3.13 and Pydantic V2. Serves as the architectural
base for the FLEXT ecosystem with reusability, type safety,
and reliability.
"""

from __future__ import annotations

# ====================
# CORE MODERN COMPONENTS - ONLY THESE
# ====================
# Configuration Management (pydantic-settings)
from flext_core.config import FlextCoreSettings
from flext_core.config import configure_settings
from flext_core.config import get_settings

# Advanced Constants System
from flext_core.constants import FlextConstants
from flext_core.constants import FlextEnvironment
from flext_core.constants import FlextLogLevel
from flext_core.constants import FlextResultStatus

# Enterprise Dependency Injection (FlextContainer)
from flext_core.container import FlextContainer
from flext_core.container import FlextServiceFactory
from flext_core.container import configure_flext_container
from flext_core.container import get_flext_container

# Domain-Driven Design Building Blocks
from flext_core.domain import FlextAggregateRoot
from flext_core.domain import FlextDomainService
from flext_core.domain import FlextEntity
from flext_core.domain import FlextValueObject

# Unified Enterprise Patterns
from flext_core.patterns import FlextCommand  # Command Pattern
from flext_core.patterns import FlextCommandBus
from flext_core.patterns import FlextCommandHandler
from flext_core.patterns import FlextCommandId
from flext_core.patterns import FlextCommandResult
from flext_core.patterns import FlextEventHandler
from flext_core.patterns import FlextField  # Field System
from flext_core.patterns import FlextFieldId
from flext_core.patterns import FlextFieldMetadata
from flext_core.patterns import FlextFieldType
from flext_core.patterns import FlextFieldValidator
from flext_core.patterns import FlextHandler  # Handler Pattern
from flext_core.patterns import FlextHandlerId  # Type Definitions
from flext_core.patterns import FlextLogContext
from flext_core.patterns import FlextLogger  # Logging System
from flext_core.patterns import FlextLoggerFactory
from flext_core.patterns import FlextLoggerName
from flext_core.patterns import FlextMessageHandler
from flext_core.patterns import FlextRequestHandler
from flext_core.patterns import FlextValidationResult
from flext_core.patterns import FlextValidationRule
from flext_core.patterns import FlextValidator  # Validation System
from flext_core.patterns import FlextValidatorId

# Modern Type System
from flext_core.payload import FlextPayload

# Core Result Pattern
from flext_core.result import FlextResult
from flext_core.types_system import FlextConfigKey
from flext_core.types_system import FlextContextData
from flext_core.types_system import FlextEntityId
from flext_core.types_system import FlextEventType
from flext_core.types_system import FlextIdentifier
from flext_core.types_system import FlextResourceId
from flext_core.types_system import FlextServiceName
from flext_core.types_system import FlextTraceId
from flext_core.types_system import FlextTypedDict
from flext_core.types_system import flext_validate_config_key
from flext_core.types_system import flext_validate_event_type
from flext_core.types_system import flext_validate_identifier
from flext_core.types_system import flext_validate_non_empty_string
from flext_core.types_system import flext_validate_service_name

# Public API - Modern Components Only
__all__ = [
    # Domain-Driven Design
    "FlextAggregateRoot",
    # Unified Enterprise Patterns - Command Pattern
    "FlextCommand",
    "FlextCommandBus",
    "FlextCommandHandler",
    "FlextCommandId",
    "FlextCommandResult",
    "FlextConfigKey",
    # Version and Constants
    "FlextConstants",
    # Enterprise Dependency Injection
    "FlextContainer",
    "FlextContextData",
    # Configuration Management
    "FlextCoreSettings",
    "FlextDomainService",
    "FlextEntity",
    "FlextEntityId",
    "FlextEnvironment",
    "FlextEventHandler",
    "FlextEventType",
    # Unified Enterprise Patterns - Field System
    "FlextField",
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldType",
    "FlextFieldValidator",
    # Unified Enterprise Patterns - Handler Pattern
    "FlextHandler",
    "FlextHandlerId",
    "FlextIdentifier",
    "FlextLogContext",
    "FlextLogLevel",
    # Unified Enterprise Patterns - Logging System
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerName",
    "FlextMessageHandler",
    # Modern Type System
    "FlextPayload",
    "FlextRequestHandler",
    "FlextResourceId",
    # Core Result Pattern
    "FlextResult",
    "FlextResultStatus",
    "FlextServiceFactory",
    "FlextServiceName",
    "FlextTraceId",
    "FlextTypedDict",
    "FlextValidationResult",
    "FlextValidationRule",
    # Unified Enterprise Patterns - Validation System
    "FlextValidator",
    "FlextValidatorId",
    "FlextValueObject",
    "configure_flext_container",
    "configure_settings",
    # Type Validation Functions
    "flext_validate_config_key",
    "flext_validate_event_type",
    "flext_validate_identifier",
    "flext_validate_non_empty_string",
    "flext_validate_service_name",
    "get_flext_container",
    "get_settings",
]

# Library metadata
__version__ = "0.8.0"
__author__ = "FLEXT Contributors"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 FLEXT Contributors"
