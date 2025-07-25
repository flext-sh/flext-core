"""FLEXT Core - Foundation Library.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Foundational library following software engineering principles
with Python 3.13 and Pydantic V2. Serves as the architectural
base for the FLEXT ecosystem with reusability, type safety,
and reliability.
"""

from __future__ import annotations

# Builder and Factory Patterns
from flext_core.builders import FlextBuilder
from flext_core.builders import FlextConfigBuilder
from flext_core.builders import FlextFactory
from flext_core.builders import FlextServiceBuilder
from flext_core.builders import FlextSingletonFactory
from flext_core.builders import build_config
from flext_core.builders import build_service_config
from flext_core.builders import create_factory
from flext_core.builders import create_singleton_factory

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
from flext_core.exceptions import FlextAuthenticationError
from flext_core.exceptions import FlextConfigError
from flext_core.exceptions import FlextConfigurationError
from flext_core.exceptions import FlextConnectionError
from flext_core.exceptions import FlextCriticalError

# Exception Hierarchy (FLEXT Exception System - Only Available)
from flext_core.exceptions import FlextError
from flext_core.exceptions import FlextMigrationError
from flext_core.exceptions import FlextProcessingError
from flext_core.exceptions import FlextSchemaError
from flext_core.exceptions import FlextTimeoutError
from flext_core.exceptions import FlextValidationError

# Helper Functions and Classes (reduce boilerplate)
from flext_core.helpers import ContainerMixin
from flext_core.helpers import LoggerMixin
from flext_core.helpers import Pipeline
from flext_core.helpers import QuickEntity
from flext_core.helpers import QuickValueObject
from flext_core.helpers import ValidatorMixin
from flext_core.helpers import cache_result
from flext_core.helpers import chain
from flext_core.helpers import fail
from flext_core.helpers import from_dict
from flext_core.helpers import get_service
from flext_core.helpers import inject
from flext_core.helpers import log_calls
from flext_core.helpers import ok
from flext_core.helpers import pipeline
from flext_core.helpers import register
from flext_core.helpers import retry
from flext_core.helpers import safe
from flext_core.helpers import to_dict
from flext_core.helpers import validate_email
from flext_core.helpers import validate_non_empty
from flext_core.helpers import validate_required

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
from flext_core.patterns import get_logger  # Helper for quick logger creation

# Boilerplate Reduction - Decorators
from flext_core.patterns.decorators import flext_async_safe
from flext_core.patterns.decorators import flext_cache_result
from flext_core.patterns.decorators import flext_require_non_none
from flext_core.patterns.decorators import flext_retry
from flext_core.patterns.decorators import flext_robust
from flext_core.patterns.decorators import flext_safe_result
from flext_core.patterns.decorators import flext_timed
from flext_core.patterns.decorators import flext_validate_result

# Boilerplate Reduction - Dict Helpers
from flext_core.patterns.dict_helpers import FlextDict
from flext_core.patterns.dict_helpers import flatten_dict
from flext_core.patterns.dict_helpers import merge_dicts
from flext_core.patterns.dict_helpers import omit_keys
from flext_core.patterns.dict_helpers import pick_keys
from flext_core.patterns.dict_helpers import safe_get
from flext_core.patterns.dict_helpers import safe_set_nested

# Boilerplate Reduction - Mixins
from flext_core.patterns.mixins import FlextAuditMixin
from flext_core.patterns.mixins import FlextCacheableMixin
from flext_core.patterns.mixins import FlextComparableMixin
from flext_core.patterns.mixins import FlextEntityMixin
from flext_core.patterns.mixins import FlextMetadataMixin
from flext_core.patterns.mixins import FlextSerializationMixin
from flext_core.patterns.mixins import FlextTimestampMixin
from flext_core.patterns.mixins import FlextValidationMixin
from flext_core.patterns.mixins import FlextValueObjectMixin

# Note: 100+ more types available - see patterns.typedefs for complete list
# Boilerplate Reduction - Quick Builders
from flext_core.patterns.quick_builders import flext_api_response
from flext_core.patterns.quick_builders import flext_batch_builder
from flext_core.patterns.quick_builders import flext_command
from flext_core.patterns.quick_builders import flext_config
from flext_core.patterns.quick_builders import flext_database_config
from flext_core.patterns.quick_builders import flext_env_config
from flext_core.patterns.quick_builders import flext_event
from flext_core.patterns.quick_builders import flext_field_rule
from flext_core.patterns.quick_builders import flext_paginated_response
from flext_core.patterns.quick_builders import flext_query_filter
from flext_core.patterns.quick_builders import flext_query_sort
from flext_core.patterns.quick_builders import flext_result_builder
from flext_core.patterns.quick_builders import flext_search_query
from flext_core.patterns.quick_builders import flext_validation_rules

# Boilerplate Reduction - Extended Types (100+ semantic types)
from flext_core.patterns.typedefs import FlextApiKey
from flext_core.patterns.typedefs import FlextCorrelationId
from flext_core.patterns.typedefs import FlextCustomerId
from flext_core.patterns.typedefs import FlextDatabaseUrl
from flext_core.patterns.typedefs import FlextEmail
from flext_core.patterns.typedefs import FlextEventId
from flext_core.patterns.typedefs import FlextFileName
from flext_core.patterns.typedefs import FlextHostname
from flext_core.patterns.typedefs import FlextOrderId
from flext_core.patterns.typedefs import FlextPassword
from flext_core.patterns.typedefs import FlextProductId
from flext_core.patterns.typedefs import FlextUrl
from flext_core.patterns.typedefs import FlextUserId
from flext_core.patterns.typedefs import FlextUserName

# Modern Type System
from flext_core.payload import FlextPayload

# Core Result Pattern
from flext_core.result import FlextResult
from flext_core.result import error
from flext_core.result import failed
from flext_core.result import failure
from flext_core.result import success
from flext_core.result import successful

# Advanced Type System (additional boilerplate reduction)
from flext_core.types_advanced import AnyId
from flext_core.types_advanced import AsyncResult
from flext_core.types_advanced import Builder
from flext_core.types_advanced import Cacheable
from flext_core.types_advanced import CommandHandler
from flext_core.types_advanced import DataInput
from flext_core.types_advanced import DataOutput
from flext_core.types_advanced import Either
from flext_core.types_advanced import EntityId
from flext_core.types_advanced import ErrorHandler
from flext_core.types_advanced import EventHandler
from flext_core.types_advanced import Factory
from flext_core.types_advanced import FlextDict
from flext_core.types_advanced import FlextList
from flext_core.types_advanced import FlextMapping
from flext_core.types_advanced import FlextSequence
from flext_core.types_advanced import Identifiable
from flext_core.types_advanced import MetadataDict
from flext_core.types_advanced import Pipe
from flext_core.types_advanced import Predicate
from flext_core.types_advanced import QueryHandler
from flext_core.types_advanced import Repository
from flext_core.types_advanced import RequestId
from flext_core.types_advanced import ResultOf
from flext_core.types_advanced import Serializable
from flext_core.types_advanced import Service
from flext_core.types_advanced import SessionId
from flext_core.types_advanced import Timestamp
from flext_core.types_advanced import Timestamped
from flext_core.types_advanced import Transformer
from flext_core.types_advanced import UserId
from flext_core.types_advanced import Validatable
from flext_core.types_advanced import Validator
from flext_core.types_advanced import ensure_dict
from flext_core.types_advanced import ensure_list
from flext_core.types_advanced import ensure_result
from flext_core.types_advanced import is_identifiable
from flext_core.types_advanced import is_result_type
from flext_core.types_advanced import is_serializable
from flext_core.types_advanced import is_timestamped
from flext_core.types_advanced import is_validatable

# NOTE: Singer patterns moved to flext-meltano (Layer 3)
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

# Validation System
from flext_core.validators import ChoiceValidator
from flext_core.validators import EmailValidator
from flext_core.validators import FlextValidationChain
from flext_core.validators import LengthValidator
from flext_core.validators import NotEmptyValidator
from flext_core.validators import NotNoneValidator
from flext_core.validators import RangeValidator
from flext_core.validators import RegexValidator
from flext_core.validators import TypeValidator
from flext_core.validators import validate
from flext_core.validators import validate_all
from flext_core.validators import validate_any
from flext_core.validators import validate_choice
from flext_core.validators import validate_email
from flext_core.validators import validate_number
from flext_core.validators import validate_string
from flext_core.validators import validate_type
from flext_core.version import get_version

# Public API - Modern Components Only
__all__ = [
    "AnyId",
    "AsyncResult",
    "Builder",
    "Cacheable",
    "ChoiceValidator",
    "CommandHandler",
    "ContainerMixin",
    "DataInput",
    "DataOutput",
    "Either",
    "EmailValidator",
    "EntityId",
    "ErrorHandler",
    "EventHandler",
    "Factory",
    "FlextAggregateRoot",
    "FlextAuditMixin",
    "FlextAuthenticationError",
    "FlextBuilder",
    "FlextCacheableMixin",
    "FlextCommand",
    "FlextCommandBus",
    "FlextCommandHandler",
    "FlextCommandId",
    "FlextCommandResult",
    "FlextComparableMixin",
    "FlextConfigBuilder",
    "FlextConfigError",
    "FlextConfigKey",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextConstants",
    "FlextContainer",
    "FlextContextData",
    "FlextCoreSettings",
    "FlextCriticalError",
    "FlextDict",
    "FlextDomainService",
    "FlextEntity",
    "FlextEntityId",
    "FlextEntityMixin",
    "FlextEnvironment",
    "FlextError",
    "FlextEventHandler",
    "FlextEventType",
    "FlextFactory",
    "FlextField",
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldType",
    "FlextFieldValidator",
    "FlextHandler",
    "FlextHandlerId",
    "FlextIdentifier",
    "FlextList",
    "FlextLogContext",
    "FlextLogLevel",
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerName",
    "FlextMapping",
    "FlextMessageHandler",
    "FlextMetadataMixin",
    "FlextMigrationError",
    "FlextPayload",
    "FlextProcessingError",
    "FlextRequestHandler",
    "FlextResourceId",
    "FlextResult",
    "FlextResultStatus",
    "FlextSchemaError",
    "FlextSequence",
    "FlextSerializationMixin",
    "FlextServiceBuilder",
    "FlextServiceFactory",
    "FlextServiceName",
    "FlextSingletonFactory",
    "FlextTimeoutError",
    "FlextTimestampMixin",
    "FlextTraceId",
    "FlextTypedDict",
    "FlextValidationChain",
    "FlextValidationError",
    "FlextValidationMixin",
    "FlextValidationResult",
    "FlextValidationRule",
    "FlextValidator",
    "FlextValidatorId",
    "FlextValueObject",
    "FlextValueObjectMixin",
    "Identifiable",
    "LengthValidator",
    "LoggerMixin",
    "MetadataDict",
    "NotEmptyValidator",
    "NotNoneValidator",
    "Pipe",
    "Pipeline",
    "Predicate",
    "QueryHandler",
    "QuickEntity",
    "QuickValueObject",
    "RangeValidator",
    "RegexValidator",
    "Repository",
    "RequestId",
    "ResultOf",
    "Serializable",
    "Service",
    "SessionId",
    "Timestamp",
    "Timestamped",
    "Transformer",
    "TypeValidator",
    "UserId",
    "Validatable",
    "Validator",
    "ValidatorMixin",
    "build_config",
    "build_service_config",
    "cache_result",
    "chain",
    "configure_flext_container",
    "configure_settings",
    "create_factory",
    "create_singleton_factory",
    "ensure_dict",
    "ensure_list",
    "ensure_result",
    "error",
    "fail",
    "failed",
    "failure",
    "flatten_dict",
    "flext_api_response",
    "flext_async_safe",
    "flext_batch_builder",
    "flext_cache_result",
    "flext_command",
    "flext_config",
    "flext_database_config",
    "flext_env_config",
    "flext_event",
    "flext_field_rule",
    "flext_paginated_response",
    "flext_query_filter",
    "flext_query_sort",
    "flext_require_non_none",
    "flext_result_builder",
    "flext_retry",
    "flext_robust",
    "flext_safe_result",
    "flext_search_query",
    "flext_timed",
    "flext_validate_config_key",
    "flext_validate_event_type",
    "flext_validate_identifier",
    "flext_validate_non_empty_string",
    "flext_validate_result",
    "flext_validate_service_name",
    "flext_validation_rules",
    "from_dict",
    "get_flext_container",
    "get_logger",
    "get_service",
    "get_settings",
    "get_version",
    "inject",
    "is_identifiable",
    "is_result_type",
    "is_serializable",
    "is_timestamped",
    "is_validatable",
    "log_calls",
    "merge_dicts",
    "ok",
    "omit_keys",
    "pick_keys",
    "pipeline",
    "register",
    "retry",
    "safe",
    "safe_get",
    "safe_set_nested",
    "success",
    "successful",
    "to_dict",
    "validate",
    "validate_all",
    "validate_any",
    "validate_choice",
    "validate_email",
    "validate_non_empty",
    "validate_number",
    "validate_required",
    "validate_string",
    "validate_type",
]

# Library metadata
__version__ = "0.8.0"
__author__ = "FLEXT Contributors"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 FLEXT Contributors"
