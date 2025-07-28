"""FLEXT Core Types Module.

Public interface for the FLEXT Core type system providing enterprise-grade
type definitions
and structural typing. Exposes types from internal implementation following
single source
of truth pattern.

Architecture:
    - Single source of truth pattern: imports from flext_types.py internal
      implementation
    - Clean public API without underscore prefixes for maximum usability
    - Direct exposure of all type definitions for external consumption
    - Maintains backward compatibility through consistent naming patterns

Usage:
    from flext_core.types import TContextDict, TLogMessage, TEntityId

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Import all types from internal implementation
from flext_core.flext_types import (
    # Core type variables
    E,
    F,
    P,
    R,
    T,
    # Collection types
    TAnyDict,
    TAnyList,
    # Security types
    TApiKey,
    # Business type aliases
    TBusinessCode,
    # Cache types
    TCacheKey,
    TCacheValue,
    # CQRS type variables
    TCommand,
    # Configuration types
    TConfigKey,
    TConfigValue,
    # Database types
    TConnectionString,
    # Context and logging
    TContextDict,
    TCorrelationId,
    # Business identifier types
    TCustomerId,
    TDatabaseName,
    # Data types
    TEmailAddress,
    # Domain type variables
    TEntity,
    TEntityId,
    TEnvironment,
    # Error types
    TErrorCode,
    TErrorMessage,
    TEvent,
    # Message types
    TEventName,
    # Validation types
    TFieldName,
    # File system types
    TFileName,
    TFilePath,
    # Handler type variables
    THandler,
    # Network types
    THostname,
    TIPAddress,
    # Process types
    TJobId,
    TJson,
    TLogMessage,
    TMessageType,
    # Metric types
    TMetricName,
    TMetricValue,
    TOrderId,
    TPassword,
    TPhoneNumber,
    TPort,
    # Functional types
    TPredicate,
    TProcessId,
    TProductId,
    TQuery,
    TService,
    # Service type variables
    TServiceKey,
    TServiceName,
    TSessionId,
    # Workflow types
    TStepName,
    TTableName,
    TTaskId,
    # Template types
    TTemplateName,
    TTemplateVar,
    TTimestamp,
    TToken,
    TTopicName,
    TTransformer,
    TUrl,
    TUserId,
    TUsername,
    TValidationRule,
    TValue,
    TWorkflowId,
    # Protocol definitions (if any are exported)
    U,
    V,
)

__all__ = [
    # Core type variables
    "E",
    "F",
    "P",
    "R",
    "T",
    "U",
    "V",
    # Domain type variables
    "TEntity",
    "TService",
    "TValue",
    # CQRS type variables
    "TCommand",
    "TEvent",
    "TQuery",
    # Handler type variables
    "THandler",
    # Service type variables
    "TServiceKey",
    "TServiceName",
    # Business type aliases
    "TBusinessCode",
    "TCorrelationId",
    "TEntityId",
    # Context and logging
    "TContextDict",
    "TLogMessage",
    # Collection types
    "TAnyDict",
    "TAnyList",
    # Functional types
    "TPredicate",
    "TTransformer",
    # Network types
    "THostname",
    "TIPAddress",
    "TPort",
    "TUrl",
    # Database types
    "TConnectionString",
    "TDatabaseName",
    "TTableName",
    # Security types
    "TApiKey",
    "TPassword",
    "TToken",
    "TUsername",
    # Data types
    "TEmailAddress",
    "TJson",
    "TPhoneNumber",
    "TTimestamp",
    # Business identifier types
    "TCustomerId",
    "TOrderId",
    "TProductId",
    "TUserId",
    # Process types
    "TJobId",
    "TProcessId",
    "TSessionId",
    "TTaskId",
    # Configuration types
    "TConfigKey",
    "TConfigValue",
    "TEnvironment",
    # Validation types
    "TFieldName",
    "TValidationRule",
    # Error types
    "TErrorCode",
    "TErrorMessage",
    # File system types
    "TFileName",
    "TFilePath",
    # Message types
    "TEventName",
    "TMessageType",
    "TTopicName",
    # Metric types
    "TMetricName",
    "TMetricValue",
    # Cache types
    "TCacheKey",
    "TCacheValue",
    # Template types
    "TTemplateName",
    "TTemplateVar",
    # Workflow types
    "TStepName",
    "TWorkflowId",
]
