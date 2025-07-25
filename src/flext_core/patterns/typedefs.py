"""FLEXT Core Type Definitions - Unified Type System.

Standardized type aliases for consistent typing across FLEXT ecosystem.
Dramatically reduces boilerplate by providing semantic types for all common
application patterns.
"""

from __future__ import annotations

from typing import Any
from typing import NewType

# =============================================================================
# PATTERN TYPE ALIASES - Standardized types for enterprise patterns
# =============================================================================

# Handler System Types
FlextHandlerId = NewType("FlextHandlerId", str)
FlextHandlerName = NewType("FlextHandlerName", str)
FlextMessageType = NewType("FlextMessageType", str)

# Command Pattern Types
FlextCommandId = NewType("FlextCommandId", str)
FlextCommandName = NewType("FlextCommandName", str)
FlextCommandType = NewType("FlextCommandType", str)

# Validation System Types
FlextValidatorId = NewType("FlextValidatorId", str)
FlextValidatorName = NewType("FlextValidatorName", str)
FlextRuleName = NewType("FlextRuleName", str)

# Field System Types
FlextFieldId = NewType("FlextFieldId", str)
FlextFieldName = NewType("FlextFieldName", str)
FlextFieldPath = NewType("FlextFieldPath", str)

# Logging System Types
FlextLoggerName = NewType("FlextLoggerName", str)
FlextLoggerContext = NewType("FlextLoggerContext", str)
FlextLogTag = NewType("FlextLogTag", str)

# Generic System Types
FlextPatternName = NewType("FlextPatternName", str)
FlextPatternId = NewType("FlextPatternId", str)
FlextMetadataKey = NewType("FlextMetadataKey", str)

# Data Processing Types
FlextDataPath = NewType("FlextDataPath", str)
FlextDataKey = NewType("FlextDataKey", str)
FlextDataValue = Any

# =============================================================================
# APPLICATION DOMAIN TYPES - Common Business Concepts
# =============================================================================

# User & Authentication Types
FlextUserId = NewType("FlextUserId", str)
FlextUserName = NewType("FlextUserName", str)
FlextEmail = NewType("FlextEmail", str)
FlextPassword = NewType("FlextPassword", str)
FlextToken = NewType("FlextToken", str)
FlextApiKey = NewType("FlextApiKey", str)
FlextSessionId = NewType("FlextSessionId", str)

# Organization & Role Types
FlextOrgId = NewType("FlextOrgId", str)
FlextOrgName = NewType("FlextOrgName", str)
FlextRoleId = NewType("FlextRoleId", str)
FlextRoleName = NewType("FlextRoleName", str)
FlextPermission = NewType("FlextPermission", str)

# Document & Content Types
FlextDocumentId = NewType("FlextDocumentId", str)
FlextDocumentTitle = NewType("FlextDocumentTitle", str)
FlextFileName = NewType("FlextFileName", str)
FlextFileHash = NewType("FlextFileHash", str)
FlextMimeType = NewType("FlextMimeType", str)
FlextContentType = NewType("FlextContentType", str)

# Database & Storage Types
FlextTableName = NewType("FlextTableName", str)
FlextColumnName = NewType("FlextColumnName", str)
FlextIndexName = NewType("FlextIndexName", str)
FlextDatabaseUrl = NewType("FlextDatabaseUrl", str)
FlextConnectionString = NewType("FlextConnectionString", str)

# Network & API Types
FlextUrl = NewType("FlextUrl", str)
FlextEndpoint = NewType("FlextEndpoint", str)
FlextHostname = NewType("FlextHostname", str)
FlextPort = NewType("FlextPort", int)
FlextIpAddress = NewType("FlextIpAddress", str)
FlextRequestId = NewType("FlextRequestId", str)

# Configuration Types
FlextConfigName = NewType("FlextConfigName", str)
FlextConfigValue = NewType("FlextConfigValue", str)
FlextEnvironment = NewType("FlextEnvironment", str)
FlextNamespace = NewType("FlextNamespace", str)
FlextProfile = NewType("FlextProfile", str)

# Business Entity Types
FlextCustomerId = NewType("FlextCustomerId", str)
FlextOrderId = NewType("FlextOrderId", str)
FlextProductId = NewType("FlextProductId", str)
FlextInvoiceId = NewType("FlextInvoiceId", str)
FlextTransactionId = NewType("FlextTransactionId", str)

# Temporal Types
FlextTimestamp = NewType("FlextTimestamp", str)
FlextDateString = NewType("FlextDateString", str)
FlextTimeString = NewType("FlextTimeString", str)
FlextDuration = NewType("FlextDuration", str)
FlextCronExpression = NewType("FlextCronExpression", str)

# =============================================================================
# MESSAGING & COMMUNICATION TYPES - Event Driven Architecture
# =============================================================================

# Event System Types
FlextEventId = NewType("FlextEventId", str)
FlextEventName = NewType("FlextEventName", str)
FlextEventVersion = NewType("FlextEventVersion", str)
FlextEventStream = NewType("FlextEventStream", str)
FlextCorrelationId = NewType("FlextCorrelationId", str)

# Queue & Messaging Types
FlextQueueName = NewType("FlextQueueName", str)
FlextTopicName = NewType("FlextTopicName", str)
FlextChannelName = NewType("FlextChannelName", str)
FlextSubscriptionId = NewType("FlextSubscriptionId", str)
FlextMessageId = NewType("FlextMessageId", str)

# Notification Types
FlextNotificationId = NewType("FlextNotificationId", str)
FlextNotificationType = NewType("FlextNotificationType", str)
FlextTemplateId = NewType("FlextTemplateId", str)
FlextRecipient = NewType("FlextRecipient", str)

# =============================================================================
# MONITORING & OBSERVABILITY TYPES - Operations
# =============================================================================

# Metrics & Monitoring Types
FlextMetricName = NewType("FlextMetricName", str)
FlextMetricValue = NewType("FlextMetricValue", float)
FlextMetricTag = NewType("FlextMetricTag", str)
FlextAlertId = NewType("FlextAlertId", str)
FlextDashboardId = NewType("FlextDashboardId", str)

# Health & Status Types
FlextHealthStatus = NewType("FlextHealthStatus", str)
FlextServiceStatus = NewType("FlextServiceStatus", str)
FlextCheckName = NewType("FlextCheckName", str)
FlextErrorCode = NewType("FlextErrorCode", str)

# =============================================================================
# INTEGRATION & EXTERNAL TYPES - Third Party Systems
# =============================================================================

# External System Types
FlextSystemId = NewType("FlextSystemId", str)
FlextSystemName = NewType("FlextSystemName", str)
FlextExternalId = NewType("FlextExternalId", str)
FlextExternalRef = NewType("FlextExternalRef", str)
FlextIntegrationId = NewType("FlextIntegrationId", str)

# Schema & Format Types
FlextSchemaId = NewType("FlextSchemaId", str)
FlextSchemaVersion = NewType("FlextSchemaVersion", str)
FlextFormat = NewType("FlextFormat", str)
FlextProtocol = NewType("FlextProtocol", str)

# =============================================================================
# SEARCH & QUERY TYPES - Data Access Patterns
# =============================================================================

# Search Types
FlextSearchQuery = NewType("FlextSearchQuery", str)
FlextSearchFilter = NewType("FlextSearchFilter", str)
FlextSearchSort = NewType("FlextSearchSort", str)
FlextIndexKey = NewType("FlextIndexKey", str)

# Pagination Types
FlextPageToken = NewType("FlextPageToken", str)
FlextCursor = NewType("FlextCursor", str)
FlextOffset = NewType("FlextOffset", int)
FlextLimit = NewType("FlextLimit", int)

# =============================================================================
# RESOURCE & LOCATION TYPES - Physical and Logical Resources
# =============================================================================

# Path & Location Types
FlextPath = NewType("FlextPath", str)
FlextDirectoryPath = NewType("FlextDirectoryPath", str)
FlextFilePath = NewType("FlextFilePath", str)
FlextResourcePath = NewType("FlextResourcePath", str)
FlextLocation = NewType("FlextLocation", str)

# Version & Release Types
FlextVersion = NewType("FlextVersion", str)
FlextReleaseId = NewType("FlextReleaseId", str)
FlextBranch = NewType("FlextBranch", str)
FlextCommitHash = NewType("FlextCommitHash", str)
FlextTag = NewType("FlextTag", str)

# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextAlertId",
    "FlextApiKey",
    "FlextBranch",
    "FlextChannelName",
    "FlextCheckName",
    "FlextColumnName",
    # Original Pattern Types
    "FlextCommandId",
    "FlextCommandName",
    "FlextCommandType",
    "FlextCommitHash",
    "FlextConfigName",
    "FlextConfigValue",
    "FlextConnectionString",
    "FlextContentType",
    "FlextCorrelationId",
    "FlextCronExpression",
    "FlextCursor",
    "FlextCustomerId",
    "FlextDashboardId",
    "FlextDataKey",
    "FlextDataPath",
    "FlextDataValue",
    "FlextDatabaseUrl",
    "FlextDateString",
    "FlextDirectoryPath",
    "FlextDocumentId",
    "FlextDocumentTitle",
    "FlextDuration",
    "FlextEmail",
    "FlextEndpoint",
    "FlextEnvironment",
    "FlextErrorCode",
    # Messaging & Communication Types
    "FlextEventId",
    "FlextEventName",
    "FlextEventStream",
    "FlextEventVersion",
    "FlextExternalId",
    "FlextExternalRef",
    "FlextFieldId",
    "FlextFieldName",
    "FlextFieldPath",
    "FlextFileHash",
    "FlextFileName",
    "FlextFilePath",
    "FlextFormat",
    "FlextHandlerId",
    "FlextHandlerName",
    "FlextHealthStatus",
    "FlextHostname",
    "FlextIndexKey",
    "FlextIndexName",
    "FlextIntegrationId",
    "FlextInvoiceId",
    "FlextIpAddress",
    "FlextLimit",
    "FlextLocation",
    "FlextLogTag",
    "FlextLoggerContext",
    "FlextLoggerName",
    "FlextMessageId",
    "FlextMessageType",
    "FlextMetadataKey",
    # Monitoring & Observability Types
    "FlextMetricName",
    "FlextMetricTag",
    "FlextMetricValue",
    "FlextMimeType",
    "FlextNamespace",
    "FlextNotificationId",
    "FlextNotificationType",
    "FlextOffset",
    "FlextOrderId",
    "FlextOrgId",
    "FlextOrgName",
    "FlextPageToken",
    "FlextPassword",
    # Resource & Location Types
    "FlextPath",
    "FlextPatternId",
    "FlextPatternName",
    "FlextPermission",
    "FlextPort",
    "FlextProductId",
    "FlextProfile",
    "FlextProtocol",
    "FlextQueueName",
    "FlextRecipient",
    "FlextReleaseId",
    "FlextRequestId",
    "FlextResourcePath",
    "FlextRoleId",
    "FlextRoleName",
    "FlextRuleName",
    "FlextSchemaId",
    "FlextSchemaVersion",
    "FlextSearchFilter",
    # Search & Query Types
    "FlextSearchQuery",
    "FlextSearchSort",
    "FlextServiceStatus",
    "FlextSessionId",
    "FlextSubscriptionId",
    # Integration & External Types
    "FlextSystemId",
    "FlextSystemName",
    "FlextTableName",
    "FlextTag",
    "FlextTemplateId",
    "FlextTimeString",
    "FlextTimestamp",
    "FlextToken",
    "FlextTopicName",
    "FlextTransactionId",
    "FlextUrl",
    # Application Domain Types
    "FlextUserId",
    "FlextUserName",
    "FlextValidatorId",
    "FlextValidatorName",
    "FlextVersion",
]
