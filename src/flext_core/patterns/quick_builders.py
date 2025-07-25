"""FLEXT Core Quick Builders - Rapid Object Construction.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Quick builders that dramatically reduce boilerplate for common object
construction patterns while maintaining type safety and validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from flext_core.patterns.dict_helpers import FlextDict
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

# =============================================================================
# CONFIGURATION BUILDERS - Rapid Config Construction
# =============================================================================


def flext_config(**kwargs: Any) -> FlextDict:
    """Quick builder for configuration objects.

    Args:
        kwargs: Configuration key-value pairs

    Returns:
        FlextDict with configuration data

    Example:
        config = flext_config(
            database_url="postgresql://localhost/db",
            debug=True,
            max_connections=10
        )

        db_url = config.get_str("database_url").unwrap()
        debug = config.get_bool("debug").unwrap()
    """
    return FlextDict(kwargs)


def flext_env_config(
    prefix: str = "",
    defaults: Mapping[str, Any] | None = None,
) -> FlextDict:
    """Build configuration from environment variables.

    Args:
        prefix: Environment variable prefix to filter by
        defaults: Default values for missing environment variables

    Returns:
        FlextDict with environment configuration

    Example:
        # Reads DATABASE_URL, DEBUG, etc.
        config = flext_env_config(prefix="APP_", defaults={
            "debug": False,
            "port": 8000
        })
    """
    import os

    config_data = dict(defaults) if defaults else {}

    for key, value in os.environ.items():
        if prefix and not key.startswith(prefix):
            continue

        # Remove prefix and convert to lowercase
        config_key = key.removeprefix(prefix).lower()
        config_data[config_key] = value

    return FlextDict(config_data)


def flext_database_config(
    url: str,
    **options: Any,
) -> FlextDict:
    """Build database configuration with common defaults.

    Args:
        url: Database connection URL
        options: Additional database options

    Returns:
        FlextDict with database configuration

    Example:
        db_config = flext_database_config(
            "postgresql://localhost/mydb",
            pool_size=20,
            timeout=30,
            echo=True
        )
    """
    config = {
        "url": url,
        "pool_size": options.get("pool_size", 10),
        "pool_timeout": options.get("pool_timeout", 30),
        "pool_recycle": options.get("pool_recycle", 3600),
        "echo": options.get("echo", False),
        "autocommit": options.get("autocommit", False),
        "autoflush": options.get("autoflush", True),
        **{k: v for k, v in options.items() if k not in {
            "pool_size", "pool_timeout", "pool_recycle",
            "echo", "autocommit", "autoflush",
        }},
    }
    return FlextDict(config)


# =============================================================================
# API BUILDERS - REST API Construction
# =============================================================================


def flext_api_response(
    success: bool = True,
    data: Any = None,
    error: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FlextDict:
    """Build standardized API response.

    Args:
        success: Whether the operation succeeded
        data: Response data
        error: Error message if failed
        metadata: Additional response metadata

    Returns:
        FlextDict with API response structure

    Example:
        # Success response
        response = flext_api_response(
            success=True,
            data={"user_id": "123", "name": "Alice"},
            metadata={"timestamp": "2025-01-01T00:00:00Z"}
        )

        # Error response
        error_response = flext_api_response(
            success=False,
            error="User not found",
            metadata={"error_code": "USER_NOT_FOUND"}
        )
    """
    response_data = {
        "success": success,
        "data": data,
        "error": error,
        "metadata": metadata or {},
    }
    return FlextDict(response_data)


def flext_paginated_response(
    items: list[Any],
    page: int = 1,
    page_size: int = 20,
    total_items: int | None = None,
    **metadata: Any,
) -> FlextDict:
    """Build paginated API response.

    Args:
        items: List of items for current page
        page: Current page number (1-based)
        page_size: Number of items per page
        total_items: Total number of items (if known)
        metadata: Additional pagination metadata

    Returns:
        FlextDict with paginated response

    Example:
        response = flext_paginated_response(
            items=[{"id": 1, "name": "Item 1"}],
            page=1,
            page_size=10,
            total_items=100,
            has_next=True
        )
    """
    pagination_data = {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "item_count": len(items),
            "has_next": None,
            "has_previous": page > 1,
            **metadata,
        },
    }

    # Calculate has_next if total_items is known
    if total_items is not None:
        total_pages = (total_items + page_size - 1) // page_size
        pagination_data["pagination"]["total_pages"] = total_pages
        pagination_data["pagination"]["has_next"] = page < total_pages

    return FlextDict(pagination_data)


# =============================================================================
# EVENT BUILDERS - Event-Driven Architecture
# =============================================================================


def flext_event(
    event_type: str,
    data: Any = None,
    event_id: str | None = None,
    correlation_id: str | None = None,
    **metadata: Any,
) -> FlextDict:
    """Build standardized event object.

    Args:
        event_type: Type of event (e.g., "user.created")
        data: Event payload data
        event_id: Unique event identifier
        correlation_id: Correlation ID for event tracing
        metadata: Additional event metadata

    Returns:
        FlextDict with event structure

    Example:
        event = flext_event(
            event_type="user.created",
            data={"user_id": "123", "email": "alice@example.com"},
            correlation_id="request-456",
            source="user-service",
            version="1.0"
        )
    """
    import uuid
    from datetime import UTC
    from datetime import datetime

    event_data = {
        "event_id": event_id or str(uuid.uuid4()),
        "event_type": event_type,
        "data": data,
        "correlation_id": correlation_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {
            "version": metadata.get("version", "1.0"),
            "source": metadata.get("source", "unknown"),
            **{k: v for k, v in metadata.items() if k not in {"version", "source"}},
        },
    }
    return FlextDict(event_data)


def flext_command(
    command_type: str,
    data: Any = None,
    command_id: str | None = None,
    user_id: str | None = None,
    **metadata: Any,
) -> FlextDict:
    """Build standardized command object.

    Args:
        command_type: Type of command (e.g., "create_user")
        data: Command payload data
        command_id: Unique command identifier
        user_id: ID of user executing command
        metadata: Additional command metadata

    Returns:
        FlextDict with command structure

    Example:
        command = flext_command(
            command_type="create_user",
            data={"name": "Alice", "email": "alice@example.com"},
            user_id="REDACTED_LDAP_BIND_PASSWORD-123",
            priority="high"
        )
    """
    import uuid
    from datetime import UTC
    from datetime import datetime

    command_data = {
        "command_id": command_id or str(uuid.uuid4()),
        "command_type": command_type,
        "data": data,
        "user_id": user_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {
            "priority": metadata.get("priority", "normal"),
            "retry_count": metadata.get("retry_count", 0),
            **{k: v for k, v in metadata.items() if k not in {"priority", "retry_count"}},
        },
    }
    return FlextDict(command_data)


# =============================================================================
# VALIDATION BUILDERS - Input Validation
# =============================================================================


def flext_validation_rules(**rules: Any) -> dict[str, Any]:
    """Build validation rules dictionary.

    Args:
        rules: Validation rules as key-value pairs

    Returns:
        Dictionary with validation rules

    Example:
        rules = flext_validation_rules(
            name={"required": True, "min_length": 2, "max_length": 50},
            email={"required": True, "format": "email"},
            age={"required": False, "type": "integer", "min": 0, "max": 150}
        )
    """
    return dict(rules)


def flext_field_rule(
    required: bool = False,
    field_type: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    pattern: str | None = None,
    allowed_values: list[Any] | None = None,
    **custom: Any,
) -> dict[str, Any]:
    r"""Build field validation rule.

    Args:
        required: Whether field is required
        field_type: Expected field type
        min_length: Minimum string/list length
        max_length: Maximum string/list length
        min_value: Minimum numeric value
        max_value: Maximum numeric value
        pattern: Regex pattern for validation
        allowed_values: List of allowed values
        custom: Additional custom validation rules

    Returns:
        Dictionary with field validation rule

    Example:
        name_rule = flext_field_rule(
            required=True,
            field_type="string",
            min_length=2,
            max_length=50,
            pattern=r"^[a-zA-Z\s]+$"
        )

        age_rule = flext_field_rule(
            required=False,
            field_type="integer",
            min_value=0,
            max_value=150
        )
    """
    rule = {"required": required}

    if field_type is not None:
        rule["type"] = field_type
    if min_length is not None:
        rule["min_length"] = min_length
    if max_length is not None:
        rule["max_length"] = max_length
    if min_value is not None:
        rule["min_value"] = min_value
    if max_value is not None:
        rule["max_value"] = max_value
    if pattern is not None:
        rule["pattern"] = pattern
    if allowed_values is not None:
        rule["allowed_values"] = allowed_values

    rule.update(custom)
    return rule


# =============================================================================
# QUERY BUILDERS - Database and Search Queries
# =============================================================================


def flext_query_filter(
    field: str,
    operator: str = "eq",
    value: Any = None,
    **options: Any,
) -> FlextDict:
    """Build query filter.

    Args:
        field: Field name to filter on
        operator: Filter operator (eq, ne, gt, lt, gte, lte, in, contains, etc.)
        value: Filter value
        options: Additional filter options

    Returns:
        FlextDict with filter specification

    Example:
        # Simple equality filter
        name_filter = flext_query_filter("name", "eq", "Alice")

        # Range filter
        age_filter = flext_query_filter("age", "gte", 18)

        # Contains filter
        email_filter = flext_query_filter(
            "email",
            "contains",
            "@example.com",
            case_sensitive=False
        )
    """
    filter_data = {
        "field": field,
        "operator": operator,
        "value": value,
        **options,
    }
    return FlextDict(filter_data)


def flext_query_sort(
    field: str,
    direction: str = "asc",
    **options: Any,
) -> FlextDict:
    """Build query sort specification.

    Args:
        field: Field name to sort by
        direction: Sort direction ("asc" or "desc")
        options: Additional sort options

    Returns:
        FlextDict with sort specification

    Example:
        # Simple ascending sort
        name_sort = flext_query_sort("name", "asc")

        # Descending sort with nulls handling
        date_sort = flext_query_sort(
            "created_at",
            "desc",
            nulls="last"
        )
    """
    sort_data = {
        "field": field,
        "direction": direction,
        **options,
    }
    return FlextDict(sort_data)


def flext_search_query(
    query_text: str = "",
    filters: list[FlextDict] | None = None,
    sorts: list[FlextDict] | None = None,
    page: int = 1,
    page_size: int = 20,
    **options: Any,
) -> FlextDict:
    """Build comprehensive search query.

    Args:
        query_text: Free-text search query
        filters: List of query filters
        sorts: List of sort specifications
        page: Page number (1-based)
        page_size: Number of results per page
        options: Additional query options

    Returns:
        FlextDict with complete search query

    Example:
        search = flext_search_query(
            query_text="python developer",
            filters=[
                flext_query_filter("department", "eq", "engineering"),
                flext_query_filter("experience_years", "gte", 3)
            ],
            sorts=[
                flext_query_sort("relevance_score", "desc"),
                flext_query_sort("created_at", "desc")
            ],
            page=1,
            page_size=10,
            highlight=True
        )
    """
    query_data = {
        "query": query_text,
        "filters": [f.to_dict() for f in (filters or [])],
        "sorts": [s.to_dict() for s in (sorts or [])],
        "pagination": {
            "page": page,
            "page_size": page_size,
        },
        "options": options,
    }
    return FlextDict(query_data)


# =============================================================================
# UTILITY BUILDERS - Common Patterns
# =============================================================================


def flext_result_builder(
    success: bool = True,
    data: Any = None,
    error: str | None = None,
) -> FlextResult[Any]:
    """Quick builder for FlextResult objects.

    Args:
        success: Whether the result represents success
        data: Result data if successful
        error: Error message if failed

    Returns:
        FlextResult with specified state

    Example:
        # Success result
        result = flext_result_builder(True, {"user_id": "123"})

        # Error result
        error_result = flext_result_builder(False, None, "Validation failed")
    """
    if success:
        return FlextResult.ok(data)
    return FlextResult.fail(error or "Operation failed")


def flext_batch_builder[T](
    items: list[T],
    batch_size: int = 100,
    processor: Callable[[list[T]], FlextResult[Any]] | None = None,
) -> FlextResult[list[Any]]:
    """Build batch processing operation.

    Args:
        items: Items to process in batches
        batch_size: Number of items per batch
        processor: Optional function to process each batch

    Returns:
        FlextResult with batch processing results

    Example:
        def process_users(user_batch):
            # Process batch of users
            return FlextResult.ok([user.id for user in user_batch])

        result = flext_batch_builder(
            items=all_users,
            batch_size=50,
            processor=process_users
        )
    """
    if not processor:
        # If no processor, just return batched items
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        return FlextResult.ok(batches)

    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = processor(batch)

        if not batch_result.is_success:
            return FlextResult.fail(f"Batch processing failed: {batch_result.error}")

        if batch_result.data is not None:
            results.append(batch_result.data)

    return FlextResult.ok(results)


# =============================================================================
# EXPORTS - Clean Public API
# =============================================================================

__all__ = [
    "flext_api_response",
    "flext_batch_builder",
    "flext_command",
    "flext_config",
    "flext_database_config",
    "flext_env_config",
    "flext_event",
    "flext_field_rule",
    "flext_paginated_response",
    "flext_query_filter",
    "flext_query_sort",
    "flext_result_builder",
    "flext_search_query",
    "flext_validation_rules",
]
