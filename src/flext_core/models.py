"""Domain models aligned with the FLEXT 1.0.0 modernization charter.

Entities, value objects, and aggregates mirror the design captured in
``README.md`` and ``docs/architecture.md`` so downstream packages share a
consistent DDD foundation during the rollout.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import re
import uuid
import warnings
from collections import UserString
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import T
from flext_core.utilities import FlextUtilities


def _get_global_config() -> FlextConfig:
    """Lazy import helper to break circular dependency."""
    from flext_core.config import FlextConfig
    return FlextConfig.get_global_instance()


class FlextModels:
    """Namespace for the canonical FLEXT domain primitives.

    Conservative refactored to maintain only used subclasses (Entity, Command, Value)
    and use composition with FlextUtilities for validation to eliminate duplication.
    These types lock in the immutable, timestamped data structures referenced
    by the modernization plan and exercised across the FLEXT ecosystem.
    """

    # Composition helper for validation - delegates to FlextUtilities
    class _ValidationHelper:
        """Internal validation helper using FlextUtilities composition."""

        @staticmethod
        def validate_non_empty_string(value: str, field_name: str) -> str:
            """Validate non-empty string using consolidated validation patterns."""
            result = FlextUtilities.Validation.validate_string(
                value, min_length=1, field_name=field_name
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap()

        @staticmethod
        def validate_email_format(email: str) -> str:
            """Validate email format using consolidated validation patterns."""
            result = FlextUtilities.Validation.validate_email(email)
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap()

    class TimestampedModel(BaseModel):
        """Base model capturing creation/update timestamps for observability."""

        created_at: datetime = Field(default_factory=datetime.now)
        updated_at: datetime = Field(default_factory=datetime.now)

    class Entity(TimestampedModel):
        """Entity base with identity, versioning, and modernization-friendly events."""

        id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        version: int = Field(default=1)
        domain_events: list[FlextModels.Event] = Field(default_factory=list)

        def add_domain_event(self, event: FlextModels.Event) -> None:
            """Add domain event."""
            self.domain_events.append(event)

        def clear_domain_events(self) -> list[FlextModels.Event]:
            """Clear and return domain events."""
            events = self.domain_events.copy()
            self.domain_events.clear()
            return events

        def increment_version(self) -> None:
            """Increment version."""
            self.version += 1

        # Identity-based equality and hashing (by ID only)
        def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
            """Compare entities by ID."""
            if not isinstance(other, FlextModels.Entity):
                return False
            return self.id == other.id

        def __hash__(self) -> int:  # pragma: no cover - trivial
            """Hash entity by ID."""
            return hash(self.id)

    class Value(BaseModel):
        """Immutable value object base aligned with modernization guidance."""

        model_config = ConfigDict(frozen=True)

        def __hash__(self) -> int:  # pragma: no cover - trivial
            """Hash value object by all fields."""

            def make_hashable(obj: object) -> object:
                """Convert object to hashable representation."""
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                if isinstance(obj, list):
                    return tuple(make_hashable(item) for item in obj)
                if isinstance(obj, set):
                    # Convert to hashable items first, then sort by string representation
                    hashable_items = [make_hashable(item) for item in obj]
                    return tuple(sorted(hashable_items, key=str))
                return obj

            model_data = self.model_dump()
            hashable_data = make_hashable(model_data)
            return hash((self.__class__, hashable_data))

    class Payload(BaseModel, Generic[T]):
        """Message payload wrapper carrying standardized modernization metadata."""

        data: T = Field(..., description="Payload data")
        metadata: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        message_type: str = Field(default="", description="Message type identifier")
        source_service: str = Field(default="", description="Source service name")
        correlation_id: str = Field(
            default="",
            description="Correlation ID for tracking",
        )
        message_id: str = Field(default="", description="Unique message identifier")
        expires_at: datetime | None = Field(
            default=None,
            description="Payload expiration time",
        )

        def extract(self) -> T:
            """Extract payload data."""
            return self.data

        @property
        def is_expired(self) -> bool:
            """Check if payload is expired."""
            if self.expires_at is None:
                return False
            return datetime.now(UTC) > self.expires_at

    class Event(BaseModel):
        """Domain event record mirroring modernization telemetry needs."""

        event_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        event_type: str = Field(...)
        payload: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        aggregate_id: str = Field(..., min_length=1, description="Aggregate identifier")

        @field_validator("aggregate_id")
        @classmethod
        def validate_aggregate_id(cls, v: str) -> str:
            """Validate aggregate_id using FlextUtilities composition."""
            return FlextModels._ValidationHelper.validate_non_empty_string(
                v, "Aggregate identifier"
            )

    class Command(BaseModel):
        """Command message template for dispatcher-driven CQRS flows."""

        command_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        command_type: str = Field(...)
        payload: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        correlation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id,
        )
        user_id: str | None = Field(
            default=None,
            description="User ID associated with the command",
        )

        def validate_command(self) -> FlextResult[bool]:
            """Validate command using consolidated railway patterns."""
            return FlextUtilities.Validation.validate_string(
                self.command_type, min_length=1, field_name="command_type"
            ) >> (lambda _: FlextResult[bool].ok(data=True))

    class Query(BaseModel):
        """Query message template for dispatcher-driven CQRS flows with enhanced Pydantic 2 validation."""

        query_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        query_type: str = Field(..., min_length=1)
        filters: dict[str, object] = Field(default_factory=dict)
        pagination: dict[str, int] = Field(
            default_factory=lambda: {"page": 1, "size": 10},
        )
        user_id: str | None = Field(
            default=None,
            description="User ID associated with the query",
        )

        model_config = ConfigDict(
            extra="ignore",
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        @field_validator("query_type")
        @classmethod
        def validate_query_type(cls, v: str) -> str:
            """Validate query type using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="query_type"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

        @field_validator("query_id")
        @classmethod
        def validate_query_id(cls, v: str) -> str:
            """Validate query ID using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="query_id"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

        @field_validator("pagination")
        @classmethod
        def validate_pagination(cls, v: dict[str, int]) -> dict[str, int]:
            """Validate pagination parameters using FlextConstants."""
            page = v.get("page", 1)
            size = v.get("size", 10)

            if not isinstance(page, int) or page < 1:
                msg = "pagination.page must be a positive integer"
                raise ValueError(msg)

            if (
                not isinstance(size, int)
                or size < 1
                or size > FlextConstants.Cqrs.MAX_PAGE_SIZE
            ):
                msg = f"pagination.size must be between 1 and {FlextConstants.Cqrs.MAX_PAGE_SIZE}"
                raise ValueError(msg)

            return {"page": page, "size": size}

        @model_validator(mode="after")
        def validate_query_consistency(self) -> Self:
            """Validate query consistency using business rules."""
            # Additional business validation can be added here
            return self

        def validate_query(self) -> FlextResult[bool]:
            """Validate query using consolidated railway patterns."""
            try:
                # Pydantic 2 validation is already done during model creation
                # Additional business logic validation
                if not self.query_type:
                    return FlextResult[bool].fail(
                        "Query type cannot be empty",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[bool].ok(data=True)
            except Exception as e:
                return FlextResult[bool].fail(
                    f"Query validation failed: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

    class CqrsCommand(Command):
        """CQRS command with derived type metadata for consistent routing."""

        @model_validator(mode="before")
        @classmethod
        def _ensure_command_type(cls, data: object) -> object:
            """Populate command_type based on class name if missing."""
            if not isinstance(data, dict):
                return data
            if "command_type" not in data or not data.get("command_type"):
                name = cls.__name__
                base = name.removesuffix("Command")
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                data["command_type"] = re.sub(
                    r"([a-z0-9])([A-Z])",
                    r"\1_\2",
                    s1,
                ).lower()
            return data

        @property
        def id(self) -> str:
            """Get command ID (alias for command_id)."""
            return self.command_id

        def get_command_type(self) -> str:
            """Get command type derived from class name."""
            name = self.__class__.__name__
            base = name.removesuffix("Command")
            s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    class CqrsQuery(Query):
        """CQRS query with derived type metadata for consistent routing."""

        @model_validator(mode="before")
        @classmethod
        def _ensure_query_type(cls, data: object) -> object:
            """Populate query_type based on class name if missing."""
            if not isinstance(data, dict):
                return data
            if "query_type" not in data or not data.get("query_type"):
                name = cls.__name__
                base = name.removesuffix("Query")
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                data["query_type"] = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
            return data

        @property
        def id(self) -> str:
            """Get query ID (alias for query_id)."""
            return self.query_id

    class CqrsConfig:
        """Configuration schemas backing the unified dispatcher stack."""

        model_config = ConfigDict(extra="ignore", validate_assignment=True)

        class Handler(BaseModel):
            """Configuration schema for CQRS handlers in the modernization plan."""

            handler_id: str = Field(..., min_length=1)
            handler_name: str = Field(..., min_length=1)
            handler_type: Literal["command", "query"] = Field(default="command")
            enabled: bool = Field(default=True)
            metadata: dict[str, object] = Field(default_factory=dict)

            model_config = ConfigDict(
                extra="ignore",
                validate_assignment=True,
                str_strip_whitespace=True,
            )

            @model_validator(mode="after")
            def _ensure_metadata(self) -> FlextModels.CqrsConfig.Handler:
                if not self.metadata:
                    object.__setattr__(self, "metadata", {})
                return self

        @staticmethod
        def create_handler_config(
            *,
            handler_type: Literal["command", "query"],
            default_name: str,
            default_id: str | None,
            handler_config: Handler | dict[str, object] | None,
            command_timeout: int = 0,
            max_command_retries: int = 0,
        ) -> Handler:
            """Build validated handler configuration with roadmap defaults."""
            if isinstance(handler_config, FlextModels.CqrsConfig.Handler):
                if handler_config.handler_type == handler_type:
                    return handler_config
                return handler_config.model_copy(update={"handler_type": handler_type})

            metadata_defaults: dict[str, object] = {
                "command_timeout": command_timeout,
                "max_command_retries": max_command_retries,
            }

            overrides: dict[str, object]
            overrides = dict(handler_config) if isinstance(handler_config, dict) else {}

            overrides.pop("handler_type", None)

            resolved_id = overrides.pop(
                "handler_id",
                default_id or FlextUtilities.Generators.generate_id(),
            )
            resolved_name = overrides.pop("handler_name", default_name)
            enabled_value = overrides.pop("enabled", True)

            metadata_override_raw = overrides.pop("metadata", {})
            metadata_override = (
                dict(metadata_override_raw)
                if isinstance(metadata_override_raw, dict)
                else {}
            )
            metadata_payload = {**metadata_defaults, **metadata_override}

            payload = {
                "handler_id": resolved_id,
                "handler_name": resolved_name,
                "handler_type": handler_type,
                "enabled": bool(enabled_value),
                "metadata": metadata_payload,
            }
            payload.update(overrides)

            return FlextModels.CqrsConfig.Handler.model_validate(payload)

        class Bus(BaseModel):
            """Configuration schema for the command bus pipeline used in 1.0.0."""

            enable_middleware: bool = Field(default=True)
            enable_metrics: bool = Field(default=True)
            enable_caching: bool = Field(default=True)
            execution_timeout: int = Field(
                default=30,
                ge=1,
                le=600,
                description="Middleware execution timeout in seconds",
            )
            max_cache_size: int = Field(
                default=1000,
                ge=10,
                le=100000,
                description="Maximum cached query results",
            )
            implementation_path: str = Field(
                default="flext_core.bus:FlextBus",
                description="Import path for command bus implementation (module:Class)",
                min_length=3,
            )

            model_config = ConfigDict(
                extra="ignore",
                validate_assignment=True,
            )

            @model_validator(mode="after")
            def _validate_path(self) -> FlextModels.CqrsConfig.Bus:
                if ":" not in self.implementation_path:
                    msg = "implementation_path must be in 'module:Class' format"
                    raise ValueError(msg)
                return self

        @staticmethod
        def create_bus_config(
            bus_config: FlextModels.CqrsConfig.Bus | dict[str, object] | None,
            *,
            enable_middleware: bool = True,
            enable_metrics: bool = True,
            enable_caching: bool = True,
            execution_timeout: int = 30,
            max_cache_size: int = 1000,
            implementation_path: str = "flext_core.bus:FlextBus",
        ) -> FlextModels.CqrsConfig.Bus:
            """Build validated bus configuration with modernization defaults."""
            if isinstance(bus_config, FlextModels.CqrsConfig.Bus):
                return bus_config

            defaults: dict[str, object] = {
                "enable_middleware": enable_middleware,
                "enable_metrics": enable_metrics,
                "enable_caching": enable_caching,
                "execution_timeout": execution_timeout,
                "max_cache_size": max_cache_size,
                "implementation_path": implementation_path,
            }

            overrides: dict[str, object]
            overrides = dict(bus_config) if isinstance(bus_config, dict) else {}

            payload = {**defaults, **overrides}
            return FlextModels.CqrsConfig.Bus.model_validate(payload)

    class EmailAddress(RootModel[str]):
        """Email address value object honoring modernization validation rules."""

        root: str

        @model_validator(mode="before")
        @classmethod
        def _coerce_input(cls, v: str | dict[str, str]) -> str:
            """Simplified input coercion with backward compatibility."""
            if isinstance(v, dict) and "value" in v and isinstance(v["value"], str):
                # Direct return for backward compatibility - no complex bypass mechanism
                return v["value"]
            if isinstance(v, str):
                return v
            return str(v)

        @model_validator(mode="after")
        def _validate_email(self) -> FlextModels.EmailAddress:
            """Email validation using FlextUtilities composition."""
            # Use composition pattern with _ValidationHelper
            self.root = FlextModels._ValidationHelper.validate_email_format(self.root)
            return self

        @property
        def value(self) -> str:
            """Return the email address value for backward compatibility."""
            return self.root

        @classmethod
        def create(cls, email: str) -> FlextResult[FlextModels.EmailAddress]:
            """Create factory with validation returning FlextResult."""
            try:
                return FlextResult["FlextModels.EmailAddress"].ok(cls(email))
            except Exception as e:
                return FlextResult["FlextModels.EmailAddress"].fail(str(e))

        def domain(self) -> str:
            """Get email domain."""
            return self.root.split("@")[1] if "@" in self.root else ""

    class Host(Value):
        """Host value object enforcing modernization-friendly hostname rules."""

        value: str

        @classmethod
        def create(cls, host: str) -> FlextResult[FlextModels.Host]:
            """Create host with validation."""
            if not host or not host.strip():
                return FlextResult["FlextModels.Host"].fail("Host cannot be empty")
            host = host.strip()
            # Check for spaces in hostname
            if " " in host:
                return FlextResult["FlextModels.Host"].fail(
                    "Host cannot contain spaces",
                )
            return FlextResult["FlextModels.Host"].ok(cls(value=host))

    class Timestamp(Value):
        """Timestamp value object ensuring UTC normalization for telemetry."""

        value: datetime

        @classmethod
        def create(cls, dt: datetime) -> FlextResult[FlextModels.Timestamp]:
            """Create timestamp with UTC conversion."""
            if dt.tzinfo is None:
                # Assume UTC if naive
                dt = dt.replace(tzinfo=None)
            return FlextResult["FlextModels.Timestamp"].ok(cls(value=dt))

    class EntityId(Value):
        """Entity identifier value object used across modernization APIs."""

        value: str

        @classmethod
        def create(cls, entity_id: str) -> FlextResult[FlextModels.EntityId]:
            """Create entity ID with validation and trimming."""
            if not entity_id or not entity_id.strip():
                return FlextResult["FlextModels.EntityId"].fail(
                    "Entity ID cannot be empty",
                )
            return FlextResult["FlextModels.EntityId"].ok(cls(value=entity_id.strip()))

    class JsonData(Value):
        """JSON payload value object with serialization guardrails."""

        value: dict[str, object]

        @classmethod
        def create(cls, data: dict[str, object]) -> FlextResult[FlextModels.JsonData]:
            """Create JSON data with validation."""
            try:
                # Validate that data is JSON serializable
                json.dumps(data)
                return FlextResult["FlextModels.JsonData"].ok(cls(value=data))
            except (TypeError, ValueError) as e:
                return FlextResult["FlextModels.JsonData"].fail(
                    f"Invalid JSON data: {e}",
                )

    class Metadata(Value):
        """Metadata map value object for dispatcher/context enrichment."""

        value: dict[str, str]

        @classmethod
        def create(cls, metadata: dict[str, str]) -> FlextResult[FlextModels.Metadata]:
            """Create metadata with validation."""
            # Ensure all values are strings
            invalid_keys = [
                key for key, value in metadata.items() if not isinstance(value, str)
            ]
            if invalid_keys:
                return FlextResult["FlextModels.Metadata"].fail(
                    f"Metadata values for keys {invalid_keys} must be strings",
                )
            return FlextResult["FlextModels.Metadata"].ok(cls(value=metadata))

    class Url(Value):
        """URL value object with HTTP-aware validation matched to docs."""

        value: str

        @classmethod
        def create(cls, url: str) -> FlextResult[FlextModels.Url]:
            """Create URL with basic validation."""
            if not url or not url.strip():
                return FlextResult["FlextModels.Url"].fail("URL cannot be empty")

            url = url.strip()
            if not url.startswith(("http://", "https://")):
                return FlextResult["FlextModels.Url"].fail(
                    "URL must start with http:// or https://",
                )

            if "://" in url and not url.split("://", 1)[1]:
                return FlextResult["FlextModels.Url"].fail(
                    "URL must have a valid hostname",
                )

            return FlextResult["FlextModels.Url"].ok(cls(value=url))

        @classmethod
        def create_http_url(
            cls,
            url: str,
            *,
            max_length: int = 2048,
            max_port: int = 65535,
        ) -> FlextResult[FlextModels.Url]:
            """Create URL with HTTP-specific validation using Pydantic v2 patterns.

            Args:
                url: URL string to validate
                max_length: Maximum URL length (default: 2048)
                max_port: Maximum port number (default: 65535)

            Returns:
                FlextResult containing validated URL or error message

            """
            # First use basic validation
            base_result = cls.create(url)
            if base_result.is_failure:
                return base_result

            # Additional HTTP-specific validation
            try:
                parsed = urlparse(url)

                # Port validation
                if parsed.port is not None:
                    port = parsed.port
                    if port == 0:
                        return FlextResult["FlextModels.Url"].fail("Invalid port 0")
                    if port > max_port:
                        return FlextResult["FlextModels.Url"].fail(
                            f"Invalid port {port}",
                        )

                # URL length validation
                if len(url) > max_length:
                    return FlextResult["FlextModels.Url"].fail("URL is too long")

            except Exception as e:
                return FlextResult["FlextModels.Url"].fail(f"URL parsing failed: {e}")

            return base_result

        def get_port(self) -> int | None:
            """Get port from URL using urlparse."""
            try:
                parsed = urlparse(self.value)
                return parsed.port
            except Exception:
                return None

        def get_scheme(self) -> str:
            """Get scheme from URL."""
            try:
                parsed = urlparse(self.value)
                return parsed.scheme or ""
            except Exception:
                return ""

        def get_hostname(self) -> str:
            """Get hostname from URL."""
            try:
                parsed = urlparse(self.value)
                return parsed.hostname or ""
            except Exception:
                return ""

        def normalize(self) -> FlextResult[FlextModels.Url]:
            """Normalize URL by removing trailing slash (except for scheme-only URLs)."""
            try:
                # Use FlextUtilities for text processing
                cleaned_result = FlextUtilities.TextProcessor.clean_text(self.value)
                if cleaned_result.is_failure:
                    return FlextResult["FlextModels.Url"].fail(
                        f"URL cleaning failed: {cleaned_result.error}"
                    )

                cleaned = cleaned_result.unwrap()
                if not cleaned:
                    return FlextResult["FlextModels.Url"].fail("URL cannot be empty")

                normalized = (
                    cleaned.rstrip("/") if not cleaned.endswith("://") else cleaned
                )
                return FlextResult["FlextModels.Url"].ok(
                    self.__class__(value=normalized),
                )
            except Exception as e:
                return FlextResult["FlextModels.Url"].fail(
                    f"URL normalization failed: {e}",
                )

    # AggregateRoot for compatibility - but SIMPLE
    class AggregateRoot(Entity):
        """Aggregate root with version-based concurrency control."""

        version: int = Field(default=1)
        aggregate_type: str = Field(default="")

        def apply_domain_event(self, event: FlextModels.Event) -> None:
            """Apply domain event to aggregate root."""
            # Add event to domain events list
            self.add_domain_event(event)
            # Increment version to reflect state change
            self.increment_version()

    # =========================================================================
    # CONFIGURATION CLASSES - Simple configuration models
    # =========================================================================

    class SystemConfigs:
        """System-wide configuration classes.

        DEPRECATED: Use FlextConfig.SystemConfigs instead.
        This alias will be removed in a future version.
        """

        def __getattr__(self, name: str) -> type:
            """Dynamic attribute access for backward compatibility."""
            valid_configs = {
                "ContainerConfig",
                "DatabaseConfig",
                "SecurityConfig",
                "LoggingConfig",
                "MiddlewareConfig",
            }

            if name in valid_configs:
                warnings.warn(
                    f"FlextModels.SystemConfigs.{name} is deprecated. "
                    f"Use FlextConfig.SystemConfigs.{name} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return cast("type", getattr(FlextConfig.SystemConfigs, name))

            error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(error_msg)

    # =========================================================================
    # SIMPLE CONFIG CLASSES - Direct access versions
    # =========================================================================

    # =========================================================================
    # EXAMPLE-SPECIFIC CLASSES - For examples and demos
    # =========================================================================

    class Config(BaseModel):
        """Simple configuration class for examples."""

        name: str = Field(default="", description="Configuration name")
        enabled: bool = Field(default=True, description="Whether enabled")
        settings: dict[str, object] = Field(
            default_factory=dict,
            description="Additional settings",
        )

    class Message(BaseModel):
        """Simple message class for examples."""

        message_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        content: str = Field(...)
        message_type: str = Field(default="info")
        priority: str = Field(default="normal")
        target_service: str = Field(default="")
        headers: dict[str, str] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=datetime.now)
        source_service: str = Field(default="")
        aggregate_id: str = Field(default="")
        aggregate_type: str = Field(default="")

    # =========================================================================
    # WORKSPACE MODELS - Consolidated from ecosystem projects
    # =========================================================================

    class Project(Value):
        """Project value object consolidated for modernization workspace scans."""

        name: str = Field(..., description="Project name")
        path: str = Field(..., description="Project path")
        project_type: str = Field(
            ...,
            description="Project type",
        )  # Use string to avoid enum import
        has_tests: bool = Field(default=False, description="Has test directory")
        has_pyproject: bool = Field(default=False, description="Has pyproject.toml")
        has_go_mod: bool = Field(default=False, description="Has go.mod file")
        test_count: int = Field(0, ge=0, description="Number of test files")

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate project business rules using consolidated patterns."""

            # Define business rule validators
            def validate_test_consistency(
                project: FlextModels.Project,
            ) -> FlextResult[None]:
                if project.has_tests and project.test_count <= 0:
                    return FlextResult[None].fail(
                        "test_count must be positive when has_tests is True"
                    )
                return FlextResult[None].ok(None)

            def validate_project_structure(
                project: FlextModels.Project,
            ) -> FlextResult[None]:
                if not project.has_pyproject and not project.has_go_mod:
                    return FlextResult[None].fail(
                        "project must have either pyproject.toml or go.mod"
                    )
                return FlextResult[None].ok(None)

            # Apply business rules using railway composition
            return validate_test_consistency(self) >> (
                lambda _: validate_project_structure(self)
            )

    class WorkspaceContext(Config):
        """Workspace context configuration reused across modernization tooling."""

        workspace_root: str = Field(..., description="Workspace root path")
        project_filter: str | None = Field(None, description="Project name filter")
        include_hidden: bool = Field(
            default=False,
            description="Include hidden directories",
        )
        max_depth: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Maximum directory depth",
        )

    class WorkspaceInfo(Value):
        """Workspace information object feeding modernization dashboards."""

        name: str = Field(..., description="Workspace name")
        path: str = Field(..., description="Workspace path")
        project_count: int = Field(default=0, ge=0, description="Number of projects")
        total_size_mb: float = Field(default=0.0, ge=0, description="Total size in MB")
        projects: list[str] | None = Field(
            default=None,
            description="List of project names",
        )
        status: str = Field(
            default="ready",
            description="Workspace status",
        )  # Use string to avoid enum import

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate workspace business rules using consolidated patterns."""

            # Define business rule validators
            def validate_workspace_consistency(
                workspace: FlextModels.WorkspaceInfo,
            ) -> FlextResult[None]:
                if workspace.projects and workspace.project_count != len(
                    workspace.projects
                ):
                    return FlextResult[None].fail(
                        "project_count must match length of projects list"
                    )
                return FlextResult[None].ok(None)

            def validate_workspace_size(
                workspace: FlextModels.WorkspaceInfo,
            ) -> FlextResult[None]:
                if workspace.total_size_mb < 0:
                    return FlextResult[None].fail("total_size_mb cannot be negative")
                return FlextResult[None].ok(None)

            # Apply business rules using railway composition
            return validate_workspace_consistency(self) >> (
                lambda _: validate_workspace_size(self)
            )

    # Simple factory methods - no over-engineering
    @staticmethod
    def create_entity(**data: object) -> FlextResult[FlextModels.Entity]:
        """Create an entity."""
        try:
            # Convert object values to appropriate types
            id_value = ""

            for key, value in data.items():
                if key == "id":
                    if isinstance(value, str):
                        id_value = value
                    elif value is not None:
                        id_value = str(value)

            entity = FlextModels.Entity(id=id_value)
            return FlextResult[FlextModels.Entity].ok(entity)
        except Exception as e:
            return FlextResult[FlextModels.Entity].fail(str(e))

    @staticmethod
    def create_event(
        event_type: str,
        payload: dict[str, object],
        aggregate_id: str,
    ) -> Event:
        """Create an event."""
        return FlextModels.Event(
            event_type=event_type,
            payload=payload,
            aggregate_id=aggregate_id,
        )

    @staticmethod
    def create_command(command_type: str, payload: dict[str, object]) -> Command:
        """Create a command."""
        return FlextModels.Command(command_type=command_type, payload=payload)

    @staticmethod
    def create_query(
        query_type: str,
        filters: dict[str, object] | None = None,
    ) -> Query:
        """Create a query."""
        return FlextModels.Query(query_type=query_type, filters=filters or {})

    # Internal helper to mark EmailAddress inputs that should bypass strict validation

    class _EmailBypassStr(UserString):
        __slots__ = ()

    # =========================================================================
    # FIELD VALIDATION FUNCTIONS - Replace FieldValidators patterns
    # =========================================================================

    @staticmethod
    def create_validated_email(email: str) -> FlextResult[str]:
        """Create validated email using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_email(email)

    @staticmethod
    def create_validated_url(url: str) -> FlextResult[str]:
        """Create validated URL using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_url(url)

    @staticmethod
    def create_validated_http_url(url: str, max_length: int = 2048) -> FlextResult[str]:
        """Create validated HTTP URL using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_string(
            url, min_length=8, max_length=max_length, field_name="URL"
        ) >> (FlextUtilities.Validation.validate_url)

    @staticmethod
    def create_validated_http_method(method: str) -> FlextResult[str]:
        """Create validated HTTP method using consolidated validation patterns."""
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if method.upper() in valid_methods:
            return FlextResult[str].ok(method.upper())
        return FlextResult[str].fail(f"Invalid HTTP method: {method}")

    @staticmethod
    def create_validated_http_status(code: int) -> FlextResult[int]:
        """Create validated HTTP status code using consolidated validation patterns."""
        if 100 <= code <= 599:
            return FlextResult[int].ok(code)
        return FlextResult[int].fail(f"Invalid HTTP status code: {code}")

    @staticmethod
    def create_validated_phone(phone: str) -> FlextResult[str]:
        """Create validated phone using consolidated validation patterns."""
        # Basic phone validation - can be enhanced later
        cleaned = "".join(filter(str.isdigit, phone))
        if len(cleaned) >= 10:
            return FlextResult[str].ok(phone)
        return FlextResult[str].fail(f"Invalid phone number: {phone}")

    @staticmethod
    def create_validated_uuid(uuid_str: str) -> FlextResult[str]:
        """Create validated UUID using consolidated validation patterns."""
        try:
            uuid.UUID(uuid_str)
            return FlextResult[str].ok(uuid_str)
        except ValueError:
            return FlextResult[str].fail(f"Invalid UUID: {uuid_str}")

    @staticmethod
    def create_validated_iso_date(date_str: str) -> FlextResult[str]:
        """Create validated ISO date using consolidated validation patterns."""
        try:
            datetime.fromisoformat(date_str)
            return FlextResult[str].ok(date_str)
        except ValueError:
            return FlextResult[str].fail(f"Invalid ISO date: {date_str}")

    @staticmethod
    def create_validated_date_range(
        start_date: str,
        end_date: str,
    ) -> FlextResult[tuple[str, str]]:
        """Create validated date range - centralizes date range validation logic.

        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            FlextResult containing validated date tuple or validation error

        """
        # Validate individual dates first
        start_result = FlextModels.create_validated_iso_date(start_date)
        if start_result.is_failure:
            return FlextResult[tuple[str, str]].fail(
                f"Invalid start date: {start_result.error}",
            )

        end_result = FlextModels.create_validated_iso_date(end_date)
        if end_result.is_failure:
            return FlextResult[tuple[str, str]].fail(
                f"Invalid end date: {end_result.error}",
            )

        # Validate date range
        try:
            start_dt = datetime.fromisoformat(start_date.strip())
            end_dt = datetime.fromisoformat(end_date.strip())

            if start_dt >= end_dt:
                return FlextResult[tuple[str, str]].fail(
                    "Start date must be before end date",
                )

            return FlextResult[tuple[str, str]].ok(
                (start_date.strip(), end_date.strip()),
            )
        except ValueError as e:
            return FlextResult[tuple[str, str]].fail(
                f"Date range validation failed: {e}",
            )

    @staticmethod
    def create_validated_file_path(file_path: str) -> FlextResult[str]:
        """Create validated file path using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_file_path(file_path)

    @staticmethod
    def create_validated_existing_file_path(file_path: str) -> FlextResult[str]:
        """Create validated existing file path using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_existing_file_path(file_path)

    @staticmethod
    def create_validated_directory_path(dir_path: str) -> FlextResult[str]:
        """Create validated directory path using consolidated validation patterns."""
        return FlextUtilities.Validation.validate_directory_path(dir_path)

    # HTTP status code constants
    # =========================================================================
    # FLEXT CONTAINER PYDANTIC MODELS (OPTIMIZATION)
    # =========================================================================

    class ServiceRegistrationModel(BaseModel):
        """Pydantic model for service registration validation."""

        name: str = Field(min_length=1, description="Service name identifier")
        service: object = Field(description="Service instance to register")

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class FactoryRegistrationModel(BaseModel):
        """Pydantic model for factory registration validation."""

        name: str = Field(min_length=1, description="Factory name identifier")
        factory: Callable[[], object] = Field(description="Parameterless factory function")

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("factory")
        @classmethod
        def validate_factory_signature(cls, v: Callable[[], object]) -> Callable[[], object]:
            """Validate factory has no required parameters."""
            import inspect

            sig = inspect.signature(v)
            required_params = sum(
                1
                for p in sig.parameters.values()
                if p.default == p.empty
                and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
            )
            if required_params > 0:
                msg = f"Factory requires {required_params} parameter(s), must be parameterless"
                raise ValueError(msg)
            return v

    class ServiceRetrievalModel(BaseModel):
        """Pydantic model for service retrieval validation."""

        name: str = Field(min_length=1, description="Service name to retrieve")
        expected_type: type | None = Field(
            default=None, description="Expected service type"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class BatchRegistrationModel(BaseModel):
        """Pydantic model for batch registration validation."""

        registrations: dict[str, object] = Field(
            description="Dictionary of name->service/factory mappings"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("registrations")
        @classmethod
        def validate_non_empty(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate registrations dictionary is not empty."""
            if not v:
                msg = "Registrations dictionary cannot be empty"
                raise ValueError(msg)
            return v

    # =========================================================================
    # PROCESSING MODELS - For FlextProcessing optimization
    # =========================================================================

    class ProcessingRequest(BaseModel):
        """Pydantic model for processing requests in FlextProcessing."""

        data: object = Field(description="Data to be processed")
        context: str = Field(
            default="",
            max_length=500,
            description="Processing context for error messages",
        )
        timeout_seconds: float = Field(
            default=30.0, ge=0.1, le=3600.0, description="Processing timeout in seconds"
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Additional processing metadata"
        )
        enable_validation: bool = Field(
            default=True, description="Enable input validation"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: str) -> str:
            """Validate context using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=0, max_length=500, field_name="context"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

        @field_validator("timeout_seconds")
        @classmethod
        def validate_timeout(cls, v: float) -> float:
            """Validate timeout is within reasonable bounds."""
            if not 0.1 <= v <= 3600.0:
                msg = "timeout_seconds must be between 0.1 and 3600.0 seconds"
                raise ValueError(msg)
            return v

    class HandlerRegistration(BaseModel):
        """Pydantic model for handler registration in FlextProcessing."""

        name: str = Field(min_length=1, description="Handler name identifier")
        handler: object = Field(description="Handler instance or callable")
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Handler metadata"
        )
        enabled: bool = Field(default=True, description="Whether handler is enabled")

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Validate handler name using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="handler name"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

        @field_validator("handler")
        @classmethod
        def validate_handler(cls, v: object) -> object:
            """Validate handler is callable or has handle method."""
            if not (callable(v) or hasattr(v, "handle")):
                msg = "Handler must be callable or have a 'handle' method"
                raise ValueError(msg)
            return v

    class BatchProcessingConfig(BaseModel):
        """Pydantic model for batch processing configuration in FlextProcessing."""

        data_items: list[object] = Field(description="List of data items to process")
        fail_fast: bool = Field(
            default=True, description="Stop processing on first failure"
        )
        max_workers: int = Field(
            default=4, ge=1, le=32, description="Maximum worker threads"
        )
        timeout_seconds: float = Field(
            default=300.0, ge=1.0, le=3600.0, description="Total batch timeout"
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Batch processing metadata"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("data_items")
        @classmethod
        def validate_data_items(cls, v: list[object]) -> list[object]:
            """Validate data items list is not empty."""
            if not v:
                msg = "data_items cannot be empty"
                raise ValueError(msg)
            return v

        @field_validator("max_workers")
        @classmethod
        def validate_max_workers(cls, v: int) -> int:
            """Validate max_workers is within reasonable bounds."""
            if not 1 <= v <= 32:
                msg = "max_workers must be between 1 and 32"
                raise ValueError(msg)
            return v

        def validate_batch(self) -> FlextResult[None]:
            """Validate batch configuration using business rules."""
            try:
                # Pydantic validation already done during model creation
                if not self.data_items:
                    return FlextResult[None].fail("Batch cannot be empty")

                if len(self.data_items) > 1000:
                    return FlextResult[None].fail("Batch size cannot exceed 1000 items")

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Batch validation failed: {e}")

    class HandlerExecutionConfig(BaseModel):
        """Pydantic model for handler execution configuration."""

        handler_name: str = Field(min_length=1, description="Handler name to execute")
        request_data: object = Field(description="Request data to process")
        timeout_seconds: float = Field(
            default=30.0, ge=0.1, le=300.0, description="Execution timeout"
        )
        retry_count: int = Field(
            default=0, ge=0, le=5, description="Number of retries on failure"
        )
        fallback_handlers: list[str] = Field(
            default_factory=list, description="Fallback handler names"
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Execution metadata"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("handler_name")
        @classmethod
        def validate_handler_name(cls, v: str) -> str:
            """Validate handler name using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="handler_name"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

    class PipelineConfiguration(BaseModel):
        """Pydantic model for pipeline configuration in FlextProcessing."""

        name: str = Field(min_length=1, description="Pipeline name")
        steps: list[object] = Field(description="Pipeline processing steps")
        enable_parallel: bool = Field(
            default=False, description="Enable parallel step execution"
        )
        fail_fast: bool = Field(default=True, description="Stop on first step failure")
        timeout_seconds: float = Field(
            default=300.0, ge=1.0, le=3600.0, description="Pipeline timeout"
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Pipeline metadata"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Validate pipeline name using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="pipeline name"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.value

        @field_validator("steps")
        @classmethod
        def validate_steps(cls, v: list[object]) -> list[object]:
            """Validate pipeline has at least one step."""
            if not v:
                msg = "Pipeline must have at least one step"
                raise ValueError(msg)
            return v

    class ProcessingResult(BaseModel):
        """Pydantic model for processing results."""

        success: bool = Field(description="Whether processing succeeded")
        data: object = Field(default=None, description="Processing result data")
        error: str = Field(default="", description="Error message if failed")
        execution_time_ms: float = Field(
            default=0.0, ge=0.0, description="Execution time in milliseconds"
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Result metadata"
        )
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(UTC), description="Result timestamp"
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

        @field_validator("execution_time_ms")
        @classmethod
        def validate_execution_time(cls, v: float) -> float:
            """Validate execution time is non-negative."""
            if v < 0.0:
                msg = "execution_time_ms cannot be negative"
                raise ValueError(msg)
            return v

    class AutoWireModel(BaseModel):
        """Pydantic model for auto-wire validation."""

        service_class: type = Field(description="Service class to auto-wire")
        service_name: str = Field(min_length=1, description="Service registration name")

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class ContainerConfigModel(BaseModel):
        """Pydantic model for container configuration validation.

        Uses FlextConfig as primary source of truth for defaults with FlextConstants as fallback.
        """

        max_workers: int = Field(
            default_factory=lambda: _get_global_config().max_workers,
            ge=1,
            description="Maximum worker threads from FlextConfig",
        )
        timeout_seconds: float = Field(
            default_factory=lambda: _get_global_config().timeout_seconds,
            ge=0.1,
            description="Operation timeout in seconds from FlextConfig",
        )
        environment: str = Field(
            default_factory=lambda: _get_global_config().environment,
            description="Runtime environment from FlextConfig",
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    # === FlextLogger Optimization Models ===
    # These models consolidate FlextLogger method parameters for optimization

    class LoggerConfigurationModel(BaseModel):
        """Pydantic model for FlextLogger.configure() method parameter consolidation.
        
        Consolidates 5 parameters (log_level, json_output, include_source, 
        structured_output, log_verbosity) into a validated model using FlextConfig 
        and FlextConstants as source of truth.
        """

        log_level: str = Field(
            default_factory=lambda: _get_global_config().log_level,
            description="Logging level from FlextConfig",
        )
        json_output: bool | None = Field(
            default=None,
            description="Use JSON output format (auto-detected if None)",
        )
        include_source: bool = Field(
            default_factory=lambda: FlextConstants.Logging.INCLUDE_SOURCE,
            description="Include source code location info",
        )
        structured_output: bool = Field(
            default_factory=lambda: FlextConstants.Logging.STRUCTURED_OUTPUT,
            description="Use structured logging format",
        )
        log_verbosity: str = Field(
            default_factory=lambda: FlextConstants.Logging.VERBOSITY,
            pattern="^(compact|detailed|full)$",
            description="Console output verbosity level",
        )

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class LoggerInitializationModel(BaseModel):
        """Pydantic model for FlextLogger.__init__() method parameter consolidation.
        
        Consolidates 5 parameters (name, level, service_name, service_version, 
        correlation_id) into a validated model using FlextConfig defaults.
        """

        name: str = Field(
            min_length=1,
            description="Logger name identifier",
        )
        level: str | None = Field(
            default=None,
            description="Log level override (None uses FlextConfig defaults)",
        )
        service_name: str | None = Field(
            default=None,
            description="Service name (auto-detected if None)",
        )
        service_version: str | None = Field(
            default=None,
            description="Service version (auto-detected if None)",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Correlation ID (auto-generated if None)",
        )

        @field_validator("level")
        @classmethod
        def validate_log_level(cls, v: str | None) -> str | None:
            """Validate log level is in allowed values."""
            if v is None:
                return v
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v.upper() not in valid_levels:
                raise ValueError(f"Log level must be one of: {valid_levels}")
            return v.upper()

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class LogContextModel(BaseModel):
        """Pydantic model for log context binding and management.
        
        Consolidates context-related operations for bind(), set_context(),
        with_context() methods.
        """

        context_data: dict[str, object] = Field(
            default_factory=dict,
            description="Context data to bind to logger",
        )
        persistent: bool = Field(
            default=False,
            description="Whether context should persist across log calls",
        )
        merge_strategy: str = Field(
            default="update",
            pattern="^(update|replace|merge_deep)$",
            description="How to handle existing context data",
        )

        @field_validator("context_data")
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data is serializable."""
            if not isinstance(v, dict):
                raise ValueError("Context data must be a dictionary")
            # Basic serialization check - avoid complex objects
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Context keys must be strings")
                if callable(value) or hasattr(value, "__dict__"):
                    raise ValueError(f"Context value for '{key}' must be serializable")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class LoggerContextBindingModel(BaseModel):
        """Pydantic model for logger context binding operations."""

        source_logger_name: str = Field(
            description="Name of the source logger to bind from"
        )
        context_data: dict[str, object] = Field(
            default_factory=dict,
            description="Context data to bind to new logger instance"
        )
        force_new_instance: bool = Field(
            default=True,
            description="Whether to force creation of new logger instance"
        )
        copy_request_context: bool = Field(
            default=True,
            description="Whether to copy existing request context"
        )
        copy_permanent_context: bool = Field(
            default=True,
            description="Whether to copy existing permanent context"
        )

        @field_validator("context_data")
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data is serializable."""
            if not isinstance(v, dict):
                raise ValueError("Context data must be a dictionary")
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Context keys must be strings")
                if callable(value) or (hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict))):
                    raise ValueError(f"Context value for '{key}' must be serializable")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class LoggerRequestContextModel(BaseModel):
        """Pydantic model for logger request-specific context management."""

        request_context: dict[str, object] = Field(
            default_factory=dict,
            description="Request-specific context data"
        )
        correlation_id: str | None = Field(
            default=None,
            description="Request correlation ID for tracing"
        )
        clear_existing: bool = Field(
            default=False,
            description="Whether to clear existing request context before setting"
        )

        @field_validator("request_context")
        @classmethod
        def validate_request_context(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate request context data."""
            if not isinstance(v, dict):
                raise ValueError("Request context must be a dictionary")
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Request context keys must be strings")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class LoggerPermanentContextModel(BaseModel):
        """Pydantic model for logger permanent context management."""

        permanent_context: dict[str, object] = Field(
            default_factory=dict,
            description="Permanent context data that persists across requests"
        )
        replace_existing: bool = Field(
            default=False,
            description="Whether to replace existing permanent context completely"
        )
        merge_strategy: str = Field(
            default="update",
            pattern="^(update|replace|merge_deep)$",
            description="Strategy for merging with existing permanent context"
        )

        @field_validator("permanent_context")
        @classmethod
        def validate_permanent_context(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate permanent context data."""
            if not isinstance(v, dict):
                raise ValueError("Permanent context must be a dictionary")
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Permanent context keys must be strings")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class OperationTrackingModel(BaseModel):
        """Pydantic model for operation tracking in FlextLogger.
        
        Consolidates start_operation() and complete_operation() method parameters
        for tracking long-running operations.
        """

        operation_name: str = Field(
            min_length=1,
            description="Name of the operation being tracked",
        )
        operation_id: str | None = Field(
            default=None,
            description="Unique operation identifier (auto-generated if None)",
        )
        metadata: dict[str, object] = Field(
            default_factory=dict,
            description="Additional operation metadata",
        )
        include_performance: bool = Field(
            default_factory=lambda: FlextConstants.Logging.TRACK_PERFORMANCE,
            description="Include performance metrics",
        )
        timeout_seconds: float | None = Field(
            default=None,
            ge=0.1,
            description="Operation timeout for monitoring",
        )

        @field_validator("metadata")
        @classmethod
        def validate_metadata(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate metadata is serializable."""
            if not isinstance(v, dict):
                raise ValueError("Metadata must be a dictionary")
            # Basic serialization check
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError("Metadata keys must be strings")
                if callable(value):
                    raise ValueError(f"Metadata value for '{key}' cannot be callable")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class PerformanceTrackingModel(BaseModel):
        """Pydantic model for performance tracking in FlextLogger.
        
        Consolidates performance-related logging configuration and context
        for operation timing and resource monitoring.
        """

        track_memory: bool = Field(
            default_factory=lambda: FlextConstants.Logging.TRACK_MEMORY,
            description="Track memory usage during operations",
        )
        track_timing: bool = Field(
            default_factory=lambda: FlextConstants.Logging.TRACK_TIMING,
            description="Track execution timing",
        )
        threshold_warning_ms: float = Field(
            default_factory=lambda: FlextConstants.Logging.PERFORMANCE_THRESHOLD_WARNING,
            ge=0,
            description="Warning threshold in milliseconds",
        )
        threshold_critical_ms: float = Field(
            default_factory=lambda: FlextConstants.Logging.PERFORMANCE_THRESHOLD_CRITICAL,
            ge=0,
            description="Critical threshold in milliseconds",
        )
        sampling_rate: float = Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Performance sampling rate (0.0-1.0)",
        )

        @field_validator("threshold_critical_ms")
        @classmethod
        def validate_critical_threshold(cls, v: float, info: ValidationInfo) -> float:
            """Ensure critical threshold is higher than warning threshold."""
            if "threshold_warning_ms" in info.data:
                warning = info.data["threshold_warning_ms"]
                if v <= warning:
                    raise ValueError("Critical threshold must be higher than warning threshold")
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    # =============================================================================
    # DOMAIN SERVICE MODELS - For FlextDomainService optimization
    # =============================================================================

    class DomainServiceExecutionRequest(BaseModel):
        """Pydantic model for domain service execution parameters.

        Reduces method parameters by consolidating execution context,
        timeout settings, retry configuration, and metadata into a single model.
        """

        context: str = Field(
            default="",
            description="Execution context for enhanced error messaging",
            max_length=500,
        )
        timeout_seconds: float = Field(
            default_factory=lambda: _get_global_config().timeout_seconds,
            ge=0.1,
            le=3600.0,
            description="Execution timeout in seconds from FlextConfig",
        )
        retry_count: int = Field(
            default=3, ge=0, le=10, description="Maximum number of retry attempts"
        )
        enable_metrics: bool = Field(
            default_factory=lambda: _get_global_config().enable_metrics,
            description="Enable performance metrics collection from FlextConfig",
        )
        enable_circuit_breaker: bool = Field(
            default_factory=lambda: _get_global_config().enable_circuit_breaker,
            description="Enable circuit breaker pattern from FlextConfig",
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Additional execution metadata"
        )

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: str) -> str:
            """Validate context using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=0, max_length=500, field_name="context"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap()

        @field_validator("timeout_seconds")
        @classmethod
        def validate_timeout(cls, v: float) -> float:
            """Validate timeout using FlextUtilities."""
            if v <= 0:
                msg = "Timeout must be positive"
                raise ValueError(msg)
            return v

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
        )

    class DomainServiceValidationRequest(BaseModel):
        """Pydantic model for domain service validation parameters.

        Consolidates validation configuration using FlextConfig as source of truth
        for strict mode, business rules, and validation behavior.
        """

        enable_strict_mode: bool = Field(
            default_factory=lambda: _get_global_config().validation_strict_mode,
            description="Enable strict validation mode from FlextConfig",
        )
        validate_business_rules: bool = Field(
            default=True, description="Enable business rule validation"
        )
        validate_config: bool = Field(
            default=True, description="Enable configuration validation"
        )
        validation_rules: list[str] = Field(
            default_factory=list, description="Additional validation rules to apply"
        )
        fail_fast: bool = Field(
            default=True, description="Stop validation on first failure"
        )

        @field_validator("validation_rules")
        @classmethod
        def validate_rules(cls, v: list[str]) -> list[str]:
            """Validate validation rules using FlextUtilities."""
            for rule in v:
                result = FlextUtilities.Validation.validate_string(
                    rule, min_length=1, max_length=100, field_name="validation_rule"
                )
                if result.is_failure:
                    msg = f"Invalid validation rule: {result.error}"
                    raise ValueError(msg)
            return v

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, extra="forbid"
        )

    class DomainServiceBatchRequest(BaseModel):
        """Pydantic model for batch domain service operations.

        Consolidates batch processing parameters with proper configuration
        integration and validation using Pydantic 2 patterns.
        """

        operations: list[Callable[[], FlextResult[object]]] = Field(
            description="List of operations to execute in batch"
        )
        max_workers: int = Field(
            default_factory=lambda: _get_global_config().max_workers,
            ge=1,
            le=50,
            description="Maximum concurrent workers from FlextConfig",
        )
        timeout_per_operation: float = Field(
            default_factory=lambda: _get_global_config().timeout_seconds,
            ge=0.1,
            le=3600.0,
            description="Timeout per operation from FlextConfig",
        )
        fail_fast: bool = Field(
            default=True, description="Stop batch on first operation failure"
        )
        collect_metrics: bool = Field(
            default_factory=lambda: _get_global_config().enable_metrics,
            description="Collect batch execution metrics from FlextConfig",
        )

        @field_validator("operations")
        @classmethod
        def validate_operations(
            cls, v: list[Callable[[], FlextResult[object]]]
        ) -> list[Callable[[], FlextResult[object]]]:
            """Validate operations list."""
            if not v:
                msg = "Operations list cannot be empty"
                raise ValueError(msg)
            if len(v) > 100:
                msg = "Too many operations (max 100)"
                raise ValueError(msg)
            return v

        model_config = ConfigDict(
            validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
        )

    class DomainServiceMetricsRequest(BaseModel):
        """Pydantic model for domain service metrics collection.

        Consolidates metrics configuration with proper defaults from FlextConfig
        and FlextConstants integration.
        """

        collect_execution_time: bool = Field(
            default_factory=lambda: _get_global_config().enable_metrics,
            description="Collect execution time metrics from FlextConfig",
        )
        collect_memory_usage: bool = Field(
            default=False, description="Collect memory usage metrics"
        )
        collect_validation_metrics: bool = Field(
            default=True, description="Collect validation performance metrics"
        )
        metrics_prefix: str = Field(
            default="domain_service",
            min_length=1,
            max_length=50,
            description="Prefix for metric names",
        )
        custom_tags: dict[str, str] = Field(
            default_factory=dict, description="Custom tags for metrics"
        )

        @field_validator("metrics_prefix")
        @classmethod
        def validate_prefix(cls, v: str) -> str:
            """Validate metrics prefix using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, max_length=50, field_name="metrics_prefix"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap()

        model_config = ConfigDict(validate_assignment=True, extra="forbid")

    class DomainServiceResourceRequest(BaseModel):
        """Pydantic model for domain service resource management.

        Consolidates resource management parameters with proper validation
        and configuration integration for resource lifecycle management.
        """

        resource_type: str = Field(
            description="Type of resource to manage", min_length=1, max_length=100
        )
        resource_config: dict[str, object] = Field(
            default_factory=dict, description="Resource-specific configuration"
        )
        cleanup_timeout: float = Field(
            default_factory=lambda: _get_global_config().timeout_seconds,
            ge=0.1,
            le=300.0,
            description="Cleanup timeout in seconds from FlextConfig",
        )
        auto_cleanup: bool = Field(
            default=True, description="Enable automatic resource cleanup"
        )

        @field_validator("resource_type")
        @classmethod
        def validate_resource_type(cls, v: str) -> str:
            """Validate resource type using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, max_length=100, field_name="resource_type"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap()

        model_config = ConfigDict(
            validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
        )

    # =========================================================================
    # DOMAIN SERVICE MODELS - For FlextDomainService optimization
    # =========================================================================

    class OperationExecutionRequest(BaseModel):
        """Pydantic model for operation execution in FlextDomainService.

        USAGE: Replaces multi-parameter execute_operation() method with single model.
        OPTIMIZATION: Consolidates operation_name, operation, args, kwargs into validated model.
        """

        operation_name: str = Field(
            min_length=1,
            max_length=255,
            description="Name of the operation to execute",
            examples=["validate_user", "process_data", "send_notification"],
        )
        operation: Callable[..., object] = Field(
            description="Callable operation to execute"
        )
        args: list[object] = Field(
            default_factory=list, description="Positional arguments for the operation"
        )
        kwargs: dict[str, object] = Field(
            default_factory=dict, description="Keyword arguments for the operation"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        @field_validator("operation_name")
        @classmethod
        def validate_operation_name(cls, v: str) -> str:
            """Validate operation name using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="operation_name"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap().strip()

        @field_validator("operation")
        @classmethod
        def validate_operation_callable(
            cls, v: Callable[..., object]
        ) -> Callable[..., object]:
            """Validate that operation is callable."""
            if not callable(v):
                msg = "Operation must be callable"
                raise ValueError(msg)
            return v

    class RetryConfiguration(BaseModel):
        """Pydantic model for retry configuration in FlextDomainService.

        USAGE: Replaces manual retry parameters with validated configuration model.
        OPTIMIZATION: Uses FlextConstants for default values, validates parameters.
        """

        max_attempts: int = Field(
            default=FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
            ge=1,
            le=20,
            description="Maximum number of retry attempts",
        )
        backoff_strategy: str = Field(
            default=FlextConstants.Reliability.DEFAULT_BACKOFF_STRATEGY,
            description="Backoff strategy for retries",
        )
        initial_delay_seconds: float = Field(
            default=1.0,
            ge=0.1,
            le=60.0,
            description="Initial delay before first retry in seconds",
        )
        max_delay_seconds: float = Field(
            default=30.0,
            ge=1.0,
            le=300.0,
            description="Maximum delay between retries in seconds",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

        @field_validator("backoff_strategy")
        @classmethod
        def validate_backoff_strategy(cls, v: str) -> str:
            """Validate backoff strategy value."""
            valid_strategies = {"exponential", "linear", "fixed", "random"}
            if v.lower() not in valid_strategies:
                msg = f"Backoff strategy must be one of: {valid_strategies}"
                raise ValueError(msg)
            return v.lower()

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate that max_delay >= initial_delay."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class CircuitBreakerConfiguration(BaseModel):
        """Pydantic model for circuit breaker configuration in FlextDomainService.

        USAGE: Replaces manual circuit breaker parameters with validated configuration.
        OPTIMIZATION: Uses FlextConstants for defaults, validates thresholds.
        """

        failure_threshold: int = Field(
            default=FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD,
            ge=1,
            le=100,
            description="Number of failures before opening circuit",
        )
        recovery_timeout: float = Field(
            default=FlextConstants.Reliability.DEFAULT_RECOVERY_TIMEOUT,
            gt=0,
            le=3600.0,
            description="Timeout in seconds before attempting recovery",
        )
        half_open_max_calls: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Maximum calls allowed in half-open state",
        )
        success_threshold: int = Field(
            default=2,
            ge=1,
            le=10,
            description="Successes needed in half-open state to close circuit",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

        @model_validator(mode="after")
        def validate_circuit_breaker_consistency(self) -> Self:
            """Validate circuit breaker parameter consistency."""
            if self.success_threshold > self.half_open_max_calls:
                msg = "success_threshold cannot exceed half_open_max_calls"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(BaseModel):
        """Pydantic model for validation configuration in FlextDomainService.

        USAGE: Replaces manual validation parameters with structured configuration.
        OPTIMIZATION: Provides type-safe validation configuration.
        """

        config_validation: bool = Field(
            default=True, description="Enable configuration validation"
        )
        business_rules_validation: bool = Field(
            default=True, description="Enable business rules validation"
        )
        fail_fast: bool = Field(
            default=True, description="Stop on first validation failure"
        )
        additional_validators: list[Callable[[], FlextResult[None]]] = Field(
            default_factory=list,
            description="Additional validation functions to execute",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

        @field_validator("additional_validators")
        @classmethod
        def validate_additional_validators(
            cls, v: list[Callable[[], FlextResult[None]]]
        ) -> list[Callable[[], FlextResult[None]]]:
            """Validate that additional validators are callable."""
            for i, validator in enumerate(v):
                if not callable(validator):
                    msg = f"Validator at index {i} must be callable"
                    raise ValueError(msg)
            return v

    class ServiceExecutionContext(BaseModel):
        """Pydantic model for service execution context in FlextDomainService.

        USAGE: Replaces string context parameter with structured context model.
        OPTIMIZATION: Provides rich context information for error handling.
        """

        context_name: str = Field(
            min_length=1,
            max_length=255,
            description="Name of the execution context",
            examples=["user_registration", "payment_processing", "data_sync"],
        )
        service_name: str = Field(
            default="", description="Name of the service executing the operation"
        )
        correlation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id,
            description="Correlation ID for tracing",
        )
        additional_metadata: dict[str, object] = Field(
            default_factory=dict, description="Additional context metadata"
        )
        started_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Context start timestamp",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        @field_validator("context_name")
        @classmethod
        def validate_context_name(cls, v: str) -> str:
            """Validate context name using FlextUtilities."""
            result = FlextUtilities.Validation.validate_string(
                v, min_length=1, field_name="context_name"
            )
            if result.is_failure:
                raise ValueError(result.error)
            return result.unwrap().strip()

        @field_validator("correlation_id")
        @classmethod
        def validate_correlation_id(cls, v: str) -> str:
            """Validate correlation ID format."""
            if not v or not v.strip():
                msg = "Correlation ID cannot be empty"
                raise ValueError(msg)
            return v.strip()

    # =============================================================================
    # ADDITIONAL DOMAIN SERVICE MODELS - Phase 5 Optimization
    # =============================================================================

    class ConditionalExecutionRequest(BaseModel):
        """Pydantic model for conditional execution parameters."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
        )

        condition: Callable[[], bool] = Field(
            ..., description="Condition function to check before execution"
        )
        fallback_result: object = Field(
            ..., description="Result to return if condition fails"
        )

        @field_validator("condition")
        @classmethod
        def validate_condition(cls, v: Callable[[], bool]) -> Callable[[], bool]:
            """Validate condition is callable."""
            if not callable(v):
                msg = "Condition must be callable"
                raise ValueError(msg)
            return v

    class StateMachineRequest(BaseModel):
        """Pydantic model for state machine execution parameters."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
        )

        initial_state: str = Field(
            ..., min_length=1, description="Initial state for state machine"
        )
        transitions: dict[str, Callable[[str], str]] = Field(
            ..., description="State transition functions mapped by transition name"
        )

        @field_validator("transitions")
        @classmethod
        def validate_transitions(
            cls, v: dict[str, Callable[[str], str]]
        ) -> dict[str, Callable[[str], str]]:
            """Validate all transitions are callable."""
            for name, transition in v.items():
                if not callable(transition):
                    msg = f"Transition '{name}' must be callable"
                    raise ValueError(msg)
            return v

    class ResourceManagementRequest(BaseModel):
        """Pydantic model for resource management execution parameters."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
        )

        resource_manager: Callable[[], object] = Field(
            ..., description="Resource manager function for resource allocation"
        )
        cleanup_on_error: bool = Field(
            default=True, description="Whether to cleanup resources on execution error"
        )

        @field_validator("resource_manager")
        @classmethod
        def validate_resource_manager(cls, v: Callable[[], object]) -> Callable[[], object]:
            """Validate resource manager is callable."""
            if not callable(v):
                msg = "Resource manager must be callable"
                raise ValueError(msg)
            return v

    class MetricsCollectionRequest(BaseModel):
        """Pydantic model for metrics collection execution parameters."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
        )

        metrics_collector: Callable[[str, float], None] | None = Field(
            default=None, description="Optional metrics collection function"
        )
        include_execution_time: bool = Field(
            default=True, description="Whether to include execution time in metrics"
        )
        include_memory_usage: bool = Field(
            default=False, description="Whether to include memory usage in metrics"
        )
        custom_labels: dict[str, str] = Field(
            default_factory=dict, description="Custom labels to include with metrics"
        )

        @field_validator("metrics_collector")
        @classmethod
        def validate_metrics_collector(
            cls, v: Callable[[str, float], None] | None
        ) -> Callable[[str, float], None] | None:
            """Validate metrics collector is callable when provided."""
            if v is not None and not callable(v):
                msg = "Metrics collector must be callable when provided"
                raise ValueError(msg)
            return v

    class TransformationRequest(BaseModel):
        """Pydantic model for validation and transformation parameters."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
            arbitrary_types_allowed=True,
        )

        transformer: Callable[[object], object] = Field(
            ..., description="Transformation function to apply to result"
        )
        validate_before_transform: bool = Field(
            default=True, description="Whether to validate before transformation"
        )
        transform_on_failure: bool = Field(
            default=False,
            description="Whether to apply transformation even on execution failure",
        )

        @field_validator("transformer")
        @classmethod
        def validate_transformer(cls, v: Callable[[object], object]) -> Callable[[object], object]:
            """Validate transformer is callable."""
            if not callable(v):
                msg = "Transformer must be callable"
                raise ValueError(msg)
            return v

    class FallbackConfiguration(BaseModel):
        """Pydantic model for fallback service configuration in FlextDomainService.

        USAGE: Replaces *fallback_services parameter with structured configuration.
        OPTIMIZATION: Provides type-safe fallback handling.
        """

        fallback_services: list[Callable[[], FlextResult[object]]] = Field(
            description="List of fallback service functions"
        )
        max_fallback_attempts: int = Field(
            default=3, ge=1, le=10, description="Maximum number of fallback attempts"
        )
        fallback_timeout_seconds: float = Field(
            default=10.0,
            gt=0,
            le=300.0,
            description="Timeout for each fallback attempt",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

        @field_validator("fallback_services")
        @classmethod
        def validate_fallback_services(
            cls, v: list[Callable[[], FlextResult[object]]]
        ) -> list[Callable[[], FlextResult[object]]]:
            """Validate that fallback services are provided."""
            if not v:
                msg = "At least one fallback service must be provided"
                raise ValueError(msg)
            return v

        @model_validator(mode="after")
        def validate_fallback_limits(self) -> Self:
            """Validate fallback configuration limits."""
            if len(self.fallback_services) > self.max_fallback_attempts:
                msg = f"Number of fallback services ({len(self.fallback_services)}) exceeds max_fallback_attempts ({self.max_fallback_attempts})"
                raise ValueError(msg)
            return self

    # =============================================================================
    # DISPATCHER MODELS - For FlextDispatcher optimization
    # =============================================================================

    class DispatcherConfiguration(Config):
        """Pydantic model for FlextDispatcher configuration.

        USAGE: Replaces individual constructor parameters with structured configuration.
        OPTIMIZATION: Provides type-safe dispatcher configuration with FlextConfig defaults.
        """

        auto_context: bool = Field(
            default=True,
            description="Enable automatic context management during dispatch",
        )
        bus_config: FlextModels.CqrsConfig.Bus | None = Field(
            default=None, description="Optional bus configuration override"
        )
        timeout_seconds: int = Field(
            default_factory=lambda: _get_global_config().timeout_seconds,
            ge=1,
            le=600,
            description="Operation timeout in seconds from FlextConfig",
        )
        enable_metrics: bool = Field(
            default_factory=lambda: _get_global_config().enable_metrics,
            description="Enable metrics collection from FlextConfig",
        )
        enable_logging: bool = Field(
            default=True, description="Enable dispatcher operation logging"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

    class HandlerRegistrationRequest(BaseModel):
        """Pydantic model for handler registration requests.

        USAGE: Replaces multiple registration parameters with structured request.
        OPTIMIZATION: Provides type-safe registration with validation.
        """

        handler: object = Field(..., description="Handler instance to register")
        message_type: type | None = Field(
            default=None, description="Message type for typed registration"
        )
        handler_mode: Literal["command", "query"] = Field(
            default="command", description="Handler operation mode"
        )
        handler_config: FlextModels.CqrsConfig.Handler | None = Field(
            default=None, description="Optional handler configuration"
        )
        registration_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id,
            min_length=1,
            description="Unique registration identifier",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class DispatchRequest(BaseModel):
        """Pydantic model for dispatch requests.

        USAGE: Replaces individual dispatch parameters with structured request.
        OPTIMIZATION: Provides type-safe dispatch with metadata handling.
        """

        message: object = Field(..., description="Message to dispatch")
        context_metadata: FlextModels.Metadata | None = Field(
            default=None, description="Optional execution context metadata"
        )
        correlation_id: str | None = Field(
            default=None,
            min_length=1,
            description="Optional correlation ID for tracing",
        )
        timeout_override: int | None = Field(
            default=None, ge=1, le=600, description="Optional timeout override"
        )
        request_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id,
            min_length=1,
            description="Unique request identifier",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class DispatchResult(BaseModel):
        """Pydantic model for dispatch results.

        USAGE: Provides structured dispatch result information.
        OPTIMIZATION: Type-safe result handling with execution metrics.
        """

        success: bool = Field(..., description="Whether dispatch was successful")
        result: object | None = Field(default=None, description="Dispatch result data")
        error_message: str | None = Field(
            default=None, description="Error message if dispatch failed"
        )
        request_id: str = Field(
            ..., min_length=1, description="Request ID from original dispatch request"
        )
        execution_time_ms: int = Field(
            default=0, ge=0, description="Execution time in milliseconds"
        )
        correlation_id: str | None = Field(
            default=None, description="Correlation ID for tracing"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class RegistrationDetails(BaseModel):
        """Pydantic model for registration result details.

        USAGE: Provides structured registration result information.
        OPTIMIZATION: Type-safe registration tracking.
        """

        registration_id: str = Field(
            ..., min_length=1, description="Unique registration identifier"
        )
        message_type_name: str | None = Field(
            default=None, description="Name of registered message type"
        )
        handler_mode: Literal["command", "query"] = Field(
            ..., description="Handler operation mode"
        )
        timestamp: str = Field(..., description="Registration timestamp")
        status: Literal["active", "inactive", "error"] = Field(
            default="active", description="Registration status"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

    HTTP_STATUS_MIN = 100
    HTTP_STATUS_MAX = 599

    # FlextMixins Models - Pydantic models for FlextMixins operations
    class SerializationRequest(BaseModel):
        """Pydantic model for serialization requests.

        USAGE: Provides structured serialization input with validation.
        OPTIMIZATION: Type-safe serialization with configurable options.
        """

        obj: object = Field(..., description="Object to serialize")
        use_model_dump: bool = Field(
            default=True, description="Whether to use model_dump method if available"
        )
        indent: int | None = Field(
            default=None, ge=0, le=8, description="JSON indentation level"
        )
        sort_keys: bool = Field(default=False, description="Whether to sort JSON keys")
        ensure_ascii: bool = Field(
            default=False, description="Whether to ensure ASCII-only output"
        )
        encoding: str = Field(
            default="utf-8",
            pattern=r"^(utf-8|ascii|latin-1|utf-16|utf-32)$",
            description="Character encoding for serialization",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class JsonFormatConfig(BaseModel):
        """Pydantic model for JSON formatting configuration.

        USAGE: Provides structured JSON formatting settings.
        OPTIMIZATION: Type-safe JSON configuration with validation.
        """

        indent: int | None = Field(
            default=2, ge=0, le=8, description="JSON indentation level"
        )
        sort_keys: bool = Field(default=False, description="Whether to sort JSON keys")
        ensure_ascii: bool = Field(
            default=False, description="Whether to ensure ASCII-only output"
        )
        separators: tuple[str, str] | None = Field(
            default=None, description="JSON separators (item_separator, key_separator)"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
        )

    class LogOperation(BaseModel):
        """Pydantic model for logging operations.

        USAGE: Provides structured logging operation input.
        OPTIMIZATION: Type-safe logging with context management.
        """

        obj: object = Field(..., description="Object to log operation for")
        operation: str = Field(
            ..., min_length=1, max_length=100, description="Operation name to log"
        )
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO", description="Log level for the operation"
        )
        context: dict[str, object] = Field(
            default_factory=dict, description="Additional context for logging"
        )
        timestamp: datetime | None = Field(
            default=None, description="Optional timestamp for the operation"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class TimestampConfig(BaseModel):
        """Pydantic model for timestamp configuration.

        USAGE: Provides structured timestamp management settings.
        OPTIMIZATION: Type-safe timestamp operations with timezone handling.
        """

        obj: object = Field(..., description="Object to configure timestamps for")
        use_utc: bool = Field(default=True, description="Whether to use UTC timestamps")
        auto_update: bool = Field(
            default=True, description="Whether to automatically update timestamps"
        )
        field_names: dict[str, str] = Field(
            default_factory=lambda: {
                "created_at": "created_at",
                "updated_at": "updated_at",
            },
            description="Mapping of timestamp field names",
        )
        format_string: str | None = Field(
            default=None, description="Optional custom timestamp format string"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        @field_validator("field_names")
        @classmethod
        def validate_field_names(cls, v: dict[str, str]) -> dict[str, str]:
            """Validate field names mapping."""
            required_keys = {"created_at", "updated_at"}
            if not required_keys.issubset(v.keys()):
                missing = required_keys - set(v.keys())
                msg = f"field_names missing required keys: {missing}"
                raise ValueError(msg)

            return v

    class StateInitializationRequest(BaseModel):
        """Pydantic model for state initialization requests.

        USAGE: Provides structured state initialization input.
        OPTIMIZATION: Type-safe state management with validation.
        """

        obj: object = Field(..., description="Object to initialize state for")
        state: str = Field(
            ..., min_length=1, max_length=50, description="Initial state value"
        )
        field_name: str = Field(
            default="state",
            min_length=1,
            max_length=50,
            description="Name of the state field",
        )
        validate_state: bool = Field(
            default=True, description="Whether to validate state value"
        )
        allowed_states: list[str] | None = Field(
            default=None, description="Optional list of allowed state values"
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        @field_validator("state")
        @classmethod
        def validate_state_value(cls, v: str, info: ValidationInfo) -> str:
            """Validate state value against allowed states if provided."""
            data = info.data if hasattr(info, "data") else {}
            allowed_states = data.get("allowed_states")

            if allowed_states and v not in allowed_states:
                msg = f"state '{v}' not in allowed states: {allowed_states}"
                raise ValueError(msg)

            return v

    class ValidationRequest(BaseModel):
        """Pydantic model for validation requests.

        USAGE: Provides structured validation input.
        OPTIMIZATION: Type-safe validation with configurable options.
        """

        obj: object = Field(..., description="Object to validate")
        validation_type: Literal["basic", "strict", "custom"] = Field(
            default="basic", description="Type of validation to perform"
        )
        field_name: str = Field(
            default="validated",
            min_length=1,
            max_length=50,
            description="Name of the validation flag field",
        )
        custom_validators: list[str] | None = Field(
            default=None, description="Optional list of custom validation methods"
        )
        raise_on_failure: bool = Field(
            default=False,
            description="Whether to raise exceptions on validation failure",
        )

        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
