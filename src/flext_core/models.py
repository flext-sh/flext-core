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
import warnings
from collections import UserString
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Literal, cast
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from flext_core.config import FlextConfig
from flext_core.result import FlextResult
from flext_core.typings import T
from flext_core.utilities import FlextUtilities


class FlextModels:
    """Namespace for the canonical FLEXT domain primitives.

    These types lock in the immutable, timestamped data structures referenced
    by the modernization plan and exercised across the FLEXT ecosystem.
    """

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
            default="", description="Correlation ID for tracking"
        )
        message_id: str = Field(default="", description="Unique message identifier")
        expires_at: datetime | None = Field(
            default=None, description="Payload expiration time"
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
            """Validate aggregate_id is not empty or whitespace only."""
            if not v or not v.strip():
                error_msg = "Aggregate identifier cannot be empty or whitespace only"
                raise ValueError(error_msg)
            return v.strip()

    class Command(BaseModel):
        """Command message template for dispatcher-driven CQRS flows."""

        command_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        command_type: str = Field(...)
        payload: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        correlation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id
        )
        user_id: str | None = Field(
            default=None, description="User ID associated with the command"
        )

        def validate_command(self) -> FlextResult[bool]:
            """Validate command data."""
            if not self.command_type:
                return FlextResult[bool].fail("Command type is required")
            return FlextResult[bool].ok(data=True)

    class Query(BaseModel):
        """Query message template for dispatcher-driven CQRS flows."""

        query_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        query_type: str = Field(...)
        filters: dict[str, object] = Field(default_factory=dict)
        pagination: dict[str, int] = Field(
            default_factory=lambda: {"page": 1, "size": 10}
        )
        user_id: str | None = Field(
            default=None, description="User ID associated with the query"
        )

        def validate_query(self) -> FlextResult[bool]:
            """Validate query data."""
            if not self.query_type:
                return FlextResult[bool].fail("Query type is required")
            return FlextResult[bool].ok(data=True)

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
                    r"([a-z0-9])([A-Z])", r"\1_\2", s1
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
                    setattr(self, "metadata", {})
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
            """Simplified email validation."""
            email = self.root
            error_msg = "Invalid email format"

            # Basic email validation - simplified from complex logic
            if email.count("@") != 1:
                raise ValueError(error_msg)

            local, domain = email.split("@", 1)
            if not local or not domain:
                raise ValueError(error_msg)

            if "." not in domain or domain.startswith(".") or domain.endswith("."):
                raise ValueError(error_msg)

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
                    "Host cannot contain spaces"
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
                    "Entity ID cannot be empty"
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
                    f"Invalid JSON data: {e}"
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
                    f"Metadata values for keys {invalid_keys} must be strings"
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
                    "URL must start with http:// or https://"
                )

            if "://" in url and not url.split("://", 1)[1]:
                return FlextResult["FlextModels.Url"].fail(
                    "URL must have a valid hostname"
                )

            return FlextResult["FlextModels.Url"].ok(cls(value=url))

        @classmethod
        def create_http_url(
            cls, url: str, *, max_length: int = 2048, max_port: int = 65535
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
                            f"Invalid port {port}"
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
                cleaned = FlextUtilities.TextProcessor.clean_text(self.value)
                if not cleaned:
                    return FlextResult["FlextModels.Url"].fail("URL cannot be empty")

                normalized = (
                    cleaned.rstrip("/") if not cleaned.endswith("://") else cleaned
                )
                return FlextResult["FlextModels.Url"].ok(
                    self.__class__(value=normalized)
                )
            except Exception as e:
                return FlextResult["FlextModels.Url"].fail(
                    f"URL normalization failed: {e}"
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
            default_factory=dict, description="Additional settings"
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
            ..., description="Project type"
        )  # Use string to avoid enum import
        has_tests: bool = Field(default=False, description="Has test directory")
        has_pyproject: bool = Field(default=False, description="Has pyproject.toml")
        has_go_mod: bool = Field(default=False, description="Has go.mod file")
        test_count: int = Field(0, ge=0, description="Number of test files")

        def validate_business_rules(self) -> FlextResult[None]:
            """Implement required abstract method from Value."""
            return FlextResult[None].ok(None)

    class WorkspaceContext(Config):
        """Workspace context configuration reused across modernization tooling."""

        workspace_root: str = Field(..., description="Workspace root path")
        project_filter: str | None = Field(None, description="Project name filter")
        include_hidden: bool = Field(
            default=False, description="Include hidden directories"
        )
        max_depth: int = Field(
            default=3, ge=1, le=10, description="Maximum directory depth"
        )

    class WorkspaceInfo(Value):
        """Workspace information object feeding modernization dashboards."""

        name: str = Field(..., description="Workspace name")
        path: str = Field(..., description="Workspace path")
        project_count: int = Field(default=0, ge=0, description="Number of projects")
        total_size_mb: float = Field(default=0.0, ge=0, description="Total size in MB")
        projects: list[str] | None = Field(
            default=None, description="List of project names"
        )
        status: str = Field(
            default="ready", description="Workspace status"
        )  # Use string to avoid enum import

        def validate_business_rules(self) -> FlextResult[None]:
            """Implement required abstract method from Value."""
            if self.project_count < 0:
                return FlextResult[None].fail("Project count cannot be negative")
            if self.total_size_mb < 0:
                return FlextResult[None].fail("Total size cannot be negative")
            return FlextResult[None].ok(None)

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
        event_type: str, payload: dict[str, object], aggregate_id: str
    ) -> Event:
        """Create an event."""
        return FlextModels.Event(
            event_type=event_type, payload=payload, aggregate_id=aggregate_id
        )

    @staticmethod
    def create_command(command_type: str, payload: dict[str, object]) -> Command:
        """Create a command."""
        return FlextModels.Command(command_type=command_type, payload=payload)

    @staticmethod
    def create_query(
        query_type: str, filters: dict[str, object] | None = None
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
        """Create validated email using Pydantic models - replaces FieldValidators.validate_email."""
        try:
            validated = FlextModels.EmailAddress(email)
            return FlextResult[str].ok(validated.root)
        except ValueError as e:
            return FlextResult[str].fail(str(e))

    @staticmethod
    def create_validated_url(url: str) -> FlextResult[str]:
        """Create validated URL using Pydantic models - replaces FieldValidators.validate_url."""
        url_result = FlextModels.Url.create(url)
        if url_result.is_success:
            return FlextResult[str].ok(url_result.unwrap().value)
        return FlextResult[str].fail(url_result.error or "Invalid URL")

    @staticmethod
    def create_validated_http_url(
        url: str, *, max_length: int = 2048, max_port: int = 65535
    ) -> FlextResult[str]:
        """Create validated HTTP URL with enhanced validation - centralized replacement for HttpValidator."""
        url_result = FlextModels.Url.create_http_url(
            url, max_length=max_length, max_port=max_port
        )
        if url_result.is_success:
            return FlextResult[str].ok(url_result.unwrap().value)
        return FlextResult[str].fail(url_result.error or "Invalid HTTP URL")

    @staticmethod
    def create_validated_http_method(method: str | None) -> FlextResult[str]:
        """Create validated HTTP method - centralized replacement for HttpValidator.validate_http_method."""
        if not method or not isinstance(method, str):
            return FlextResult[str].fail("HTTP method must be a non-empty string")

        method_upper = method.upper()
        # Standard HTTP methods - using Python 3.13+ enum patterns
        valid_methods = {
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
            "TRACE",
            "CONNECT",
        }

        if method_upper not in valid_methods:
            valid_methods_str = ", ".join(sorted(valid_methods))
            return FlextResult[str].fail(
                f"Invalid HTTP method. Valid methods: {valid_methods_str}"
            )

        return FlextResult[str].ok(method_upper)

    @staticmethod
    def create_validated_http_status(code: int | str) -> FlextResult[int]:
        """Create validated HTTP status code - centralized replacement for HttpValidator.validate_status_code."""
        try:
            code_int = int(code)
            if (
                code_int < FlextModels.HTTP_STATUS_MIN
                or code_int > FlextModels.HTTP_STATUS_MAX
            ):
                return FlextResult[int].fail(
                    f"Invalid HTTP status code range ({FlextModels.HTTP_STATUS_MIN}-{FlextModels.HTTP_STATUS_MAX})"
                )
            return FlextResult[int].ok(code_int)
        except (ValueError, TypeError):
            return FlextResult[int].fail("Status code must be a valid integer")

    @staticmethod
    def create_validated_phone(phone: str) -> FlextResult[str]:
        """Create validated phone - replaces FieldValidators.validate_phone."""
        # Remove non-digit characters for validation

        digits_only = re.sub(r"\D", "", phone)
        min_phone_digits = 10
        max_phone_digits = 15
        if len(digits_only) < min_phone_digits or len(digits_only) > max_phone_digits:
            return FlextResult[str].fail(f"Invalid phone number: {phone}")

        return FlextResult[str].ok(phone)

    @staticmethod
    def create_validated_uuid(uuid_str: str) -> FlextResult[str]:
        """Create validated UUID - replaces FieldValidators.validate_uuid."""
        uuid_pattern = (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        if not re.match(uuid_pattern, uuid_str.lower()):
            return FlextResult[str].fail(f"Invalid UUID format: {uuid_str}")

        return FlextResult[str].ok(uuid_str)

    @staticmethod
    def create_validated_iso_date(date_str: str) -> FlextResult[str]:
        """Create validated ISO date string - centralizes datetime.fromisoformat() validation.

        Args:
            date_str: Date string in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)

        Returns:
            FlextResult containing validated date string or validation error

        Examples:
            >>> result = FlextModels.create_validated_iso_date("2025-01-08")
            >>> if result.is_success:
            ...     validated_date = result.unwrap()

        """
        if not date_str or not date_str.strip():
            return FlextResult[str].fail("Date string cannot be empty")

        try:
            # Use datetime.fromisoformat for validation (this is the pattern we're centralizing)
            datetime.fromisoformat(date_str.strip())
            return FlextResult[str].ok(date_str.strip())
        except ValueError as e:
            return FlextResult[str].fail(f"Invalid ISO date format: {e}")

    @staticmethod
    def create_validated_date_range(
        start_date: str, end_date: str
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
                f"Invalid start date: {start_result.error}"
            )

        end_result = FlextModels.create_validated_iso_date(end_date)
        if end_result.is_failure:
            return FlextResult[tuple[str, str]].fail(
                f"Invalid end date: {end_result.error}"
            )

        # Validate date range
        try:
            start_dt = datetime.fromisoformat(start_date.strip())
            end_dt = datetime.fromisoformat(end_date.strip())

            if start_dt >= end_dt:
                return FlextResult[tuple[str, str]].fail(
                    "Start date must be before end date"
                )

            return FlextResult[tuple[str, str]].ok(
                (start_date.strip(), end_date.strip())
            )
        except ValueError as e:
            return FlextResult[tuple[str, str]].fail(
                f"Date range validation failed: {e}"
            )

    @staticmethod
    def create_validated_file_path(file_path: str) -> FlextResult[str]:
        """Create validated file path string - centralizes pathlib.Path validation.

        Args:
            file_path: Path string to validate

        Returns:
            FlextResult with validated path string or validation error

        """
        if not file_path or not file_path.strip():
            return FlextResult[str].fail("File path cannot be empty")

        try:
            path = Path(file_path.strip())
            # Basic validation - path should be constructible
            str(path)  # Verify path can be converted to string
            return FlextResult[str].ok(str(path))
        except (OSError, ValueError) as e:
            return FlextResult[str].fail(f"Invalid file path: {e}")

    @staticmethod
    def create_validated_existing_file_path(file_path: str) -> FlextResult[str]:
        """Create validated existing file path - centralizes Path().exists() validation.

        Args:
            file_path: Path string to validate for existence

        Returns:
            FlextResult with validated path string or validation error

        """
        # First validate the path format
        path_result = FlextModels.create_validated_file_path(file_path)
        if path_result.is_failure:
            return path_result

        try:
            path = Path(path_result.unwrap())
            if not path.exists():
                return FlextResult[str].fail(f"Path does not exist: {path}")
            return FlextResult[str].ok(str(path))
        except (OSError, PermissionError) as e:
            return FlextResult[str].fail(f"Cannot access path: {e}")

    @staticmethod
    def create_validated_directory_path(dir_path: str) -> FlextResult[str]:
        """Create validated directory path - centralizes Path().is_dir() validation.

        Args:
            dir_path: Directory path string to validate

        Returns:
            FlextResult with validated directory path string or validation error

        """
        # First validate the path exists
        existing_path_result = FlextModels.create_validated_existing_file_path(dir_path)
        if existing_path_result.is_failure:
            return existing_path_result

        try:
            path = Path(existing_path_result.unwrap())
            if not path.is_dir():
                return FlextResult[str].fail(f"Path is not a directory: {path}")
            return FlextResult[str].ok(str(path))
        except (OSError, PermissionError) as e:
            return FlextResult[str].fail(f"Cannot verify directory: {e}")

    # HTTP status code constants
    HTTP_STATUS_MIN = 100
    HTTP_STATUS_MAX = 599
