"""Core module aggregation for FLEXT foundation library.

Simple singleton that aggregates other flext-core modules.
Not actual "orchestration" - just property assignments.

Copyright (c) 2025 FLEXT Team. All rights reserved
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from flext_core.adapters import FlextTypeAdapters
from flext_core.commands import FlextCommands
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processing import FlextProcessing
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations


class FlextCore:
    """Core session and configuration management for FLEXT ecosystem."""

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize with DIRECT ACCESS to real flext-core components."""
        self.entity_id = str(uuid.uuid4())
        self._container = FlextContainer.get_global()
        self._session_id = self._generate_session_id()

        # Get global configuration singleton
        self._config = FlextConfig.get_global_instance()

        # For test compatibility only
        self._specialized_configs: dict[str, object] = {}

        self.Adapters = FlextTypeAdapters
        self.Commands = FlextCommands
        self.Config = FlextConfig
        self.Constants = FlextConstants
        self.Container = FlextContainer
        self.Context = FlextContext
        self.Decorators = FlextDecorators
        self.DomainService = FlextDomainService
        self.Exceptions = FlextExceptions
        self.Fields = FlextValidations.FieldValidators
        self.Logger = FlextLogger
        self.Mixins = FlextMixins
        self.Models = FlextModels
        self.Processors = FlextProcessing
        self.Protocols = FlextProtocols
        self.Result = FlextResult
        self.Utilities = FlextUtilities
        self.Validations = FlextValidations

    # =============================================================================
    # CORE FUNCTIONALITY - MINIMAL REQUIRED
    # =============================================================================

    @classmethod
    def get_instance(cls) -> FlextCore:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> FlextConfig:
        """Get the global configuration singleton."""
        return self._config

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for testing."""
        cls._instance = None

    def _generate_session_id(self) -> str:
        """Generate session ID."""
        return f"session_{uuid.uuid4().hex[:12]}_{int(datetime.now(UTC).timestamp())}"

    def get_session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    def cleanup(self) -> FlextResult[None]:
        """Cleanup resources and reset session."""
        try:
            self._session_id = self._generate_session_id()
            return FlextResult[None].ok(None)
        except Exception as error:
            return FlextResult[None].fail(f"Cleanup failed: {error}")

    # Container access - direct access to global container
    @property
    def container(self) -> FlextContainer:
        """Direct access to global container. Use: core.container.register(), etc."""
        return self._container

    # String representation
    def __str__(self) -> str:
        """String representation of FlextCore."""
        return f"FlextCore - FLEXT ecosystem foundation (id={self.entity_id})"

    def __repr__(self) -> str:
        """Detailed string representation of FlextCore."""
        return f"FlextCore(entity_id='{self.entity_id}')"

    # =============================================================================
    # Convenience APIs used in functional tests
    # =============================================================================

    # ---- Entity creation helper ----
    def create_entity(
        self, entity_cls: type[FlextModels.Entity], /, **kwargs: object
    ) -> FlextResult[FlextModels.Entity]:
        """Create an entity instance and run business validation if available.

        Keeps behavior minimal and explicit for tests.

        Note: MyPy cannot verify kwargs compatibility at compile time for dynamic entity creation.
        This is expected behavior for runtime entity instantiation patterns.
        """
        try:
            # Dynamic entity instantiation - MyPy cannot verify kwargs at compile time
            # This is intentional for flexible entity creation in tests
            # Type ignore needed for dynamic instantiation with object kwargs
            entity = entity_cls(**kwargs)  # type: ignore[arg-type]
            # Run business rule validation if method exists
            validator = getattr(entity, "validate_business_rules", None)
            if callable(validator):
                res = validator()
                if getattr(res, "is_failure", False):
                    return FlextResult[FlextModels.Entity].fail(
                        getattr(res, "error", None) or "Business rule validation failed"
                    )
            return FlextResult[FlextModels.Entity].ok(entity)
        except Exception as e:
            return FlextResult[FlextModels.Entity].fail(
                f"Entity creation failed: {e}"
            )

    # ---- Payload helper tailored for tests ----
    class _CorePayload(BaseModel):
        data: dict[str, object]
        message_type: str
        source_service: str
        target_service: str | None = None
        message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
        correlation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_uuid
        )
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        priority: int = 5
        retry_count: int = 0
        expires_at: datetime | None = None

        @property
        def is_expired(self) -> bool:
            return (
                False
                if self.expires_at is None
                else datetime.now(UTC) > self.expires_at
            )

        @property
        def age_seconds(self) -> float:
            return max(0.0, (datetime.now(UTC) - self.timestamp).total_seconds())

    def create_payload(
        self,
        *,
        data: dict[str, object],
        message_type: str,
        source_service: str,
        target_service: str | None = None,
        priority: int = 5,
    ) -> FlextResult[_CorePayload]:
        """Create a lightweight message payload with metadata for tests."""
        try:
            payload = self._CorePayload(
                data=data,
                message_type=message_type,
                source_service=source_service,
                target_service=target_service,
                priority=priority,
            )
            return FlextResult[FlextCore._CorePayload].ok(payload)
        except Exception as e:
            return FlextResult[FlextCore._CorePayload].fail(str(e))

    # ---- Container helpers (thin wrappers over FlextContainer) ----
    def register_service(self, name: str, instance: object) -> FlextResult[None]:
        """Register a service instance in the container."""
        return self._container.register(name, instance)

    def register_factory(
        self, name: str, factory: Callable[[], object]
    ) -> FlextResult[None]:
        """Register a factory function in the container."""
        return self._container.register_factory(name, factory)

    def get_service(self, name: str) -> FlextResult[object]:
        """Get a service instance from the container."""
        return self._container.get(name)

    # ---- Simple database configuration helper used by tests ----
    class _DatabaseConfig(BaseModel):
        host: str
        database: str
        username: str
        password: str
        port: int
        pool_size: int

    def configure_database(
        self,
        *,
        host: str,
        database: str,
        username: str,
        password: str,
        port: int,
        pool_size: int,
    ) -> FlextResult[_DatabaseConfig]:
        """Store basic database configuration in container and return it."""
        try:
            cfg = self._DatabaseConfig(
                host=host,
                database=database,
                username=username,
                password=password,
                port=port,
                pool_size=pool_size,
            )
            # Keep a copy on container for visibility in diagnostics
            self._container.configure_database(cfg.model_dump())
            return FlextResult[FlextCore._DatabaseConfig].ok(cfg)
        except Exception as e:
            return FlextResult[FlextCore._DatabaseConfig].fail(str(e))


__all__ = ["FlextCore"]
