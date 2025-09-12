"""Core orchestration for FLEXT foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

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
        self.Commandsds = FlextCommands
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


__all__ = ["FlextCore"]
