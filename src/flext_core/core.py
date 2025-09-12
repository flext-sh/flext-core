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
    """GOD OBJECT ALERT: Massive facade importing 21+ modules - ARCHITECTURAL SIN.

    # OVER-ENGINEERED AS FUCK: This is the definition of a god object:
    # - Imports 21 different modules
    # - Exposes EVERYTHING through one massive interface
    # - Has 4 overlapping request processors: Commands, Handlers, Services, Processors
    # - Utilities module is MASSIVE and should be multiple modules
    # - Delegation system is completely over-engineered
    # - Users could just import FlextResult, FlextValidations etc directly

    # ARCHITECTURAL VIOLATION: Single Responsibility Principle completely violated
    # This class knows about EVERYTHING in the entire system
    """

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

        # GOD OBJECT VIOLATION: Exposing 21+ modules through single interface!
        # DUPLICATE FUNCTIONALITY: 4 modules do the SAME SHIT - request processing:

        self.Config = FlextConfig
        self.Models = FlextModels

        # REQUEST PROCESSING HELL - CHOOSE ONE, NOT FOUR:
        self.Commands = FlextCommands      # ← CQRS patterns
        self.Processors = FlextProcessing   # ← Processing patterns
        # ALL FOUR DO THE SAME REQUEST→RESPONSE PROCESSING!

        self.Validations = FlextValidations
        self.Utilities = FlextUtilities    # MASSIVE 1000+ line monster - should be 5+ modules
        self.Adapters = FlextTypeAdapters  # Over-abstracted wrapper hell
        # self.Decorators = FlextDecorators  # Temporarily disabled
        self.Guards = FlextValidations.Guards  # At least consolidated now
        self.Fields = FlextValidations.FieldValidators  # At least consolidated now
        self.Mixins = FlextMixins          # Responsibility violations everywhere
        self.Protocols = FlextProtocols
        self.Exceptions = FlextExceptions
        # self.Delegation = FlextDelegationSystem  # Module doesn't exist
        self.Result = FlextResult
        self.Container = FlextContainer
        self.Context = FlextContext
        self.Logger = FlextLogger
        self.Constants = FlextConstants

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
