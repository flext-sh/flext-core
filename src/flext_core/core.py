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
from flext_core.delegation import FlextDelegationSystem
from flext_core.exceptions import FlextExceptions
from flext_core.fields import FlextFields
from flext_core.guards import FlextGuards
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.services import FlextServices
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations


class FlextCore:
    """Minimal orchestration facade for FLEXT ecosystem - DIRECT ACCESS ONLY.

    Pure facade pattern - NO duplicated methods, just direct access to real classes.
    Use: core.Utilities.method(), core.Validations.method(), etc.
    """

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize with DIRECT ACCESS to real flext-core components."""
        self.entity_id = str(uuid.uuid4())
        self._container = FlextContainer.get_global()
        self._session_id = self._generate_session_id()

        # For test compatibility only
        self._specialized_configs: dict[str, object] = {}

        # DIRECT ACCESS ONLY - use these directly, no wrapper methods
        self.Config = FlextConfig
        self.Models = FlextModels
        self.Commands = FlextCommands
        self.Handlers = FlextHandlers
        self.Validations = FlextValidations
        self.Utilities = FlextUtilities
        self.Adapters = FlextTypeAdapters
        self.Services = FlextServices
        self.Decorators = FlextDecorators
        self.Processors = FlextProcessors
        self.Guards = FlextGuards
        self.Fields = FlextFields
        self.Mixins = FlextMixins
        self.Protocols = FlextProtocols
        self.Exceptions = FlextExceptions
        self.Delegation = FlextDelegationSystem
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
