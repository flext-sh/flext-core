"""FLEXT Core - Data integration foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# =============================================================================
# TYPE VARIABLES - Common typing imports for convenience
# =============================================================================

from typing import TypeVar

# Common type variables for generic programming
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")
F = TypeVar("F")
P = TypeVar("P")
R = TypeVar("R")

# =============================================================================
# FOUNDATION LAYER - Import classes directly to avoid conflicts
# =============================================================================

from flext_core.version import FlextVersionManager, __version__
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes
from flext_core.result import FlextResult
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols

# =============================================================================
# DOMAIN LAYER - Domain-driven design patterns
# =============================================================================

from flext_core.domain_services import FlextDomainService
from flext_core.models import FlextModels

# =============================================================================
# APPLICATION LAYER - CQRS, Commands, Handlers, Guards
# =============================================================================

from flext_core.commands import FlextCommands
from flext_core.guards import FlextGuards
from flext_core.handlers import FlextHandlers
from flext_core.validations import FlextValidations

# =============================================================================
# INFRASTRUCTURE LAYER - Configuration, DI, Context, Logging
# =============================================================================

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
# FlextObservability removed as per request

# =============================================================================
# SUPPORT LAYER - Utilities, Mixins, Decorators, Processors
# =============================================================================

from flext_core.decorators import FlextDecorators
from flext_core.delegation import FlextDelegationSystem
from flext_core.fields import FlextFields
from flext_core.mixins import FlextMixins
from flext_core.processors import FlextProcessors
from flext_core.services import FlextServices
from flext_core.adapters import FlextTypeAdapters
from flext_core.utilities import FlextUtilities

# =============================================================================
# CORE FUNCTIONALITY - Main FlextCore class
# =============================================================================

from flext_core.core import FlextCore

# =============================================================================
# NO WRAPPER FUNCTIONS - Use direct class access only
# =============================================================================


# =============================================================================
# EXPORTS - Explicit __all__ definition
# =============================================================================

__all__ = [
    # Core classes
    "FlextCore",
    "FlextResult",
    "FlextContainer",
    "FlextConfig",
    "FlextLogger",
    # "FlextObservability", # Removed
    # Domain classes
    "FlextDomainService",
    "FlextModels",
    # Application classes
    "FlextCommands",
    "FlextGuards",
    "FlextHandlers",
    "FlextValidations",
    # Infrastructure classes
    "FlextContext",
    "FlextExceptions",
    "FlextProtocols",
    # Support classes
    "FlextDecorators",
    "FlextDelegationSystem",
    "FlextFields",
    "FlextMixins",
    "FlextProcessors",
    "FlextServices",
    "FlextTypeAdapters",
    "FlextUtilities",
    # Foundation classes
    "FlextVersionManager",
    "FlextConstants",
    "FlextTypes",
    # Version info
    "__version__",
    # Type variables
    "T",
    "U",
    "V",
    "E",
    "F",
    "P",
    "R",
]
