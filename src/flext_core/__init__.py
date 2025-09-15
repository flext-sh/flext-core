"""FLEXT Core - Data integration foundation library.

Foundation library providing railway-oriented programming, dependency
injection, and domain-driven design patterns for the FLEXT ecosystem.
"""

from __future__ import annotations

# =============================================================================
# ALL IMPORTS AT TOP - Required by ruff E402
# =============================================================================
from typing import TypeVar

from flext_core.adapters import FlextTypeAdapters
from flext_core.commands import FlextCommands
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.core import FlextCore
from flext_core.decorators import FlextDecorators
from flext_core.delegation import FlextDelegationSystem
from flext_core.domain_services import FlextDomainService
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
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations
from flext_core.version import FlextVersionManager, __version__

# =============================================================================
# TYPE VARIABLES - Common typing variables for convenience
# =============================================================================

# Common type variables for generic programming
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")
F = TypeVar("F")
P = TypeVar("P")
R = TypeVar("R")

# =============================================================================
# CONVENIENCE FUNCTIONS - Direct access to key functionality
# =============================================================================


# Direct access to key functions without class instantiation
def flext_logger(name: str = __name__) -> FlextLogger:
    """Get a structured logger instance."""
    return FlextLogger(name)


def get_flext_container() -> FlextContainer:
    """Get the global FlextContainer instance."""
    return FlextContainer.get_global()


def get_flext_core() -> FlextCore:
    """Get the global FlextCore instance."""
    return FlextCore.get_instance()


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
    # Convenience functions
    "FlextLogger",
    "get_flext_container",
    "get_flext_core",
    # Type variables
    "T",
    "U",
    "V",
    "E",
    "F",
    "P",
    "R",
]
