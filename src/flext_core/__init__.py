"""FLEXT Core - Data integration foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


from typing import TypeVar

# Common type variables for generic programming
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")
F = TypeVar("F")
P = TypeVar("P")
R = TypeVar("R")


from flext_core.version import FlextVersionManager, __version__
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes
from flext_core.result import FlextResult
from flext_core.exceptions import FlextExceptions, create_module_exception_classes
from flext_core.protocols import FlextProtocols


from flext_core.domain_services import FlextDomainService
from flext_core.models import FlextModels, FlextFactory
from flext_core.fields import FlextFields


from flext_core.commands import FlextCommands
from flext_core.guards import FlextGuards
from flext_core.handlers import FlextHandlers
from flext_core.validations import FlextValidations


from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
# FlextObservability removed as per request


from flext_core.decorators import FlextDecorators
from flext_core.delegation import FlextDelegationSystem
from flext_core.mixins import FlextMixins
from flext_core.processors import FlextProcessors
from flext_core.services import FlextServices
from flext_core.adapters import FlextTypeAdapters
from flext_core.utilities import FlextUtilities

# Ultra-simple aliases for backward compatibility
FlextGenerators = FlextUtilities.Generators


from flext_core.core import FlextCore


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
    "FlextFactory",
    # Application classes
    "FlextCommands",
    "FlextGuards",
    "FlextHandlers",
    "FlextValidations",
    # Infrastructure classes
    "FlextContext",
    "FlextExceptions",
    "FlextProtocols",
    # Infrastructure functions
    "create_module_exception_classes",
    # Support classes
    "FlextDecorators",
    "FlextDelegationSystem",
    "FlextFields",
    "FlextMixins",
    "FlextProcessors",
    "FlextServices",
    "FlextTypeAdapters",
    "FlextUtilities",
    "FlextGenerators",
    # Foundation classes
    "FlextVersionManager",
    "FlextConstants",
    "FlextTypes",
    # Version info
    "__version__",
    "T",
    "U",
    "V",
    "E",
    "F",
    "P",
    "R",
]
