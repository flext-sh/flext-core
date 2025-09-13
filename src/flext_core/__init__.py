"""FLEXT Core - Data integration foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypeVar

from flext_core.adapters import FlextTypeAdapters
from flext_core.commands import FlextCommands
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.core import FlextCore
from flext_core.decorators import FlextDecorators
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions

# Import compatibility layers from proper modules
from flext_core.fields import FlextFields
from flext_core.guards import FlextGuards
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processing import FlextProcessing
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# NEW: Refactored modules for cleaner architecture (v0.9.x+)
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations, Predicates
from flext_core.version import FlextVersionManager, __version__

# Common type variables for generic programming
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
E = TypeVar("E")
F = TypeVar("F")
P = TypeVar("P")
R = TypeVar("R")

# Ultra-simple aliases for backward compatibility
FlextGenerators = FlextUtilities.Generators

# Compatibility aliases for deleted modules
FlextHandlers = FlextProcessing  # Handlers consolidated into Processing
FlextProcessors = FlextProcessing  # Processors consolidated into Processing
FlextServices = FlextProcessing  # Services consolidated into Processing
FlextDelegationSystem = FlextProcessing  # Delegation was fake, using Processing
# FlextDecorators imported directly from decorators module
# FlextMixins is NOT an alias - it's a real module!

__all__ = [
    "E",
    "F",
    "FlextCommands",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextDecorators",
    "FlextDomainService",
    "FlextExceptions",
    "FlextFields",
    "FlextGenerators",
    "FlextGuards",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessing",
    "FlextProtocols",
    "FlextResult",
    "FlextTypeAdapters",
    "FlextTypes",
    "FlextUtilities",
    "FlextValidations",
    "FlextVersionManager",
    "P",
    "Predicates",
    "R",
    "T",
    "U",
    "V",
    "__version__",
]
