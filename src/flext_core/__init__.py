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
from flext_core.delegation import FlextDelegationSystem
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions, create_module_exception_classes
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

__all__ = [
    # Type variables
    "E",
    "F",
    # Core classes
    "FlextCommands",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextDecorators",
    "FlextDelegationSystem",
    "FlextDomainService",
    "FlextExceptions",
    "FlextFields",
    "FlextGenerators",
    "FlextGuards",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessors",
    "FlextProtocols",
    "FlextResult",
    "FlextServices",
    "FlextTypeAdapters",
    "FlextTypes",
    "FlextUtilities",
    "FlextValidations",
    "FlextVersionManager",
    "P",
    "R",
    "T",
    "U",
    "V",
    # Version info
    "__version__",
    # Infrastructure functions
    "create_module_exception_classes",
]
