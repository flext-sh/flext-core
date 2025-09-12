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
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processing import FlextProcessing
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# NEW: Refactored modules for cleaner architecture (v0.9.x+)
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

# Compatibility aliases for deleted modules
FlextHandlers = FlextProcessing  # Handlers consolidated into Processing
FlextProcessors = FlextProcessing  # Processors consolidated into Processing
FlextServices = FlextProcessing  # Services consolidated into Processing
FlextDelegationSystem = FlextProcessing  # Delegation was fake, using Processing
# FlextDecorators imported directly from decorators module
# FlextMixins is NOT an alias - it's a real module!

# Compatibility aliases during consolidation (TEMPORARY)
# These will be removed once all code is updated to use FlextValidations
FlextGuards = type(
    "FlextGuards",
    (),
    {
        "is_dict_of": FlextValidations.Guards.is_dict_of,
        "is_list_of": FlextValidations.Guards.is_list_of,
        "require_not_none": FlextValidations.Guards.require_not_none,
        "require_positive": FlextValidations.Guards.require_positive,
        "require_in_range": FlextValidations.Guards.require_in_range,
        "require_non_empty": FlextValidations.Guards.require_non_empty,
        "ValidationUtils": type(
            "ValidationUtils",
            (),
            {
                "require_not_none": FlextValidations.Guards.require_not_none,
                "require_positive": FlextValidations.Guards.require_positive,
                "require_in_range": FlextValidations.Guards.require_in_range,
                "require_non_empty": FlextValidations.Guards.require_non_empty,
            },
        ),
    },
)

FlextFields = type(
    "FlextFields",
    (),
    {
        "validate_email": FlextValidations.FieldValidators.validate_email,
        "validate_uuid": FlextValidations.FieldValidators.validate_uuid,
        "validate_url": FlextValidations.FieldValidators.validate_url,
        "validate_phone": FlextValidations.FieldValidators.validate_phone,
    },
)

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
    "R",
    "T",
    "U",
    "V",
    "__version__",
]
