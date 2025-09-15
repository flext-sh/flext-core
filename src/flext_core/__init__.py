"""FLEXT Core - Data integration foundation library.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
from flext_core.fields import FlextFields
from flext_core.guards import FlextGuards
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processing import FlextProcessing
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import E, F, FlextTypes, P, R, T, T_co, U, V
from flext_core.utilities import FlextUtilities
from flext_core.validations import FlextValidations, Predicates
from flext_core.version import FlextVersionManager, __version__

# Backward compatibility alias
FlextHandlers = FlextProcessing

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
    "FlextGuards",
    "FlextHandlers",
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
    "T_co",
    "U",
    "V",
    "__version__",
]

# No test-time monkey patches: keep core semantics consistent.
