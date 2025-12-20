"""Public API for flext-core.

This package exposes the dispatcher-centric application layer (dispatcher,
registry, handlers), domain primitives (models, services), shared
utilities, and infrastructure bridges (configuration, logging, context)
used across the FLEXT ecosystem. Components follow clean architecture
boundaries and favor structural typing for protocol compliance.

FLEXT follows Railway-Oriented Programming with Result[T, E] patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from beartype import BeartypeConf, BeartypeStrategy

from flext_core.__version__ import __version__, __version_info__
from flext_core.constants import FlextConstants, c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators, d
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions, e
from flext_core.handlers import FlextHandlers, h
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels, m
from flext_core.protocols import FlextProtocols, p
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult, r
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService, s
from flext_core.settings import FlextSettings
from flext_core.typings import (
    E,
    FlextTypes,
    MessageT_contra,
    P,
    R,
    ResultT,
    T,
    T_co,
    T_contra,
    T_Model,
    T_Namespace,
    T_Settings,
    U,
    t,
)
from flext_core.utilities import FlextUtilities, u

# Runtime aliases
x = FlextMixins

# =============================================================================
# RUNTIME TYPE CHECKING - Python 3.13 Strict Typing Enforcement
# =============================================================================
# beartype provides RUNTIME type validation in addition to static checking.
#
# ENABLED via FlextRuntime.enable_runtime_checking() for package-wide validation.
# Critical methods can also use @beartype decorator individually.
#
# Beartype provides O(log n) runtime validation with minimal overhead.
# Static type checking (pyright strict mode) is ALWAYS active.
# Documentation: https://beartype.readthedocs.io/en/stable/
# =============================================================================

# Beartype configuration for runtime type checking (available for your use)
BEARTYPE_CONF = BeartypeConf(
    strategy=BeartypeStrategy.Ologn,  # O(log n) - thorough with acceptable overhead
    is_color=True,  # Colored error messages
    claw_is_pep526=False,  # Disable variable annotation checking
    warning_cls_on_decorator_exception=UserWarning,  # Warnings on decorator failures
)

# =============================================================================

__all__ = [
    "BEARTYPE_CONF",
    "E",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "__version__",
    "__version_info__",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
]
