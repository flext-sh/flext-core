"""Public API for flext-core.

This package exposes the dispatcher-centric application layer (dispatcher,
registry, handlers), domain primitives (models, services), shared
utilities, and infrastructure bridges (configuration, logging, context)
used across the FLEXT ecosystem. Components follow clean architecture
boundaries and favor structural typing for protocol compliance.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from beartype import BeartypeConf, BeartypeStrategy

from flext_core.__version__ import __version__, __version_info__
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import (
    CallableInputT,
    CallableOutputT,
    E,
    F,
    FactoryT,
    FlextTypes,
    K,
    MessageT_contra,
    P,
    R,
    ResultT,
    T,
    T1_co,
    T2_co,
    T3_co,
    T_co,
    T_Config,
    T_contra,
    T_Model,
    T_Namespace,
    TAggregate_co,
    TCacheKey_contra,
    TCacheValue_co,
    TCommand_contra,
    TConfigKey_contra,
    TDomainEvent_co,
    TEntity_co,
    TEvent_contra,
    TInput_contra,
    TInput_Handler_contra,
    TInput_Handler_Protocol_contra,
    TItem_contra,
    TQuery_contra,
    TResult_co,
    TResult_contra,
    TResult_Handler_co,
    TResult_Handler_Protocol,
    TState_co,
    TUtil_contra,
    TValue_co,
    TValueObject_co,
    U,
    V,
    W,
)
from flext_core.utilities import FlextUtilities

# Type aliases for nested FlextConfig attributes (pyrefly compatibility)
AutoConfig = FlextConfig.AutoConfig
auto_register = FlextConfig.auto_register

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
    "AutoConfig",
    "CallableInputT",
    "CallableOutputT",
    "E",
    "F",
    "FactoryT",
    "FlextConfig",
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
    "FlextTypes",
    "FlextUtilities",
    "K",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAggregate_co",
    "TCacheKey_contra",
    "TCacheValue_co",
    "TCommand_contra",
    "TConfigKey_contra",
    "TDomainEvent_co",
    "TEntity_co",
    "TEvent_contra",
    "TInput_Handler_Protocol_contra",
    "TInput_Handler_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_Handler_Protocol",
    "TResult_Handler_co",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_Config",
    "T_Model",
    "T_Namespace",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
    "__version__",
    "__version_info__",
    "auto_register",
]
# NOTE: UserDataMapping and ValidationResult are in FlextTypes.Example
# Access via: FlextTypes.Example.UserDataMapping, FlextTypes.Example.ValidationResult
