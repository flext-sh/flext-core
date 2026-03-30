# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Examples package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from examples._models import _LAZY_IMPORTS as _CHILD_LAZY_0
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples._models import *
    from examples.ex_01_flext_result import *
    from examples.ex_02_flext_settings import *
    from examples.ex_03_flext_logger import *
    from examples.ex_04_flext_dispatcher import *
    from examples.ex_05_flext_mixins import *
    from examples.ex_06_flext_context import *
    from examples.ex_07_flext_exceptions import *
    from examples.ex_08_flext_container import *
    from examples.ex_09_flext_decorators import *
    from examples.ex_10_flext_handlers import *
    from examples.ex_11_flext_service import *
    from examples.ex_12_flext_registry import *
    from examples.logging_config_once_pattern import *
    from examples.models import *
    from examples.shared import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    **_CHILD_LAZY_0,
    "DatabaseService": "examples.logging_config_once_pattern",
    "Ex01r": "examples.ex_01_flext_result",
    "Ex02FlextSettings": "examples.ex_02_flext_settings",
    "Ex03FlextLogger": "examples.ex_03_flext_logger",
    "Ex04FlextDispatcher": "examples.ex_04_flext_dispatcher",
    "Ex05FlextMixins": "examples.ex_05_flext_mixins",
    "Ex06FlextContext": "examples.ex_06_flext_context",
    "Ex07FlextExceptions": "examples.ex_07_flext_exceptions",
    "Ex08FlextContainer": "examples.ex_08_flext_container",
    "Ex09FlextDecorators": "examples.ex_09_flext_decorators",
    "Ex10FlextHandlers": "examples.ex_10_flext_handlers",
    "Ex11FlextService": "examples.ex_11_flext_service",
    "Ex12FlextRegistry": "examples.ex_12_flext_registry",
    "Examples": "examples.shared",
    "FlextCoreExampleModels": "examples.models",
    "MigrationService": "examples.logging_config_once_pattern",
    "UserInput": "examples.models",
    "UserProfile": "examples.models",
    "_models": "examples._models",
    "em": "examples.models",
    "ex_01_flext_result": "examples.ex_01_flext_result",
    "ex_02_flext_settings": "examples.ex_02_flext_settings",
    "ex_03_flext_logger": "examples.ex_03_flext_logger",
    "ex_04_flext_dispatcher": "examples.ex_04_flext_dispatcher",
    "ex_05_flext_mixins": "examples.ex_05_flext_mixins",
    "ex_06_flext_context": "examples.ex_06_flext_context",
    "ex_07_flext_exceptions": "examples.ex_07_flext_exceptions",
    "ex_08_flext_container": "examples.ex_08_flext_container",
    "ex_09_flext_decorators": "examples.ex_09_flext_decorators",
    "ex_10_flext_handlers": "examples.ex_10_flext_handlers",
    "ex_11_flext_service": "examples.ex_11_flext_service",
    "ex_12_flext_registry": "examples.ex_12_flext_registry",
    "logging_config_once_pattern": "examples.logging_config_once_pattern",
    "m": ["examples.models", "FlextCoreExampleModels"],
    "main": "examples.logging_config_once_pattern",
    "models": "examples.models",
    "shared": "examples.shared",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
