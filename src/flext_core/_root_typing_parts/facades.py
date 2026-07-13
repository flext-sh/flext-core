"""Generated root type-checking imports: facades."""

from __future__ import annotations

from typing import Final

# NOTE (multi-agent, mro-f8vk / kimi): facade imports must resolve from their
# owner modules (flext_core.constants/.models/.protocols/.typings/.utilities),
# never from the root package — self-importing flext_core here degraded every
# workspace `from flext_core import c/m/p/t/u` to Module("flext_core") under
# TYPE_CHECKING (pyright: "Argument to class must be a base class"). Generator
# propagation is tracked in mro-i6nq.10.
from flext_core._settings import (
    FlextSettings,
    settings,
)
from flext_core.constants import (
    FlextConstants,
    c,
)
from flext_core.container import (
    FlextContainer,
)
from flext_core.context import (
    FlextContext,
)
from flext_core.decorators import (
    FlextDecorators,
    d,
)
from flext_core.dispatcher import (
    FlextDispatcher,
)
from flext_core.exceptions import (
    FlextExceptions,
    e,
)
from flext_core.handlers import (
    FlextHandlers,
    h,
)
from flext_core.lazy import (
    FlextLazy,
    build_lazy_import_map,
    lazy,
    normalize_lazy_imports,
)
from flext_core.loggings import (
    FlextUtilitiesLogging,
)
from flext_core.mixins import (
    FlextMixins,
    x,
)
from flext_core.models import (
    FlextModels,
    m,
)
from flext_core.protocols import (
    FlextProtocols,
    p,
)
from flext_core.registry import (
    FlextRegistry,
)
from flext_core.result import (
    FlextResult,
    r,
)
from flext_core.runtime import (
    FlextRuntime,
)
from flext_core.service import (
    FlextService,
    s,
)
from flext_core.typings import (
    FlextTypes,
    t,
)
from flext_core.utilities import (
    FlextUtilities,
    u,
)

ROOT_TYPING_FACADES_ALL: Final[tuple[str, ...]] = (
    "FlextConstants",
    "c",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "d",
    "FlextDispatcher",
    "FlextExceptions",
    "e",
    "FlextHandlers",
    "h",
    "FlextLazy",
    "build_lazy_import_map",
    "lazy",
    "normalize_lazy_imports",
    "FlextUtilitiesLogging",
    "FlextMixins",
    "x",
    "FlextModels",
    "m",
    "FlextProtocols",
    "p",
    "FlextRegistry",
    "FlextResult",
    "r",
    "FlextRuntime",
    "FlextService",
    "s",
    "FlextSettings",
    "FlextTypes",
    "t",
    "FlextUtilities",
    "u",
)
__all__: tuple[str, ...] = (
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
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
    "FlextUtilitiesLogging",
    "build_lazy_import_map",
    "c",
    "d",
    "e",
    "h",
    "lazy",
    "m",
    "normalize_lazy_imports",
    "p",
    "r",
    "s",
    "settings",
    "t",
    "u",
    "x",
)
