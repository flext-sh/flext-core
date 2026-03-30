# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core._constants import _LAZY_IMPORTS as _CHILD_LAZY_0
from flext_core._models import _LAZY_IMPORTS as _CHILD_LAZY_1
from flext_core._protocols import _LAZY_IMPORTS as _CHILD_LAZY_2
from flext_core._typings import _LAZY_IMPORTS as _CHILD_LAZY_3
from flext_core._utilities import _LAZY_IMPORTS as _CHILD_LAZY_4
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core.__version__ import *
    from flext_core._constants import *
    from flext_core._models import *
    from flext_core._models._context import *
    from flext_core._protocols import *
    from flext_core._typings import *
    from flext_core._utilities import *
    from flext_core.constants import *
    from flext_core.container import *
    from flext_core.context import *
    from flext_core.decorators import *
    from flext_core.dispatcher import *
    from flext_core.exceptions import *
    from flext_core.handlers import *
    from flext_core.loggings import *
    from flext_core.mixins import *
    from flext_core.models import *
    from flext_core.protocols import *
    from flext_core.registry import *
    from flext_core.result import *
    from flext_core.runtime import *
    from flext_core.service import *
    from flext_core.settings import *
    from flext_core.typings import *
    from flext_core.utilities import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    **_CHILD_LAZY_0,
    **_CHILD_LAZY_1,
    **_CHILD_LAZY_2,
    **_CHILD_LAZY_3,
    **_CHILD_LAZY_4,
    "BaseModel": "flext_core.typings",
    "FlextConstants": "flext_core.constants",
    "FlextContainer": "flext_core.container",
    "FlextContext": "flext_core.context",
    "FlextDecorators": "flext_core.decorators",
    "FlextDispatcher": "flext_core.dispatcher",
    "FlextExceptions": "flext_core.exceptions",
    "FlextHandlers": "flext_core.handlers",
    "FlextLogger": "flext_core.loggings",
    "FlextMixins": "flext_core.mixins",
    "FlextModels": "flext_core.models",
    "FlextProtocols": "flext_core.protocols",
    "FlextRegistry": "flext_core.registry",
    "FlextResult": "flext_core.result",
    "FlextRuntime": "flext_core.runtime",
    "FlextService": "flext_core.service",
    "FlextSettings": "flext_core.settings",
    "FlextTypes": "flext_core.typings",
    "FlextUtilities": "flext_core.utilities",
    "FlextVersion": "flext_core.__version__",
    "__author__": "flext_core.__version__",
    "__author_email__": "flext_core.__version__",
    "__description__": "flext_core.__version__",
    "__license__": "flext_core.__version__",
    "__title__": "flext_core.__version__",
    "__url__": "flext_core.__version__",
    "__version__": "flext_core.__version__",
    "__version_info__": "flext_core.__version__",
    "_constants": "flext_core._constants",
    "_models": "flext_core._models",
    "_protocols": "flext_core._protocols",
    "_typings": "flext_core._typings",
    "_utilities": "flext_core._utilities",
    "c": ["flext_core.constants", "FlextConstants"],
    "constants": "flext_core.constants",
    "container": "flext_core.container",
    "context": "flext_core.context",
    "d": ["flext_core.decorators", "FlextDecorators"],
    "decorators": "flext_core.decorators",
    "dispatcher": "flext_core.dispatcher",
    "e": ["flext_core.exceptions", "FlextExceptions"],
    "exceptions": "flext_core.exceptions",
    "h": ["flext_core.handlers", "FlextHandlers"],
    "handlers": "flext_core.handlers",
    "lazy": "flext_core.lazy",
    "loggings": "flext_core.loggings",
    "m": ["flext_core.models", "FlextModels"],
    "mixins": "flext_core.mixins",
    "models": "flext_core.models",
    "p": ["flext_core.protocols", "FlextProtocols"],
    "protocols": "flext_core.protocols",
    "r": ["flext_core.result", "FlextResult"],
    "registry": "flext_core.registry",
    "result": "flext_core.result",
    "runtime": "flext_core.runtime",
    "s": ["flext_core.service", "FlextService"],
    "service": "flext_core.service",
    "settings": "flext_core.settings",
    "t": ["flext_core.typings", "FlextTypes"],
    "typings": "flext_core.typings",
    "u": ["flext_core.utilities", "FlextUtilities"],
    "utilities": "flext_core.utilities",
    "x": ["flext_core.mixins", "FlextMixins"],
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
