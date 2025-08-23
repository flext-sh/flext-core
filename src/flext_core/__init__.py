"""FLEXT Core foundation library."""

from __future__ import annotations

# Import all from each module
from flext_core.__version__ import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.aggregate_root import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.commands import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.config import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.constants import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.container import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.context import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.core import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.decorators import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.delegation_system import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.domain_services import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.exceptions import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.fields import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.guards import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.handlers import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.loggings import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.mixins import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.models import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.observability import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.payload import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.protocols import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.result import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.root_models import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.schema_processing import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.services import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.type_adapters import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.typings import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.utilities import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403
from flext_core.validation import *  # type: ignore[unused-ignore,reportWildcardImport,assignment] # noqa: F403

# Combine all __all__ from all modules
import flext_core.__version__ as _version
import flext_core.aggregate_root as _aggregate_root
import flext_core.commands as _commands
import flext_core.config as _config
import flext_core.constants as _constants
import flext_core.container as _container
import flext_core.context as _context
import flext_core.core as _core
import flext_core.decorators as _decorators
import flext_core.delegation_system as _delegation_system
import flext_core.domain_services as _domain_services
import flext_core.exceptions as _exceptions
import flext_core.fields as _fields
import flext_core.guards as _guards
import flext_core.handlers as _handlers
import flext_core.loggings as _loggings
import flext_core.mixins as _mixins
import flext_core.models as _models
import flext_core.observability as _observability
import flext_core.payload as _payload
import flext_core.protocols as _protocols
import flext_core.result as _result
import flext_core.root_models as _root_models
import flext_core.schema_processing as _schema_processing
import flext_core.services as _services
import flext_core.type_adapters as _type_adapters
import flext_core.typings as _typings
import flext_core.utilities as _utilities
import flext_core.validation as _validation

__all__ = []
for module in [
    _version,
    _aggregate_root,
    _commands,
    _config,
    _constants,
    _container,
    _context,
    _core,
    _decorators,
    _delegation_system,
    _domain_services,
    _exceptions,
    _fields,
    _guards,
    _handlers,
    _loggings,
    _mixins,
    _models,
    _observability,
    _payload,
    _protocols,
    _result,
    _root_models,
    _schema_processing,
    _services,
    _type_adapters,
    _typings,
    _utilities,
    _validation,
]:
    if hasattr(module, "__all__"):
        __all__ += module.__all__  # type: ignore[unused-ignore,reportUnsupportedDunderAll] # noqa: PYI056

# Remove duplicates and sort
__all__ = sorted(set(__all__))  # type: ignore[unused-ignore,reportUnsupportedDunderAll,reportUnknownArgumentType] # noqa: PLE0605
