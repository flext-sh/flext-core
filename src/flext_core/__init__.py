"""FLEXT Core - Foundation library for enterprise data integration.

Provides FlextResult for error handling, FlextContainer for DI,
and FlextEntity/FlextValueObject for domain modeling.

Main exports:
    - FlextResult: Railway-oriented programming
    - FlextContainer: Dependency injection
    - FlextEntity, FlextValueObject: Domain patterns
    - get_flext_container(): Global DI container

Example:
    from flext_core import FlextResult, get_flext_container

    result = FlextResult.ok("success")
    container = get_flext_container()

"""

from __future__ import annotations

# =============================================================================
# FOUNDATION LAYER - Import first, no dependencies on other modules
# =============================================================================

from flext_core.__version__ import *
from flext_core.constants import *
from flext_core.typings import *
from flext_core.result import *
from flext_core.exceptions import *
from flext_core.protocols import *

# =============================================================================
# DOMAIN LAYER - Depends only on Foundation layer
# =============================================================================

from flext_core.aggregate_root import *
from flext_core.domain_services import *
from flext_core.models import *

# =============================================================================
# APPLICATION LAYER - Depends on Domain + Foundation layers
# =============================================================================

from flext_core.commands import *
from flext_core.guards import *
from flext_core.handlers import *
from flext_core.validation import *

# =============================================================================
# INFRASTRUCTURE LAYER - Depends on Application + Domain + Foundation
# =============================================================================

# Infrastructure layer - explicit imports to avoid type conflicts
from flext_core.config import *
from flext_core.container import *
from flext_core.context import *
from flext_core.loggings import *
from flext_core.observability import *

# =============================================================================
# SUPPORT LAYER - Depends on layers as needed, imported last
# =============================================================================

from flext_core.decorators import *
from flext_core.delegation_system import *
from flext_core.fields import *
from flext_core.mixins import *
from flext_core.schema_processing import *
from flext_core.services import *
from flext_core.type_adapters import *
from flext_core.utilities import *


# =============================================================================
# LEGACY FUNCTIONALITY - Legacy implementation exports
# =============================================================================

# Legacy functionality - ensure specific exports are accessible
from flext_core.legacy import *


# =============================================================================
# CORE FUNCTIONALITY - Main implementation exports
# =============================================================================

# Core functionality - ensure specific exports are accessible
from flext_core.core import *


# =============================================================================
# CONSOLIDATED EXPORTS - Combine all __all__ from modules
# =============================================================================

# Combine all __all__ exports from imported modules
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
import flext_core.protocols as _protocols
import flext_core.result as _result
import flext_core.schema_processing as _schema_processing
import flext_core.services as _services
import flext_core.type_adapters as _type_adapters
import flext_core.typings as _typings
import flext_core.utilities as _utilities
import flext_core.validation as _validation
import flext_core.legacy as _legacy

# Collect all __all__ exports from imported modules
_temp_exports: list[str] = []

for module in [
    _version,
    _constants,
    _typings,
    _result,
    _exceptions,
    _protocols,
    _models,
    _aggregate_root,
    _domain_services,
    _commands,
    _validation,
    _guards,
    _handlers,
    _container,
    _config,
    _loggings,
    _observability,
    _context,
    _core,
    _mixins,
    _decorators,
    _utilities,
    _fields,
    _services,
    _delegation_system,
    _schema_processing,
    _type_adapters,
    _legacy,
]:
    if hasattr(module, "__all__"):
        _temp_exports.extend(module.__all__)

# Remove duplicates and sort for consistent exports - build complete list first
_seen: set[str] = set()
_final_exports: list[str] = []
for item in _temp_exports:
    if item not in _seen:
        _seen.add(item)
        _final_exports.append(item)
_final_exports.sort()

# Define __all__ as literal list for linter compatibility
# This dynamic assignment is necessary for aggregating module exports
__all__: list[str] = _final_exports  # pyright: ignore[reportUnsupportedDunderAll] # noqa: PLE0605
