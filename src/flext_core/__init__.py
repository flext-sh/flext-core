"""FLEXT Core foundation library."""

from __future__ import annotations

# ruff: noqa: F403
# Import all from each module
from flext_core.__version__ import *
from flext_core.aggregate_root import *
from flext_core.commands import *
from flext_core.config import *
from flext_core.constants import *
from flext_core.container import *  # type: ignore[assignment]
from flext_core.context import *
from flext_core.core import *  # type: ignore[assignment]
from flext_core.decorators import *  # type: ignore[assignment]
from flext_core.delegation_system import *
from flext_core.domain_services import *  # type: ignore[assignment]
from flext_core.exceptions import *
from flext_core.fields import *
from flext_core.guards import *
from flext_core.handlers import *
from flext_core.loggings import *
from flext_core.mixins import *
from flext_core.models import *  # type: ignore[assignment]
from flext_core.observability import *
from flext_core.payload import *
from flext_core.protocols import *
from flext_core.result import *
from flext_core.root_models import *
from flext_core.schema_processing import *
from flext_core.services import *
from flext_core.type_adapters import *
from flext_core.typings import *  # type: ignore[assignment]
from flext_core.utilities import *
from flext_core.validation import *

# Explicit imports are handled by wildcard imports above

# Note: __all__ is constructed dynamically at runtime
# This pattern is necessary for library aggregation but causes pyright warnings
__all__: list[str] = []
