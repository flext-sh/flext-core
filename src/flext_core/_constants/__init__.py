# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Internal module for FlextConstants nested classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._constants import (
        base,
        cqrs,
        domain,
        errors,
        infrastructure,
        mixins,
        platform,
        settings,
        validation,
    )
    from flext_core._constants.base import *
    from flext_core._constants.cqrs import *
    from flext_core._constants.domain import *
    from flext_core._constants.errors import *
    from flext_core._constants.infrastructure import *
    from flext_core._constants.mixins import *
    from flext_core._constants.platform import *
    from flext_core._constants.settings import *
    from flext_core._constants.validation import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "FlextConstantsBase": "flext_core._constants.base",
    "FlextConstantsCqrs": "flext_core._constants.cqrs",
    "FlextConstantsDomain": "flext_core._constants.domain",
    "FlextConstantsErrors": "flext_core._constants.errors",
    "FlextConstantsInfrastructure": "flext_core._constants.infrastructure",
    "FlextConstantsMixins": "flext_core._constants.mixins",
    "FlextConstantsPlatform": "flext_core._constants.platform",
    "FlextConstantsSettings": "flext_core._constants.settings",
    "FlextConstantsValidation": "flext_core._constants.validation",
    "base": "flext_core._constants.base",
    "cqrs": "flext_core._constants.cqrs",
    "domain": "flext_core._constants.domain",
    "errors": "flext_core._constants.errors",
    "infrastructure": "flext_core._constants.infrastructure",
    "mixins": "flext_core._constants.mixins",
    "platform": "flext_core._constants.platform",
    "settings": "flext_core._constants.settings",
    "validation": "flext_core._constants.validation",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
