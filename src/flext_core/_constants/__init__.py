# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Constants package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
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
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.errors import FlextConstantsErrors
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.platform import FlextConstantsPlatform
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.validation import FlextConstantsValidation

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
