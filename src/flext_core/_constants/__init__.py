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
        base as base,
        cqrs as cqrs,
        domain as domain,
        errors as errors,
        infrastructure as infrastructure,
        mixins as mixins,
        platform as platform,
        settings as settings,
        validation as validation,
    )
    from flext_core._constants.base import FlextConstantsBase as FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs as FlextConstantsCqrs
    from flext_core._constants.domain import (
        FlextConstantsDomain as FlextConstantsDomain,
    )
    from flext_core._constants.errors import (
        FlextConstantsErrors as FlextConstantsErrors,
    )
    from flext_core._constants.infrastructure import (
        FlextConstantsInfrastructure as FlextConstantsInfrastructure,
    )
    from flext_core._constants.mixins import (
        FlextConstantsMixins as FlextConstantsMixins,
    )
    from flext_core._constants.platform import (
        FlextConstantsPlatform as FlextConstantsPlatform,
    )
    from flext_core._constants.settings import (
        FlextConstantsSettings as FlextConstantsSettings,
    )
    from flext_core._constants.validation import (
        FlextConstantsValidation as FlextConstantsValidation,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FlextConstantsBase": ["flext_core._constants.base", "FlextConstantsBase"],
    "FlextConstantsCqrs": ["flext_core._constants.cqrs", "FlextConstantsCqrs"],
    "FlextConstantsDomain": ["flext_core._constants.domain", "FlextConstantsDomain"],
    "FlextConstantsErrors": ["flext_core._constants.errors", "FlextConstantsErrors"],
    "FlextConstantsInfrastructure": [
        "flext_core._constants.infrastructure",
        "FlextConstantsInfrastructure",
    ],
    "FlextConstantsMixins": ["flext_core._constants.mixins", "FlextConstantsMixins"],
    "FlextConstantsPlatform": [
        "flext_core._constants.platform",
        "FlextConstantsPlatform",
    ],
    "FlextConstantsSettings": [
        "flext_core._constants.settings",
        "FlextConstantsSettings",
    ],
    "FlextConstantsValidation": [
        "flext_core._constants.validation",
        "FlextConstantsValidation",
    ],
    "base": ["flext_core._constants.base", ""],
    "cqrs": ["flext_core._constants.cqrs", ""],
    "domain": ["flext_core._constants.domain", ""],
    "errors": ["flext_core._constants.errors", ""],
    "infrastructure": ["flext_core._constants.infrastructure", ""],
    "mixins": ["flext_core._constants.mixins", ""],
    "platform": ["flext_core._constants.platform", ""],
    "settings": ["flext_core._constants.settings", ""],
    "validation": ["flext_core._constants.validation", ""],
}

_EXPORTS: Sequence[str] = [
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsErrors",
    "FlextConstantsInfrastructure",
    "FlextConstantsMixins",
    "FlextConstantsPlatform",
    "FlextConstantsSettings",
    "FlextConstantsValidation",
    "base",
    "cqrs",
    "domain",
    "errors",
    "infrastructure",
    "mixins",
    "platform",
    "settings",
    "validation",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
