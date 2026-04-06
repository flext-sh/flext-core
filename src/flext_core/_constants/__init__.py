# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Constants package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._constants.base as _flext_core__constants_base

    base = _flext_core__constants_base
    import flext_core._constants.cqrs as _flext_core__constants_cqrs
    from flext_core._constants.base import FlextConstantsBase

    cqrs = _flext_core__constants_cqrs
    import flext_core._constants.domain as _flext_core__constants_domain
    from flext_core._constants.cqrs import FlextConstantsCqrs

    domain = _flext_core__constants_domain
    import flext_core._constants.enforcement as _flext_core__constants_enforcement
    from flext_core._constants.domain import FlextConstantsDomain

    enforcement = _flext_core__constants_enforcement
    import flext_core._constants.errors as _flext_core__constants_errors
    from flext_core._constants.enforcement import FlextConstantsEnforcement

    errors = _flext_core__constants_errors
    import flext_core._constants.infrastructure as _flext_core__constants_infrastructure
    from flext_core._constants.errors import FlextConstantsErrors

    infrastructure = _flext_core__constants_infrastructure
    import flext_core._constants.mixins as _flext_core__constants_mixins
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure

    mixins = _flext_core__constants_mixins
    import flext_core._constants.platform as _flext_core__constants_platform
    from flext_core._constants.mixins import FlextConstantsMixins

    platform = _flext_core__constants_platform
    import flext_core._constants.settings as _flext_core__constants_settings
    from flext_core._constants.platform import FlextConstantsPlatform

    settings = _flext_core__constants_settings
    import flext_core._constants.validation as _flext_core__constants_validation
    from flext_core._constants.settings import FlextConstantsSettings

    validation = _flext_core__constants_validation
    from flext_core._constants.validation import FlextConstantsValidation
_LAZY_IMPORTS = {
    "FlextConstantsBase": ("flext_core._constants.base", "FlextConstantsBase"),
    "FlextConstantsCqrs": ("flext_core._constants.cqrs", "FlextConstantsCqrs"),
    "FlextConstantsDomain": ("flext_core._constants.domain", "FlextConstantsDomain"),
    "FlextConstantsEnforcement": (
        "flext_core._constants.enforcement",
        "FlextConstantsEnforcement",
    ),
    "FlextConstantsErrors": ("flext_core._constants.errors", "FlextConstantsErrors"),
    "FlextConstantsInfrastructure": (
        "flext_core._constants.infrastructure",
        "FlextConstantsInfrastructure",
    ),
    "FlextConstantsMixins": ("flext_core._constants.mixins", "FlextConstantsMixins"),
    "FlextConstantsPlatform": (
        "flext_core._constants.platform",
        "FlextConstantsPlatform",
    ),
    "FlextConstantsSettings": (
        "flext_core._constants.settings",
        "FlextConstantsSettings",
    ),
    "FlextConstantsValidation": (
        "flext_core._constants.validation",
        "FlextConstantsValidation",
    ),
    "base": "flext_core._constants.base",
    "cqrs": "flext_core._constants.cqrs",
    "domain": "flext_core._constants.domain",
    "enforcement": "flext_core._constants.enforcement",
    "errors": "flext_core._constants.errors",
    "infrastructure": "flext_core._constants.infrastructure",
    "mixins": "flext_core._constants.mixins",
    "platform": "flext_core._constants.platform",
    "settings": "flext_core._constants.settings",
    "validation": "flext_core._constants.validation",
}

__all__ = [
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsEnforcement",
    "FlextConstantsErrors",
    "FlextConstantsInfrastructure",
    "FlextConstantsMixins",
    "FlextConstantsPlatform",
    "FlextConstantsSettings",
    "FlextConstantsValidation",
    "base",
    "cqrs",
    "domain",
    "enforcement",
    "errors",
    "infrastructure",
    "mixins",
    "platform",
    "settings",
    "validation",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
