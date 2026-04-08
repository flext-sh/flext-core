# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Constants package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports

_LAZY_IMPORTS = {
    "FlextConstantsBase": ".base",
    "FlextConstantsCqrs": ".cqrs",
    "FlextConstantsDomain": ".domain",
    "FlextConstantsEnforcement": ".enforcement",
    "FlextConstantsErrors": ".errors",
    "FlextConstantsInfrastructure": ".infrastructure",
    "FlextConstantsMixins": ".mixins",
    "FlextConstantsPlatform": ".platform",
    "FlextConstantsSettings": ".settings",
    "FlextConstantsValidation": ".validation",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
