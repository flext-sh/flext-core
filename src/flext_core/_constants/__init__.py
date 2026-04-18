# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".base": ("FlextConstantsBase",),
        ".cqrs": ("FlextConstantsCqrs",),
        ".domain": ("FlextConstantsDomain",),
        ".enforcement": ("FlextConstantsEnforcement",),
        ".errors": ("FlextConstantsErrors",),
        ".infrastructure": ("FlextConstantsInfrastructure",),
        ".mixins": ("FlextConstantsMixins",),
        ".output": ("FlextConstantsOutput",),
        ".platform": ("FlextConstantsPlatform",),
        ".project_metadata": ("FlextConstantsProjectMetadata",),
        ".pydantic": ("FlextConstantsPydantic",),
        ".settings": ("FlextConstantsSettings",),
        ".validation": ("FlextConstantsValidation",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
