# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal module for FlextConstants nested classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.platform import FlextConstantsPlatform
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.validation import FlextConstantsValidation
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextConstantsBase": ("flext_core._constants.base", "FlextConstantsBase"),
    "FlextConstantsCqrs": ("flext_core._constants.cqrs", "FlextConstantsCqrs"),
    "FlextConstantsDomain": ("flext_core._constants.domain", "FlextConstantsDomain"),
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
}

__all__ = [
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsInfrastructure",
    "FlextConstantsMixins",
    "FlextConstantsPlatform",
    "FlextConstantsSettings",
    "FlextConstantsValidation",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
