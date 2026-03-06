"""Public API for flext_infra.refactor with lazy loading."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra.refactor.analysis import FlextInfraRefactorViolationAnalyzer
    from flext_infra.refactor.engine import FlextInfraRefactorEngine
    from flext_infra.refactor.rule import FlextInfraRefactorRule
    from flext_infra.refactor.rules.class_reconstructor import (
        FlextInfraRefactorClassReconstructorRule,
    )
    from flext_infra.refactor.rules.ensure_future_annotations import (
        FlextInfraRefactorEnsureFutureAnnotationsRule,
    )
    from flext_infra.refactor.rules.import_modernizer import (
        FlextInfraRefactorImportModernizerRule,
    )
    from flext_infra.refactor.rules.legacy_removal import (
        FlextInfraRefactorLegacyRemovalRule,
    )
    from flext_infra.refactor.rules.mro_redundancy_checker import (
        FlextInfraRefactorMRORedundancyChecker,
    )
    from flext_infra.refactor.rules.pattern_corrections import (
        FlextInfraRefactorPatternCorrectionsRule,
    )
    from flext_infra.refactor.rules.symbol_propagation import (
        FlextInfraRefactorSignaturePropagationRule,
        FlextInfraRefactorSymbolPropagationRule,
    )

_LAZY_IMPORTS: Final[dict[str, tuple[str, str]]] = {
    "FlextInfraRefactorEngine": (
        "flext_infra.refactor.engine",
        "FlextInfraRefactorEngine",
    ),
    "FlextInfraRefactorRule": (
        "flext_infra.refactor.rule",
        "FlextInfraRefactorRule",
    ),
    "FlextInfraRefactorClassReconstructorRule": (
        "flext_infra.refactor.rules.class_reconstructor",
        "FlextInfraRefactorClassReconstructorRule",
    ),
    "FlextInfraRefactorEnsureFutureAnnotationsRule": (
        "flext_infra.refactor.rules.ensure_future_annotations",
        "FlextInfraRefactorEnsureFutureAnnotationsRule",
    ),
    "FlextInfraRefactorImportModernizerRule": (
        "flext_infra.refactor.rules.import_modernizer",
        "FlextInfraRefactorImportModernizerRule",
    ),
    "FlextInfraRefactorLegacyRemovalRule": (
        "flext_infra.refactor.rules.legacy_removal",
        "FlextInfraRefactorLegacyRemovalRule",
    ),
    "FlextInfraRefactorMRORedundancyChecker": (
        "flext_infra.refactor.rules.mro_redundancy_checker",
        "FlextInfraRefactorMRORedundancyChecker",
    ),
    "FlextInfraRefactorSymbolPropagationRule": (
        "flext_infra.refactor.rules.symbol_propagation",
        "FlextInfraRefactorSymbolPropagationRule",
    ),
    "FlextInfraRefactorSignaturePropagationRule": (
        "flext_infra.refactor.rules.symbol_propagation",
        "FlextInfraRefactorSignaturePropagationRule",
    ),
    "FlextInfraRefactorPatternCorrectionsRule": (
        "flext_infra.refactor.rules.pattern_corrections",
        "FlextInfraRefactorPatternCorrectionsRule",
    ),
    "FlextInfraRefactorViolationAnalyzer": (
        "flext_infra.refactor.analysis",
        "FlextInfraRefactorViolationAnalyzer",
    ),
    "FlextInfraRefactorMigrateToClassMRO": (
        "flext_infra.refactor.migrate_to_class_mro",
        "FlextInfraRefactorMigrateToClassMRO",
    ),
}


def __getattr__(name: str) -> object:
    if name not in _LAZY_IMPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, symbol_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


__all__ = [
    "FlextInfraRefactorClassReconstructorRule",
    "FlextInfraRefactorEngine",
    "FlextInfraRefactorEnsureFutureAnnotationsRule",
    "FlextInfraRefactorImportModernizerRule",
    "FlextInfraRefactorLegacyRemovalRule",
    "FlextInfraRefactorMRORedundancyChecker",
    "FlextInfraRefactorPatternCorrectionsRule",
    "FlextInfraRefactorRule",
    "FlextInfraRefactorSignaturePropagationRule",
    "FlextInfraRefactorSymbolPropagationRule",
    "FlextInfraRefactorViolationAnalyzer",
]
