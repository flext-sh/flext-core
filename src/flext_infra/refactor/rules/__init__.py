"""Rule classes for flext_infra.refactor."""

from __future__ import annotations

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

__all__ = [
    "FlextInfraRefactorClassReconstructorRule",
    "FlextInfraRefactorEnsureFutureAnnotationsRule",
    "FlextInfraRefactorImportModernizerRule",
    "FlextInfraRefactorLegacyRemovalRule",
    "FlextInfraRefactorMRORedundancyChecker",
    "FlextInfraRefactorPatternCorrectionsRule",
    "FlextInfraRefactorSignaturePropagationRule",
    "FlextInfraRefactorSymbolPropagationRule",
]
