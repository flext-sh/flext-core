"""Transformer classes for flext_infra.refactor."""

from __future__ import annotations

from flext_infra.refactor.transformers.alias_remover import (
    FlextInfraRefactorAliasRemover,
)
from flext_infra.refactor.transformers.class_nesting import ClassNestingTransformer
from flext_infra.refactor.transformers.class_reconstructor import (
    FlextInfraRefactorClassReconstructor,
)
from flext_infra.refactor.transformers.deprecated_remover import (
    FlextInfraRefactorDeprecatedRemover,
)
from flext_infra.refactor.transformers.import_bypass_remover import (
    FlextInfraRefactorImportBypassRemover,
)
from flext_infra.refactor.transformers.import_modernizer import (
    FlextInfraRefactorImportModernizer,
)
from flext_infra.refactor.transformers.lazy_import_fixer import (
    FlextInfraRefactorLazyImportFixer,
)
from flext_infra.refactor.transformers.mro_remover import FlextInfraRefactorMRORemover
from flext_infra.refactor.transformers.symbol_propagator import (
    FlextInfraRefactorSymbolPropagator,
)

__all__ = [
    "ClassNestingTransformer",
    "FlextInfraRefactorAliasRemover",
    "FlextInfraRefactorClassReconstructor",
    "FlextInfraRefactorDeprecatedRemover",
    "FlextInfraRefactorImportBypassRemover",
    "FlextInfraRefactorImportModernizer",
    "FlextInfraRefactorLazyImportFixer",
    "FlextInfraRefactorMRORemover",
    "FlextInfraRefactorSymbolPropagator",
]
