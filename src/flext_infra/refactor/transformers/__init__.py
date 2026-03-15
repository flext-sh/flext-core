# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Transformer classes for flext_infra.refactor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.refactor.transformers.alias_remover import (
        FlextInfraRefactorAliasRemover,
    )
    from flext_infra.refactor.transformers.census_visitors import (
        CensusImportDiscoveryVisitor,
        CensusUsageCollector,
    )
    from flext_infra.refactor.transformers.class_nesting import (
        FlextInfraRefactorClassNestingTransformer,
    )
    from flext_infra.refactor.transformers.class_reconstructor import (
        FlextInfraRefactorClassReconstructor,
    )
    from flext_infra.refactor.transformers.deprecated_remover import (
        FlextInfraRefactorDeprecatedRemover,
    )
    from flext_infra.refactor.transformers.helper_consolidation import (
        HelperConsolidationTransformer,
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
    from flext_infra.refactor.transformers.mro_private_inline import (
        FlextInfraRefactorMROPrivateInlineTransformer,
        FlextInfraRefactorMROQualifiedReferenceTransformer,
    )
    from flext_infra.refactor.transformers.mro_reference_rewriter import (
        FlextInfraRefactorMROReferenceRewriter,
    )
    from flext_infra.refactor.transformers.mro_remover import (
        FlextInfraRefactorMRORemover,
    )
    from flext_infra.refactor.transformers.nested_class_propagation import (
        NestedClassPropagationTransformer,
    )
    from flext_infra.refactor.transformers.policy import (
        FlextInfraRefactorTransformerPolicyUtilities,
        FlextInfraRefactorTransformerPolicyUtilities as u,
    )
    from flext_infra.refactor.transformers.symbol_propagator import (
        FlextInfraRefactorSymbolPropagator,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CensusImportDiscoveryVisitor": (
        "flext_infra.refactor.transformers.census_visitors",
        "CensusImportDiscoveryVisitor",
    ),
    "CensusUsageCollector": (
        "flext_infra.refactor.transformers.census_visitors",
        "CensusUsageCollector",
    ),
    "FlextInfraRefactorAliasRemover": (
        "flext_infra.refactor.transformers.alias_remover",
        "FlextInfraRefactorAliasRemover",
    ),
    "FlextInfraRefactorClassNestingTransformer": (
        "flext_infra.refactor.transformers.class_nesting",
        "FlextInfraRefactorClassNestingTransformer",
    ),
    "FlextInfraRefactorClassReconstructor": (
        "flext_infra.refactor.transformers.class_reconstructor",
        "FlextInfraRefactorClassReconstructor",
    ),
    "FlextInfraRefactorDeprecatedRemover": (
        "flext_infra.refactor.transformers.deprecated_remover",
        "FlextInfraRefactorDeprecatedRemover",
    ),
    "FlextInfraRefactorImportBypassRemover": (
        "flext_infra.refactor.transformers.import_bypass_remover",
        "FlextInfraRefactorImportBypassRemover",
    ),
    "FlextInfraRefactorImportModernizer": (
        "flext_infra.refactor.transformers.import_modernizer",
        "FlextInfraRefactorImportModernizer",
    ),
    "FlextInfraRefactorLazyImportFixer": (
        "flext_infra.refactor.transformers.lazy_import_fixer",
        "FlextInfraRefactorLazyImportFixer",
    ),
    "FlextInfraRefactorMROPrivateInlineTransformer": (
        "flext_infra.refactor.transformers.mro_private_inline",
        "FlextInfraRefactorMROPrivateInlineTransformer",
    ),
    "FlextInfraRefactorMROQualifiedReferenceTransformer": (
        "flext_infra.refactor.transformers.mro_private_inline",
        "FlextInfraRefactorMROQualifiedReferenceTransformer",
    ),
    "FlextInfraRefactorMROReferenceRewriter": (
        "flext_infra.refactor.transformers.mro_reference_rewriter",
        "FlextInfraRefactorMROReferenceRewriter",
    ),
    "FlextInfraRefactorMRORemover": (
        "flext_infra.refactor.transformers.mro_remover",
        "FlextInfraRefactorMRORemover",
    ),
    "FlextInfraRefactorSymbolPropagator": (
        "flext_infra.refactor.transformers.symbol_propagator",
        "FlextInfraRefactorSymbolPropagator",
    ),
    "FlextInfraRefactorTransformerPolicyUtilities": (
        "flext_infra.refactor.transformers.policy",
        "FlextInfraRefactorTransformerPolicyUtilities",
    ),
    "HelperConsolidationTransformer": (
        "flext_infra.refactor.transformers.helper_consolidation",
        "HelperConsolidationTransformer",
    ),
    "NestedClassPropagationTransformer": (
        "flext_infra.refactor.transformers.nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "u": (
        "flext_infra.refactor.transformers.policy",
        "FlextInfraRefactorTransformerPolicyUtilities",
    ),
}

__all__ = [
    "CensusImportDiscoveryVisitor",
    "CensusUsageCollector",
    "FlextInfraRefactorAliasRemover",
    "FlextInfraRefactorClassNestingTransformer",
    "FlextInfraRefactorClassReconstructor",
    "FlextInfraRefactorDeprecatedRemover",
    "FlextInfraRefactorImportBypassRemover",
    "FlextInfraRefactorImportModernizer",
    "FlextInfraRefactorLazyImportFixer",
    "FlextInfraRefactorMROPrivateInlineTransformer",
    "FlextInfraRefactorMROQualifiedReferenceTransformer",
    "FlextInfraRefactorMROReferenceRewriter",
    "FlextInfraRefactorMRORemover",
    "FlextInfraRefactorSymbolPropagator",
    "FlextInfraRefactorTransformerPolicyUtilities",
    "HelperConsolidationTransformer",
    "NestedClassPropagationTransformer",
    "u",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
