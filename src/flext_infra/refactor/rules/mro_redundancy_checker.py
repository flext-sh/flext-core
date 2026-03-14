"""Rule that removes nested MRO redeclarations."""

from __future__ import annotations

from pathlib import Path
from typing import override

import libcst as cst

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.mro_remover import FlextInfraRefactorMRORemover


class FlextInfraRefactorMRORedundancyChecker(FlextInfraRefactorRule):
    """Detect and fix nested classes inheriting from their parent namespace."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply MRO redeclaration cleanup transformer."""
        transformer = FlextInfraRefactorMRORemover()
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)


__all__ = ["FlextInfraRefactorMRORedundancyChecker"]
