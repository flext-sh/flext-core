"""Rule that reorders class methods based on configured method order."""

from __future__ import annotations

from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import t
from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.class_reconstructor import (
    FlextInfraRefactorClassReconstructor,
)


class FlextInfraRefactorClassReconstructorRule(FlextInfraRefactorRule):
    """Apply class method ordering reconstruction to matching class nodes."""

    @override
    def apply(
        self, tree: cst.Module, _file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        """Apply method reordering transformer when order config is available."""
        order_config_raw = self.config.get("method_order") or self.config.get(
            "order", []
        )
        try:
            order_config = TypeAdapter(list[t.Infra.RuleConfig]).validate_python(
                order_config_raw
            )
        except ValidationError:
            return (tree, [])
        if not order_config:
            return (tree, [])
        transformer = FlextInfraRefactorClassReconstructor(order_config=order_config)
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)


__all__ = ["FlextInfraRefactorClassReconstructorRule"]
