"""Rule that reorders class methods based on configured method order."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, override

import libcst as cst

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.class_reconstructor import (
    FlextInfraRefactorClassReconstructor,
)


class FlextInfraRefactorClassReconstructorRule(FlextInfraRefactorRule):
    """Apply class method ordering reconstruction to matching class nodes."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply method reordering transformer when order config is available."""
        order_config_raw = self.config.get("method_order") or self.config.get(
            "order", []
        )
        if not isinstance(order_config_raw, list) or not order_config_raw:
            return tree, []

        order_config = [
            cast("dict[str, Any]", item)
            for item in cast("list[object]", order_config_raw)
            if isinstance(item, dict)
        ]

        if not order_config:
            return tree, []

        transformer = FlextInfraRefactorClassReconstructor(order_config=order_config)
        new_tree = tree.visit(transformer)
        return new_tree, transformer.changes


__all__ = ["FlextInfraRefactorClassReconstructorRule"]
