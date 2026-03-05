"""Rule that propagates refactor API/module renames across callsites."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import libcst as cst
from libcst.metadata import MetadataWrapper

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.symbol_propagator import (
    FlextInfraRefactorSymbolPropagator,
)


class FlextInfraRefactorSymbolPropagationRule(FlextInfraRefactorRule):
    """Apply declarative module/symbol renames for workspace-wide propagation."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        target_modules_raw = self.config.get("target_modules", [])
        module_renames_raw = self.config.get("module_renames", {})
        symbol_renames_raw = self.config.get("import_symbol_renames", {})

        target_modules = {
            str(item)
            for item in cast("list[object]", target_modules_raw)
            if isinstance(item, str)
        }
        module_renames = {
            str(k): str(v)
            for k, v in cast("dict[object, object]", module_renames_raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }
        symbol_renames = {
            str(k): str(v)
            for k, v in cast("dict[object, object]", symbol_renames_raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }

        if not target_modules and not module_renames and not symbol_renames:
            return tree, []

        transformer = FlextInfraRefactorSymbolPropagator(
            target_modules=target_modules,
            module_renames=module_renames,
            import_symbol_renames=symbol_renames,
        )
        wrapper = MetadataWrapper(tree)
        return wrapper.visit(transformer), transformer.changes


__all__ = ["FlextInfraRefactorSymbolPropagationRule"]
