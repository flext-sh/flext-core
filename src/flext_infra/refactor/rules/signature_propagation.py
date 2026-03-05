"""Rule that propagates signature migrations across callsites."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import libcst as cst
from libcst.metadata import MetadataWrapper

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.signature_propagator import (
    FlextInfraRefactorSignaturePropagator,
)


class FlextInfraRefactorSignaturePropagationRule(FlextInfraRefactorRule):
    """Apply declarative signature migrations in a generic, workspace-safe way."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        migrations_raw = self.config.get("signature_migrations", [])
        migrations = [
            cast("dict[str, object]", item)
            for item in cast("list[object]", migrations_raw)
            if isinstance(item, dict) and bool(item.get("enabled", True))
        ]
        if not migrations:
            return tree, []

        transformer = FlextInfraRefactorSignaturePropagator(migrations=migrations)
        wrapper = MetadataWrapper(tree)
        return wrapper.visit(transformer), transformer.changes


__all__ = ["FlextInfraRefactorSignaturePropagationRule"]
