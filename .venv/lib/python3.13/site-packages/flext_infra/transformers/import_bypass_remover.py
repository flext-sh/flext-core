"""Import bypass remover transformer — rope-based implementation.

Replaces try/except ImportError fallback blocks with the primary import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorImportBypassRemover(FlextInfraRopeTransformer):
    """Replace import bypass try/except blocks with the primary import."""

    @override
    def transform(
        self, rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Remove try/except ImportError fallback blocks, keep the primary import."""
        source = resource.read()
        new_source, count = c.Infra.IMPORT_BYPASS_RE.subn(r"\1", source)
        if count == 0:
            no_changes: list[str] = []
            return source, no_changes
        changes = [f"Removed {count} import bypass fallback(s)"]
        for msg in changes:
            self._record_change(msg)
        resource.write(new_source)
        return new_source, list(self.changes)


__all__: list[str] = ["FlextInfraRefactorImportBypassRemover"]
