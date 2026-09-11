"""Project-selection mixin for flext-infra service bases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra import c, m, p, t
from flext_infra._utilities.base import FlextInfraUtilitiesBase as ub
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs


class FlextInfraProjectSelectionMixin:
    """Private project-selection behavior for project-scoped services."""

    if TYPE_CHECKING:
        selected_projects: t.StrSequence | None

    @property
    def root(self) -> Path:
        """Repository root supplied by the composed service base."""
        raise NotImplementedError

    @property
    def project_names(self) -> t.StrSequence | None:
        """Normalized selected project names."""
        return ub.normalize_sequence_values(self.selected_projects)

    @property
    def project_dirs(self) -> t.SequenceOf[Path] | None:
        """Resolve selected project directories relative to the repository root."""
        names = ub.normalize_sequence_values(self.selected_projects)
        if names is None:
            return None
        return [self.root / name for name in names]

    def run_scoped_docs(
        self,
        repository_root: Path,
        *,
        output_dir: Path | str | None,
        handler: Callable[[m.Infra.DocScope], m.Infra.DocsPhaseReport],
        projects: t.StrSequence | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.DocsPhaseReport]]:
        """Run one docs phase across the resolved governed scopes."""
        return FlextInfraUtilitiesDocs.run_scoped(
            repository_root,
            projects=self.selected_projects if projects is None else projects,
            output_dir=u.Cli.resolve_optional_path(
                output_dir, default=Path(c.Infra.DEFAULT_DOCS_OUTPUT_DIR)
            ),
            handler=handler,
        )


__all__: list[str] = ["FlextInfraProjectSelectionMixin"]
