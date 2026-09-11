"""Documentation dev-server service.

``serve`` previews one MkDocs site locally (blocking dev server with
livereload). It is intentionally single-scope: a dev server binds one
address, so when several governed scopes carry an ``mkdocs.yml`` the
caller narrows the selection with ``--project``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.docs.base import FlextInfraDocServiceBase

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraDocServer(FlextInfraDocServiceBase):
    """Serve one MkDocs site in dev mode (blocking local preview)."""

    dev_addr: Annotated[
        str, m.Field(description="Dev server bind address (host:port)")
    ] = "127.0.0.1:8000"
    livereload: Annotated[bool, m.Field(description="Enable MkDocs livereload")] = True
    strict: Annotated[bool, m.Field(description="Enable MkDocs strict mode")] = True

    def serve(
        self,
        repository_root: Path,
        *,
        projects: t.StrSequence | None = None,
        output_dir: Path | str | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.DocsPhaseReport]]:
        """Serve the single matching docs scope (blocks until stopped)."""
        scopes_result = u.Infra.build_scopes(
            repository_root,
            projects=self.selected_projects if projects is None else projects,
            output_dir=output_dir or c.Infra.DEFAULT_DOCS_OUTPUT_DIR,
        )
        if scopes_result.failure:
            return r[t.SequenceOf[m.Infra.DocsPhaseReport]].fail(
                scopes_result.error or "scope resolution failed"
            )
        servable = [
            scope
            for scope in scopes_result.value
            if (scope.path / "mkdocs.yml").exists()
        ]
        if not servable:
            return r[t.SequenceOf[m.Infra.DocsPhaseReport]].fail(
                "no docs scope with mkdocs.yml; run `docs generate` first"
            )
        if len(servable) > 1:
            names = ", ".join(scope.name for scope in servable)
            return r[t.SequenceOf[m.Infra.DocsPhaseReport]].fail(
                "serve targets exactly one scope; narrow with --project "
                f"(candidates: {names})"
            )
        return r[t.SequenceOf[m.Infra.DocsPhaseReport]].ok([
            self._serve_scope(servable[0])
        ])

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the configured docs serve flow."""
        return self._propagate_phase_outcome(
            "serve",
            self.serve(
                repository_root=self.repository_root,
                projects=self.selected_projects,
                output_dir=self.output_dir,
            ),
            failure_predicate=lambda report: report.result == c.Infra.ResultStatus.FAIL,
        )

    def _serve_scope(self, scope: m.Infra.DocScope) -> m.Infra.DocsPhaseReport:
        """Serve one scope through the docs build utilities (blocking)."""
        self.logger.info(
            "docs_serve_scope_started", project=scope.name, dev_addr=self.dev_addr
        )
        report: m.Infra.DocsPhaseReport = u.Infra.docs_serve_mkdocs(
            scope,
            dev_addr=self.dev_addr,
            livereload=self.livereload,
            strict=self.strict,
        )
        return report


__all__: list[str] = ["FlextInfraDocServer"]
