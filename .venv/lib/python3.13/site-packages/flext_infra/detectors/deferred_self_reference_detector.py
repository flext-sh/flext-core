"""Detect deferred self-reference and recursive models via Rope-backed AST scanning."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from flext_infra import m, u
from flext_infra._utilities.deferred_self_reference_ast import (
    collect_deferred_self_reference_findings,
)

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraDeferredSelfReferenceDetector:
    """Detect models that defer their own resolution instead of composing by FLEXT."""

    @staticmethod
    def detect_file(ctx: m.Infra.DetectorContext) -> t.SequenceOf[m.Infra.Issue]:
        """Detect deferred-self-reference findings in one Python file."""
        resource = u.Infra.fetch_python_resource(ctx.rope_project, ctx.file_path)
        if resource is None:
            return []
        source = resource.read()
        if not source.strip():
            return []
        file_path = ctx.file_path
        display_path = file_path
        if ctx.project_root is not None and file_path.is_relative_to(ctx.project_root):
            display_path = file_path.relative_to(ctx.project_root)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        return tuple(
            m.Infra.Issue(
                file=str(display_path),
                line=finding.line,
                column=finding.column,
                code=finding.kind,
                message=finding.detail,
            )
            for finding in collect_deferred_self_reference_findings(tree)
        )


__all__: list[str] = ["FlextInfraDeferredSelfReferenceDetector"]
