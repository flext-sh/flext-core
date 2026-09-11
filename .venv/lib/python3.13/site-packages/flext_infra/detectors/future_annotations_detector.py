"""Detect missing 'from __future__ import annotations' via Rope semantics.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraFutureAnnotationsDetector:
    """Detect missing future annotations import via rope resource read."""

    @staticmethod
    def detect_file(
        ctx: m.Infra.DetectorContext,
    ) -> t.SequenceOf[m.Infra.FutureAnnotationsViolation]:
        """Detect missing future annotations in a single file."""
        resource = u.Infra.fetch_python_resource(
            ctx.rope_project, ctx.file_path, skip_protected=True
        )
        if resource is None:
            return []
        file_path = ctx.file_path
        source = resource.read()
        if not source.strip():
            return []
        try:
            pymodule = u.Infra.get_pymodule(ctx.rope_project, resource)
            is_docstring_only = bool(pymodule.get_doc()) and not bool(
                pymodule.get_attributes()
            )
        except (
            *u.Infra.rope_runtime_errors(),
            *u.Infra.rope_syntax_errors(),
            TypeError,
            ValueError,
        ) as exc:
            FlextInfraFutureAnnotationsDetector._record_parse_failure(ctx, exc)
            return []
        if is_docstring_only:
            return []
        if c.Infra.FUTURE_ANNOTATIONS_RE.search(source):
            return []
        return [m.Infra.FutureAnnotationsViolation(file=str(file_path))]

    @staticmethod
    def _record_parse_failure(ctx: m.Infra.DetectorContext, exc: BaseException) -> None:
        """Record a Rope parse failure, or fail loud without a collector."""
        detail = str(exc)
        if ctx.parse_failures is None:
            msg = (
                f"future-annotations detector could not parse {ctx.file_path}: {detail}"
            )
            raise RuntimeError(msg) from exc
        ctx.parse_failures.append(
            m.Infra.ParseFailureViolation(
                file=str(ctx.file_path),
                stage="future-annotations",
                error_type=type(exc).__name__,
                detail=detail,
            )
        )


__all__: list[str] = ["FlextInfraFutureAnnotationsDetector"]
