"""Typed renderer for canonical focused cProfile artifacts."""

from __future__ import annotations

import io
import pstats
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, override

from flext_core import r
from flext_infra import m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraCProfileReport(s[bool]):
    """Render one bounded human-readable report from a pstats artifact."""

    profile: Annotated[Path, m.Field(description="Input cProfile pstats path")]
    output: Annotated[Path, m.Field(description="Output text report path")]
    sort: Annotated[
        Literal[
            "calls",
            "cumulative",
            "filename",
            "line",
            "name",
            "nfl",
            "pcalls",
            "stdname",
            "time",
        ],
        m.Field(description="Validated pstats sort key"),
    ]
    limit: Annotated[int, m.Field(gt=0, le=1000, description="Maximum rows to print")]

    @u.model_validator(mode="after")
    def _validate_report_paths(self) -> Self:
        """Keep profile input and output inside the workspace report tree."""
        report_root = (self.repository_root / ".reports").resolve()
        for path in (self.profile, self.output):
            try:
                path.resolve().relative_to(report_root)
            except ValueError as exc:
                msg = f"cProfile path must stay under {report_root}: {path}"
                raise ValueError(msg) from exc
        return self

    @override
    def execute(self) -> p.Result[bool]:
        """Load, sort, and render the profile without executing user code."""
        if not self.profile.is_file():
            return r[bool].fail(f"cProfile artifact does not exist: {self.profile}")
        try:
            stream = io.StringIO()
            stats = pstats.Stats(str(self.profile), stream=stream)
            stats.strip_dirs().sort_stats(self.sort).print_stats(self.limit)
            self.output.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError, TypeError) as exc:
            return r[bool].fail_op("render cProfile report", exc)
        written = u.Cli.atomic_write_text_file(self.output, stream.getvalue())
        if written.failure:
            return r[bool].fail(
                written.error or f"failed to write cProfile report: {self.output}"
            )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraCProfileReport"]
