"""Pipeline protocol contracts for DAG-based stage execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_cli import t
from flext_core import p

if TYPE_CHECKING:
    # mro-j47u (codex): p -> m is a reverse facade edge; keep it type-only.
    from flext_cli import m


class FlextCliProtocolsPipeline:
    """Pipeline protocol namespace."""

    @runtime_checkable
    class PipelineStageContext(Protocol):
        """Contract for stage execution context — carries shared state between stages."""

        @property
        def repository_root(self) -> Path:
            """Repository root directory."""
            ...

        @property
        def shared(self) -> t.MutableJsonMapping:
            """Mutable shared state between stages — stages write outputs here."""
            ...

        @property
        def settings(self) -> t.JsonMapping:
            """Immutable configuration for the pipeline run."""
            ...

    @runtime_checkable
    class PipelineStage(Protocol):
        """Contract for a callable pipeline stage handler."""

        def __call__(
            self, ctx: FlextCliProtocolsPipeline.PipelineStageContext
        ) -> p.Result[m.Cli.PipelineStageResult]:
            """Execute stage and return typed result."""
            ...

    @runtime_checkable
    class PipelineService(Protocol):
        """Contract for the public pipeline DSL exposed on ``cli``."""

        def stage_context(
            self,
            repository_root: Path,
            *,
            shared: t.MutableJsonMapping | None = None,
            settings: t.JsonMapping | None = None,
        ) -> m.Cli.PipelineStageContext:
            """Build the canonical pipeline execution context."""
            ...

        def stage(
            self,
            stage_id: str,
            *,
            handler: Callable[
                [FlextCliProtocolsPipeline.PipelineStageContext],
                p.Result[m.Cli.PipelineStageResult],
            ],
            depends_on: t.SequenceOf[str] | frozenset[str] = (),
            skip_if: Callable[[FlextCliProtocolsPipeline.PipelineStageContext], bool]
            | None = None,
        ) -> m.Cli.PipelineStageSpec:
            """Build one declarative pipeline stage spec."""
            ...

        def stage_result(
            self,
            stage_id: str,
            *,
            status: t.Cli.PipelineStageStatus,
            output: t.JsonMapping | None = None,
            duration_ms: float = 0.0,
            error: str | None = None,
        ) -> m.Cli.PipelineStageResult:
            """Build one typed pipeline stage result payload."""
            ...

        def ok_stage(
            self,
            stage_id: str,
            *,
            output: t.JsonMapping | None = None,
            duration_ms: float = 0.0,
        ) -> p.Result[m.Cli.PipelineStageResult]:
            """Return a successful typed stage result via ``r``."""
            ...

        def pipeline(
            self,
            stages: t.SequenceOf[m.Cli.PipelineStageSpec],
            *,
            context: m.Cli.PipelineStageContext,
            logger: p.Logger | None = None,
        ) -> p.Result[m.Cli.PipelineResult]:
            """Execute a pipeline from the public service DSL."""
            ...

        def linear_pipeline(
            self,
            stage_order: t.StrSequence,
            handlers: t.Cli.PipelineHandlerMap,
            *,
            skip_by_stage: t.Cli.PipelineSkipMap | None = None,
        ) -> t.SequenceOf[m.Cli.PipelineStageSpec]:
            """Build a linear dependency chain with canonical previous-stage deps."""
            ...


__all__: list[str] = ["FlextCliProtocolsPipeline"]
