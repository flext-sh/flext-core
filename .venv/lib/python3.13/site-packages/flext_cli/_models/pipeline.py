"""Pipeline Pydantic domain models for DAG execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import c, p, t
from flext_cli._models._defaults import EMPTY_JSON_MAPPING
from flext_core import m, u


class FlextCliModelsPipeline:
    """Pipeline models namespace — flat in m.Cli.*."""

    class PipelineStageContext(m.ContractModel):
        """Accumulated state passed between pipeline stages."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
        )

        repository_root: Annotated[
            Path, m.Field(description="Repository root directory")
        ]

        shared: Annotated[
            t.MutableJsonMapping,
            m.Field(
                default_factory=dict, description="Mutable shared state between stages"
            ),
        ]
        settings: Annotated[
            t.JsonMapping,
            m.Field(
                default_factory=lambda: EMPTY_JSON_MAPPING,
                description="Immutable pipeline configuration",
            ),
        ]

    class PipelineStageSpec(m.ContractModel):
        """Declarative stage definition with dependency tracking."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", arbitrary_types_allowed=True
        )

        stage_id: Annotated[str, m.Field(description="Unique stage identifier")]
        depends_on: Annotated[
            frozenset[str],
            m.Field(
                default_factory=frozenset, description="Stage IDs this stage depends on"
            ),
        ]
        # NOTE: handler/skip_if use inline Callable, not t.Cli.PipelineHandler /
        # t.Cli.PipelineSkipPredicate.  Those are PEP 695 `type` aliases that
        # reference p.Cli.PipelineStageContext under TYPE_CHECKING — Pydantic
        # cannot resolve them at runtime for model field validation.
        handler: Annotated[
            Callable[
                [FlextCliModelsPipeline.PipelineStageContext],
                p.Result[FlextCliModelsPipeline.PipelineStageResult],
            ],
            m.Field(description="Callable that executes the stage"),
        ]
        skip_if: Annotated[
            Callable[[FlextCliModelsPipeline.PipelineStageContext], bool] | None,
            m.Field(description="Predicate — skip stage if returns True"),
        ] = None
        retry: Annotated[
            int,
            m.Field(
                ge=0,
                le=c.Cli.PIPELINE_MAX_RETRY,
                description="Number of retries on failure",
            ),
        ] = c.Cli.PIPELINE_DEFAULT_RETRY

    class PipelineStageResult(m.ContractModel):
        """What a stage produces after execution."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")

        stage_id: Annotated[str, m.Field(description="Stage that produced this result")]
        status: Annotated[
            t.Cli.PipelineStageStatus, m.Field(description="Execution outcome")
        ]
        output: Annotated[
            t.JsonMapping,
            m.Field(
                default_factory=lambda: EMPTY_JSON_MAPPING,
                description="Stage output payload",
            ),
        ]
        duration_ms: Annotated[
            float, m.Field(description="Execution duration in milliseconds")
        ] = 0.0
        error: Annotated[str | None, m.Field(description="Error message if failed")] = (
            None
        )

    class PipelineResult(m.ContractModel):
        """Full pipeline execution result — aggregated from all stages."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")

        stages: Annotated[
            t.SequenceOf[FlextCliModelsPipeline.PipelineStageResult],
            m.Field(
                default_factory=tuple, description="Results from all executed stages"
            ),
        ]
        total_duration_ms: Annotated[
            float, m.Field(description="Total pipeline execution time")
        ] = 0.0

        @u.computed_field
        @property
        def success(self) -> bool:
            """True if no stage failed."""
            return all(
                s.status != c.Cli.PipelineStageStatus.FAILED for s in self.stages
            )

        @u.computed_field
        @property
        def failed_stages(
            self,
        ) -> t.SequenceOf[FlextCliModelsPipeline.PipelineStageResult]:
            """Only the failed stage results."""
            return [
                s for s in self.stages if s.status == c.Cli.PipelineStageStatus.FAILED
            ]

        @u.computed_field
        @property
        def skipped_stages(
            self,
        ) -> t.SequenceOf[FlextCliModelsPipeline.PipelineStageResult]:
            """Only the skipped stage results."""
            return [
                s for s in self.stages if s.status == c.Cli.PipelineStageStatus.SKIPPED
            ]


__all__: t.MutableSequenceOf[str] = ["FlextCliModelsPipeline"]
