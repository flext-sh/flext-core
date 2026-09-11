"""Pipeline type aliases for DAG engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from flext_cli import c
from flext_core import p, t

if TYPE_CHECKING:
    from flext_cli import m


class FlextCliTypesPipeline:
    """Pipeline type aliases namespace."""

    type PipelineStageStatus = Literal[
        c.Cli.PipelineStageStatus.OK,
        c.Cli.PipelineStageStatus.SKIPPED,
        c.Cli.PipelineStageStatus.FAILED,
    ]
    type PipelineHandler = Callable[
        [m.Cli.PipelineStageContext], p.Result[m.Cli.PipelineStageResult]
    ]
    type PipelineSkipPredicate = Callable[[m.Cli.PipelineStageContext], bool]
    type PipelineHandlerMap = t.MappingKV[str, PipelineHandler]
    type PipelineSkipMap = t.MappingKV[str, PipelineSkipPredicate]


__all__: list[str] = ["FlextCliTypesPipeline"]
