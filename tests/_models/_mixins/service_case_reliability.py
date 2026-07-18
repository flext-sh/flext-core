"""Service case reliability model helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m, t


class TestsFlextModelsServiceCaseReliabilityMixin:
    """Service case reliability model helpers."""

    class ReliabilityScenario(m.BaseModel):
        """Single scenario for reliability testing (circuit breaker, retry)."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique reliability scenario name")]
        strategy: Annotated[str, m.Field(description="Reliability strategy under test")]
        settings: Annotated[
            m.ConfigMap, m.Field(description="Reliability configuration payload")
        ]
        simulate_failures: Annotated[
            int, m.Field(description="Number of failures to simulate")
        ]
        expected_state: Annotated[
            str, m.Field(description="Expected strategy terminal state")
        ]
        should_succeed: Annotated[
            bool, m.Field(description="Whether scenario expects successful outcome")
        ] = True
        description: Annotated[
            str | None, m.Field(description="Human-readable scenario description")
        ] = None


__all__: list[str] = ["TestsFlextModelsServiceCaseReliabilityMixin"]
