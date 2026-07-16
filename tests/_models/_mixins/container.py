"""Container scenario model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from flext_core import m
from tests.typings import p, t

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextModelsContainerMixin:
    """Container scenario model helpers."""

    class ServiceScenario(m.BaseModel):
        """Test scenario for service registration and retrieval."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: Annotated[str, m.Field(description="Service scenario name")]
        service: Annotated[
            t.Primitives,
            m.Field(description="Service value to register"),
        ]
        description: Annotated[str, m.Field(description="Scenario description")] = ""

    class TypedRetrievalScenario(m.BaseModel):
        """Test scenario for typed service retrieval."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )
        name: Annotated[str, m.Field(description="Typed retrieval scenario name")]
        service: Annotated[
            str | int,
            m.Field(description="Registered service value"),
        ]
        expected_type: Annotated[
            type[str | int],
            m.Field(description="Expected service type"),
        ]
        should_pass: Annotated[
            bool,
            m.Field(description="Whether typed retrieval should succeed"),
        ]
        description: Annotated[str, m.Field(description="Scenario description")] = ""

    class ContainerScenarios:
        """Centralized container test scenarios using c."""

        SERVICE_SCENARIOS: ClassVar[
            t.SequenceOf[TestsFlextModelsContainerMixin.ServiceScenario]
        ] = []  # populated after class definition
        TYPED_RETRIEVAL_SCENARIOS: ClassVar[
            t.SequenceOf[TestsFlextModelsContainerMixin.TypedRetrievalScenario]
        ] = []  # populated after class definition
        CONFIG_SCENARIOS: ClassVar[Sequence[t.ScalarMapping]] = [
            {"enable_singleton": False, "max_services": 8},
            {"invalid_key": "value", "another_invalid": 42},
            {},
        ]

    # --- from test_utilities_guards.py and test_utilities_guards_full_coverage.py ---


__all__: list[str] = ["TestsFlextModelsContainerMixin"]
