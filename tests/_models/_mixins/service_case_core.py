"""Service case core model helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar, override

from flext_core import m
from flext_tests import r
from tests import p, t
from tests.base import s


class TestsFlextModelsServiceCaseCoreMixin:
    """Service case core model helpers."""

    class ServiceUserData(m.Value):
        """Public result model used by service tests."""

        user_id: Annotated[int, m.Field(description="User identifier")]
        name: Annotated[str, m.Field(description="User name")]

    class ServiceUserService(s[bool]):
        """Simple successful service for test scenarios."""

        @override
        def execute(
            self,
        ) -> p.Result[TestsFlextModelsServiceCaseCoreMixin.ServiceUserData]:
            return r[TestsFlextModelsServiceCaseCoreMixin.ServiceUserData].ok(
                TestsFlextModelsServiceCaseCoreMixin.ServiceUserData(
                    user_id=1, name="test_user"
                )
            )

    class ServiceTestCase(m.BaseModel):
        """Test case for service."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        service_type: Annotated[
            str | None, m.Field(description="Service type for factory-driven tests")
        ] = None
        input_value: Annotated[
            str | None, m.Field(description="Primary service input")
        ] = None
        user_id: Annotated[
            str | None, m.Field(description="User identifier for documented tests")
        ] = None
        expected_success: Annotated[
            bool, m.Field(description="Whether service call is expected to succeed")
        ] = True
        expected_error: Annotated[
            str | None,
            m.Field(description="Expected error substring for failure cases"),
        ] = None
        description: Annotated[
            str, m.Field(description="Human-readable test case description")
        ] = ""
        extra_param: Annotated[
            int, m.Field(description="Auxiliary numeric parameter")
        ] = 3

    class RailwayTestCase(m.BaseModel):
        """Test case for railway pattern."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        user_ids: Annotated[
            t.StrSequence, m.Field(description="User identifiers used in pipeline")
        ]
        operations: Annotated[
            t.StrSequence, m.Field(description="Pipeline operations to execute")
        ] = m.Field(default_factory=tuple)
        expected_pipeline_length: Annotated[
            int, m.Field(description="Expected number of pipeline stages")
        ] = 1
        should_fail_at: Annotated[
            int | None, m.Field(description="Optional pipeline step expected to fail")
        ] = None
        description: Annotated[
            str, m.Field(description="Human-readable railway test case description")
        ] = ""


__all__: list[str] = ["TestsFlextModelsServiceCaseCoreMixin"]
