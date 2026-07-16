"""Service case validation model helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from tests.typings import t


class TestsFlextModelsServiceCaseValidationMixin:
    """Service case validation model helpers."""

    class ValidationScenario(m.BaseModel):
        """Single scenario for validation testing."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique scenario name")]
        validator_type: Annotated[
            str,
            m.Field(description="Validator category under test"),
        ]
        input_value: Annotated[
            t.JsonValue | None,
            m.Field(description="Input value passed to validator"),
        ]
        input_params: Annotated[
            t.JsonPayload | None,
            m.Field(
                description="Optional validator parameters for scenario execution",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            m.Field(
                description="Whether scenario expects validation success",
            ),
        ] = True
        expected_value: Annotated[
            t.JsonValue | None,
            m.Field(
                description="Expected normalized value when validation succeeds",
            ),
        ] = None
        expected_error_contains: Annotated[
            str | None,
            m.Field(
                description="Expected error substring when validation fails",
            ),
        ] = None
        description: Annotated[
            str | None,
            m.Field(description="Human-readable scenario description"),
        ] = None

    class Operation(m.BaseModel):
        """Generic operation progress model used by tests utilities."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        success_count: Annotated[int, m.Field(description="Successful operations")]
        failure_count: Annotated[int, m.Field(description="Failed operations")]
        skipped_count: Annotated[int, m.Field(description="Skipped operations")]
        metadata: Annotated[
            t.JsonMapping,
            m.Field(description="Additional operation metadata"),
        ] = m.Field(default_factory=dict)

    class Conversion(m.BaseModel):
        """Generic conversion progress model used by tests utilities."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        converted: Annotated[
            t.JsonList,
            m.Field(description="Converted records"),
        ] = m.Field(default_factory=list)
        errors: Annotated[
            t.StrSequence,
            m.Field(description="Conversion errors"),
        ] = m.Field(default_factory=tuple)
        warnings: Annotated[
            t.StrSequence,
            m.Field(description="Conversion warnings"),
        ] = m.Field(default_factory=tuple)
        skipped: Annotated[
            t.JsonList,
            m.Field(description="Skipped records"),
        ] = m.Field(default_factory=list)
        metadata: Annotated[
            t.JsonMapping,
            m.Field(description="Additional conversion metadata"),
        ] = m.Field(default_factory=dict)

    class ParserScenario(m.BaseModel):
        """Single scenario for parser testing."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[str, m.Field(description="Unique parser scenario name")]
        parser_method: Annotated[str, m.Field(description="Parser method to execute")]
        input_data: Annotated[str, m.Field(description="Raw parser input data")]
        expected_output: Annotated[
            t.JsonValue | None,
            m.Field(
                description="Expected parsed output for successful scenarios",
            ),
        ] = None
        should_succeed: Annotated[
            bool,
            m.Field(
                description="Whether parser scenario expects success",
            ),
        ] = True
        error_contains: Annotated[
            str | None,
            m.Field(description="Expected parser error substring"),
        ] = None
        description: Annotated[
            str | None,
            m.Field(description="Human-readable scenario description"),
        ] = None

    class PublicParseCase(m.BaseModel):
        """Data-driven public parser contract scenario."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        name: Annotated[str, m.Field(description="Unique scenario name")]
        input_value: Annotated[
            t.JsonPayload | None,
            m.Field(description="Public value passed to u.parse()"),
        ]
        target: Annotated[
            type[object],
            m.Field(description="Public target type passed to u.parse()"),
        ]
        options: Annotated[
            m.BaseModel | None,
            m.Field(description="Optional ParseOptions instance"),
        ] = None
        should_succeed: Annotated[
            bool,
            m.Field(description="Whether parsing should succeed"),
        ] = True
        expected_value: Annotated[
            t.JsonPayload | None,
            m.Field(description="Expected parsed scalar or enum value"),
        ] = None
        expected_data: Annotated[
            t.JsonMapping | None,
            m.Field(description="Expected parsed model_dump payload"),
        ] = None
        error_contains: Annotated[
            str | None,
            m.Field(description="Expected failure error substring"),
        ] = None
        description: Annotated[
            str | None,
            m.Field(description="Human-readable scenario description"),
        ] = None


__all__: list[str] = ["TestsFlextModelsServiceCaseValidationMixin"]
