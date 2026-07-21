"""Shared r scenario fixtures for split result tests."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Annotated, ClassVar

from tests import m, t

if TYPE_CHECKING:
    from collections.abc import Sequence


@unique
class ResultOperationType(StrEnum):
    """Result operation test scenario types."""

    CREATION_SUCCESS = "creation_success"
    CREATION_FAILURE = "creation_failure"
    UNWRAP = "unwrap"
    UNWRAP_OR = "unwrap_or"
    MAP = "map"
    FLAT_MAP = "flat_map"
    FILTER = "filter"
    ALT = "alt"
    LASH = "lash"
    OR_OPERATOR = "or_operator"
    BOOL_CONVERSION = "bool_conversion"
    RAILWAY_COMPOSITION = "railway_composition"


class ResultScenario(m.BaseModel):
    """Generic result scenario for r tests."""

    model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)
    name: Annotated[str, m.Field(description="Result scenario name")]
    operation_type: Annotated[
        ResultOperationType, m.Field(description="Result operation type")
    ]
    value: Annotated[
        t.JsonValue, m.Field(description="Input value for result operation")
    ]
    is_success_expected: Annotated[
        bool, m.Field(description="Expected success state")
    ] = True
    expected_result: Annotated[
        t.JsonValue | None, m.Field(description="Optional expected result payload")
    ] = None


STRING_SCENARIOS: Sequence[ResultScenario] = [
    ResultScenario(
        name="creation_success_string",
        operation_type=ResultOperationType.CREATION_SUCCESS,
        value="success",
    ),
    ResultScenario(
        name="creation_failure_message",
        operation_type=ResultOperationType.CREATION_FAILURE,
        value="error message",
        is_success_expected=False,
    ),
    ResultScenario(
        name="unwrap_or_success",
        operation_type=ResultOperationType.UNWRAP_OR,
        value="value",
    ),
    ResultScenario(
        name="unwrap_or_failure",
        operation_type=ResultOperationType.UNWRAP_OR,
        value="error",
        is_success_expected=False,
    ),
    ResultScenario(
        name="map_failure",
        operation_type=ResultOperationType.MAP,
        value="error",
        is_success_expected=False,
    ),
    ResultScenario(
        name="flat_map_failure",
        operation_type=ResultOperationType.FLAT_MAP,
        value="error",
        is_success_expected=False,
    ),
    ResultScenario(
        name="alt_success", operation_type=ResultOperationType.ALT, value="success"
    ),
    ResultScenario(
        name="alt_failure",
        operation_type=ResultOperationType.ALT,
        value="original_error",
        is_success_expected=False,
    ),
    ResultScenario(
        name="lash_success", operation_type=ResultOperationType.LASH, value="success"
    ),
    ResultScenario(
        name="lash_failure",
        operation_type=ResultOperationType.LASH,
        value="error",
        is_success_expected=False,
    ),
    ResultScenario(
        name="or_operator_success",
        operation_type=ResultOperationType.OR_OPERATOR,
        value="value",
    ),
    ResultScenario(
        name="or_operator_failure",
        operation_type=ResultOperationType.OR_OPERATOR,
        value="error",
        is_success_expected=False,
    ),
]
INT_SCENARIOS: Sequence[ResultScenario] = [
    ResultScenario(
        name="unwrap_success", operation_type=ResultOperationType.UNWRAP, value=42
    ),
    ResultScenario(name="map_success", operation_type=ResultOperationType.MAP, value=5),
    ResultScenario(
        name="flat_map_success", operation_type=ResultOperationType.FLAT_MAP, value=5
    ),
    ResultScenario(
        name="filter_passes", operation_type=ResultOperationType.FILTER, value=10
    ),
    ResultScenario(
        name="filter_fails",
        operation_type=ResultOperationType.FILTER,
        value=3,
        is_success_expected=False,
    ),
    ResultScenario(
        name="railway_composition",
        operation_type=ResultOperationType.RAILWAY_COMPOSITION,
        value=5,
    ),
]
BOOL_SCENARIOS: Sequence[ResultScenario] = [
    ResultScenario(
        name="bool_conversion_success",
        operation_type=ResultOperationType.BOOL_CONVERSION,
        value=True,
    ),
    ResultScenario(
        name="bool_conversion_failure",
        operation_type=ResultOperationType.BOOL_CONVERSION,
        value=False,
    ),
]
