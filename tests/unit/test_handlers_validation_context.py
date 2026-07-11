"""Behavioral tests for handler validation and execution-context contract.

Every test asserts observable public behavior of the ``FlextHandlers`` surface
(``validate_message`` / ``execute`` / ``dispatch_message`` / ``record_metric`` /
``push_context`` / ``pop_context`` / ``can_handle`` and the ``handler_name`` /
``mode`` properties) through its ``r[T]`` results and public model state. No
private attributes, internal collaborators, or implementation structures are
inspected.
"""

from __future__ import annotations

from typing import Annotated, ClassVar

import pytest

from tests.constants import c
from tests.models import m
from tests.typings import t
from tests.unit._handlers_support import TestsFlextFlextHandlers
from tests.utilities import u

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextCoreHandlersValidationContext(TestsFlextFlextHandlers):
    """Contract tests for the public handler validation/context behavior."""

    @pytest.mark.parametrize(
        ("handler_type", "handler_mode"),
        [(scenario.handler_type, scenario.handler_mode) for scenario in HANDLER_TYPES],
        ids=[scenario.name for scenario in HANDLER_TYPES],
    )
    def test_validate_message_accepts_message_for_every_handler_type(
        self,
        handler_type: c.HandlerType,
        handler_mode: c.HandlerType,
    ) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            f"validate_generic_{handler_type}",
            f"Validate Generic {handler_type.title()}",
            handler_type=handler_type,
            handler_mode=handler_mode,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.validate_message("test_message")

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    @pytest.mark.parametrize(
        ("type_name", "message"),
        VALIDATION_TYPES,
        ids=[item[0] for item in VALIDATION_TYPES],
    )
    def test_validate_message_accepts_supported_payload_types(
        self,
        type_name: str,
        message: t.JsonValue,
    ) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            f"payload_{type_name}",
            f"Payload {type_name.title()}",
        )
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(message)

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_validate_message_rejects_none_with_specific_error(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("reject_none", "Reject None")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.validate_message(None)

        # Assert
        error = u.Tests.assert_failure(result, c.ERR_MESSAGE_CANNOT_BE_NONE)
        assert c.ERR_MESSAGE_CANNOT_BE_NONE in error

    @pytest.mark.parametrize(
        "falsy_message",
        ["", None],
        ids=["empty_string", "none"],
    )
    def test_validation_handler_rejects_falsy_message(
        self,
        falsy_message: t.JsonValue,
    ) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("reject_falsy", "Reject Falsy")
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(falsy_message)

        # Assert
        _ = u.Tests.assert_failure(result, c.Tests.VALIDATION_FAILED_FOR_TEST)

    def test_validate_message_accepts_pydantic_model_message(self) -> None:
        # Arrange
        class PydanticMessage(m.Value):
            value: str

        settings = u.Tests.create_handler_config("pydantic_msg", "Pydantic Message")
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(PydanticMessage(value="test"))

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_validate_message_accepts_multi_field_model_message(self) -> None:
        # Arrange
        class MultiFieldMessage(m.Value):
            value: Annotated[str, m.Field(description="Message value")]
            number: Annotated[int, m.Field(description="Message number")]

        settings = u.Tests.create_handler_config("multi_field", "Multi Field")
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(MultiFieldMessage(value="test", number=42))

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_validate_message_accepts_frozen_model_message(self) -> None:
        # Arrange
        class FrozenMessage(m.Value):
            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
            value: str
            number: int

        settings = u.Tests.create_handler_config("frozen_msg", "Frozen Message")
        handler = self.ValidationTestHandler(settings=settings)

        # Act
        result = handler.validate_message(FrozenMessage(value="test", number=42))

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_record_metric_succeeds(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("record_metric", "Record Metric")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.record_metric("latency_ms", 42.0)

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_push_context_succeeds(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("push_context", "Push Context")
        handler = self.ConcreteTestHandler(settings=settings)
        context: t.JsonMapping = t.json_mapping_adapter().validate_python({
            "user_id": "123",
            "operation": "test",
        })

        # Act
        result = handler.push_context(context)

        # Assert
        _ = u.Tests.assert_success(result, expected_value=True)

    def test_pop_context_after_push_returns_config_map(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("pop_context", "Pop Context")
        handler = self.ConcreteTestHandler(settings=settings)
        _ = u.Tests.assert_success(handler.push_context({"stage": "unit"}))

        # Act
        result = handler.pop_context()

        # Assert
        popped = u.Tests.assert_success(result)
        assert isinstance(popped, m.ConfigMap)

    def test_pop_context_on_empty_stack_returns_empty_config_map(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("pop_empty", "Pop Empty")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.pop_context()

        # Assert
        popped = u.Tests.assert_success(result)
        assert isinstance(popped, m.ConfigMap)
        assert popped.model_dump() == {}

    def test_can_handle_accepts_arbitrary_type_for_flexible_handler(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("can_handle", "Can Handle")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act & Assert
        assert handler.can_handle(str) is True
        assert handler.can_handle(int) is True

    @pytest.mark.parametrize(
        "handler_type",
        [scenario.handler_type for scenario in HANDLER_TYPES],
        ids=[scenario.name for scenario in HANDLER_TYPES],
    )
    def test_handler_properties_reflect_configuration(
        self,
        handler_type: c.HandlerType,
    ) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            f"props_{handler_type}",
            f"Props {handler_type.title()}",
            handler_type=handler_type,
            handler_mode=handler_type,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act & Assert
        assert handler.handler_name == f"Props {handler_type.title()}"
        assert handler.mode.value == handler_type

    def test_execute_validates_then_produces_processed_result(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("execute", "Execute")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.execute("payload")

        # Assert
        _ = u.Tests.assert_success(result, expected_value="processed_payload")

    def test_dispatch_message_runs_full_pipeline_for_matching_mode(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config("dispatch", "Dispatch")
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.dispatch_message("payload")

        # Assert
        _ = u.Tests.assert_success(result, expected_value="processed_payload")

    def test_dispatch_message_rejects_incompatible_pipeline_mode(self) -> None:
        # Arrange
        settings = u.Tests.create_handler_config(
            "dispatch_query",
            "Dispatch Query",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        )
        handler = self.ConcreteTestHandler(settings=settings)

        # Act
        result = handler.dispatch_message("payload", operation=c.DEFAULT_HANDLER_MODE)

        # Assert
        _ = u.Tests.assert_failure(result)
