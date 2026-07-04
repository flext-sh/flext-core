"""Handler validation and context tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

import pytest

from tests.models import m
from tests.typings import t
from tests.unit._handlers_support import TestsFlextFlextHandlers
from tests.utilities import u

if TYPE_CHECKING:
    from tests.constants import c

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
HandlerTypeScenario = TestsFlextFlextHandlers.HandlerTypeScenario
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextHandlersValidationContext(TestsFlextFlextHandlers):
    @pytest.mark.parametrize(
        ("handler_type", "handler_mode"),
        [(scenario.handler_type, scenario.handler_mode) for scenario in HANDLER_TYPES],
        ids=[scenario.name for scenario in HANDLER_TYPES],
    )
    def test_handlers_validate_generic(
        self,
        handler_type: c.HandlerType,
        handler_mode: c.HandlerType,
    ) -> None:
        handler_type_name = str(handler_type)
        settings = u.Tests.create_handler_config(
            f"test_validate_generic_{handler_type_name}",
            f"Test Validate Generic {handler_type_name.title()}",
            handler_type=handler_type,
            handler_mode=handler_mode,
        )
        handler: TestsFlextFlextHandlers.ConcreteTestHandler = self.ConcreteTestHandler(
            settings=settings,
        )
        result = handler.validate_message("test_message")
        _ = u.Tests.assert_success(result)

    @pytest.mark.parametrize(
        ("type_name", "message"),
        VALIDATION_TYPES,
        ids=[item[0] for item in VALIDATION_TYPES],
    )
    def test_handlers_message_validation_types(
        self,
        type_name: str,
        message: t.JsonValue,
    ) -> None:
        settings = u.Tests.create_handler_config(
            f"test_{type_name}_message",
            f"Test {type_name.title()} Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        result = handler.validate_message(message)
        _ = u.Tests.assert_success(result)

    def test_handlers_record_metric(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_record_metric",
            "Test Record Metric",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.record_metric("test_metric", 42.0)
        _ = u.Tests.assert_success(result)

    def test_handlers_push_context(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_push_context",
            "Test Push Context",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        context_typed = t.json_mapping_adapter().validate_python({
            "user_id": "123",
            "operation": "test",
        })
        result = handler.push_context(context_typed)
        _ = u.Tests.assert_success(result)

    def test_handlers_pop_context(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_pop_context",
            "Test Pop Context",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        handler.push_context({"test": "data"})
        result = handler.pop_context()
        _ = u.Tests.assert_success(result)

    def test_handlers_pop_context_empty_stack(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_pop_context_empty",
            "Test Pop Context Empty",
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.pop_context()
        _ = u.Tests.assert_success(result)

    def test_handlers_message_with_none_raises_validation_error(self) -> None:
        settings = u.Tests.create_handler_config(
            "test_none_message",
            "Test None Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        result = handler.validate_message("")
        _ = u.Tests.assert_failure(result)

    def test_handlers_pydantic_model_validation(self) -> None:
        class TestMessage(m.Value):
            value: str

        settings = u.Tests.create_handler_config(
            "test_pydantic_validation",
            "Test Pydantic Validation",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = TestMessage(value="test")
        result = handler.validate_message(msg)
        _ = u.Tests.assert_success(result)

    def test_handlers_dataclass_message_validation(self) -> None:
        class DataClassMessage(m.Value):
            value: Annotated[str, m.Field(description="Message value")]
            number: Annotated[int, m.Field(description="Message number")]

        settings = u.Tests.create_handler_config(
            "test_dataclass_message",
            "Test Dataclass Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = DataClassMessage(value="test", number=42)
        result = handler.validate_message(msg)
        _ = u.Tests.assert_success(result)

    def test_handlers_slots_message_validation(self) -> None:
        class SlotsMessage(m.Value):
            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
            value: str
            number: int

        settings = u.Tests.create_handler_config(
            "test_slots_message",
            "Test Slots Message",
        )
        handler = self.ValidationTestHandler(settings=settings)
        msg = SlotsMessage(value="test", number=42)
        result = handler.validate_message(msg)
        _ = u.Tests.assert_success(result)
