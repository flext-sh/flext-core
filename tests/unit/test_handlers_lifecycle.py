"""Behavioral contract tests for FlextHandlers lifecycle and dispatch.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest
from flext_tests import h, r

from tests.constants import c
from tests.models import m
from tests.typings import t
from tests.utilities import TestsFlextUtilities, u

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextHandlersLifecycle(TestsFlextUtilities.TestsFlextFlextHandlers):
    """Assert the public contract callers depend on, never internals."""

    def test_default_config_exposes_command_mode_and_given_name(self) -> None:
        """Test default config exposes command mode and given name."""
        settings = u.Tests.create_handler_config("h_default", "Default Handler")
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == c.DEFAULT_HANDLER_MODE
        assert handler.handler_name == "Default Handler"

    def test_custom_config_exposes_requested_mode_and_name(self) -> None:
        """Test custom config exposes requested mode and name."""
        settings = u.Tests.create_handler_config(
            "h_query",
            "Query Handler",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == c.HandlerType.QUERY
        assert handler.handler_name == "Query Handler"

    @pytest.mark.parametrize(
        "scenario",
        TestsFlextUtilities.TestsFlextFlextHandlers.HANDLER_TYPES,
        ids=lambda s: s.name,
    )
    def test_mode_reflects_configured_handler_mode(
        self,
        scenario: TestsFlextUtilities.TestsFlextFlextHandlers.HandlerTypeScenario,
    ) -> None:
        """Test mode reflects configured handler mode."""
        settings = u.Tests.create_handler_config(
            f"h_{scenario.name}",
            f"{scenario.name.title()} Handler",
            handler_type=scenario.handler_type,
            handler_mode=scenario.handler_mode,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        assert handler.mode == scenario.handler_mode

    def test_handle_returns_processed_value_on_success(self) -> None:
        """Test handle returns processed value on success."""
        handler = self._concrete("h_ok", "Ok Handler")
        result = handler.handle("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_handle_reports_failure_with_producer_error(self) -> None:
        """Test handle reports failure with producer error."""
        handler = self.FailingTestHandler(
            settings=u.Tests.create_handler_config("h_fail", "Fail Handler"),
        )
        result = handler.handle("payload")
        u.Tests.assert_failure(result, expected_error="Handler failed for: payload")

    def test_handle_rejects_wrong_message_type(self) -> None:
        """Test handle rejects wrong message type."""
        handler = self._concrete("h_type", "Type Handler")
        result = handler.handle(123)
        u.Tests.assert_failure(result, expected_error=c.Tests.UNEXPECTED_MESSAGE_TYPE)

    def test_execute_delivers_handle_result_on_valid_message(self) -> None:
        """Test execute delivers handle result on valid message."""
        handler = self._concrete("h_exec", "Exec Handler")
        result = handler.execute("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_execute_fails_when_validation_rejects_message(self) -> None:
        """Test execute fails when validation rejects message."""
        handler = self._concrete("h_exec_none", "Exec None Handler")
        result = handler.execute(None)
        assert result.failure
        assert result.error is not None
        assert c.ERR_MESSAGE_CANNOT_BE_NONE in result.error

    def test_validate_message_accepts_non_null_payload(self) -> None:
        """Test validate message accepts non null payload."""
        handler = self._concrete("h_val_ok", "Validate Ok Handler")
        assert handler.validate_message("payload").unwrap() is True

    def test_validate_message_rejects_none(self) -> None:
        """Test validate message rejects none."""
        handler = self._concrete("h_val_none", "Validate None Handler")
        result = handler.validate_message(None)
        u.Tests.assert_failure(result, expected_error=c.ERR_MESSAGE_CANNOT_BE_NONE)

    def test_can_handle_accepts_supported_message_type(self) -> None:
        """Test can handle accepts supported message type."""
        handler = self._concrete("h_can", "Can Handle Handler")
        assert handler.can_handle(str) is True

    def test_dispatch_runs_full_pipeline_on_success(self) -> None:
        """Test dispatch runs full pipeline on success."""
        handler = self._concrete("h_disp", "Dispatch Handler")
        result = handler.dispatch_message("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_dispatch_rejects_pipeline_mode_incompatible_with_handler_mode(
        self,
    ) -> None:
        """Test dispatch rejects pipeline mode incompatible with handler mode."""
        handler = self._concrete("h_disp_mode", "Dispatch Mode Handler")
        result = handler.dispatch_message("payload", c.HandlerMode.QUERY)
        assert result.failure
        assert result.error is not None
        assert "query" in result.error

    @pytest.mark.parametrize(
        ("label", "message"),
        TestsFlextUtilities.TestsFlextFlextHandlers.VALIDATION_TYPES,
        ids=[
            label
            for label, _ in TestsFlextUtilities.TestsFlextFlextHandlers.VALIDATION_TYPES
        ],
    )
    def test_validation_accepts_every_non_null_payload_shape(
        self,
        label: str,
        message: t.JsonPayload,
    ) -> None:
        """Test validation accepts every non null payload shape."""
        handler = self._concrete(f"h_val_{label}", f"Validate {label} Handler")
        assert handler.validate_message(message).unwrap() is True

    def test_metadata_is_preserved_and_does_not_block_execution(self) -> None:
        """Test metadata is preserved and does not block execution."""
        settings = u.Tests.create_handler_config(
            "h_meta",
            "Metadata Handler",
            metadata=m.Metadata(attributes={"priority": 1}),
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_command_timeout_config_does_not_break_execution(self) -> None:
        """Test command timeout config does not break execution."""
        settings = u.Tests.create_handler_config(
            "h_timeout",
            "Timeout Handler",
            command_timeout=60,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_retry_config_does_not_break_execution(self) -> None:
        """Test retry config does not break execution."""
        settings = u.Tests.create_handler_config(
            "h_retry",
            "Retry Handler",
            max_command_retries=3,
        )
        handler = self.ConcreteTestHandler(settings=settings)
        result = handler.execute("payload")
        u.Tests.assert_success(result, expected_value="processed_payload")

    def test_subclass_specialises_handle_for_its_own_message_type(self) -> None:
        """Test subclass specialises handle for its own message type."""

        class IntHandler(h[t.JsonPayload, t.JsonPayload]):
            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                if not isinstance(message, int):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].ok(f"processed_{message}")

        handler = IntHandler(
            settings=u.Tests.create_handler_config("h_int", "Int Handler"),
        )
        u.Tests.assert_success(handler.handle(42), expected_value="processed_42")
        u.Tests.assert_failure(
            handler.handle("nan"),
            expected_error=c.Tests.UNEXPECTED_MESSAGE_TYPE,
        )

    def _concrete(
        self,
        handler_id: str,
        handler_name: str,
    ) -> TestsFlextUtilities.TestsFlextFlextHandlers.ConcreteTestHandler:
        settings = u.Tests.create_handler_config(handler_id, handler_name)
        return self.ConcreteTestHandler(settings=settings)
