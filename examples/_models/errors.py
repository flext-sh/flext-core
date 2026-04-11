"""Centralized error messages for public examples."""

from __future__ import annotations

from enum import StrEnum


class ExamplesFlextCoreModelsErrors:
    """Error-message namespace used by example scripts."""

    class Examples:
        """Examples error-message namespace."""

        class ErrorMessages(StrEnum):
            """Canonical error messages for example runtime flow."""

            EXERCISE_NOT_IMPLEMENTED = "Subclasses must implement exercise()"
            EXPECTED_TEXT_INPUT = "Expected text input"
            TEXT_INPUT_CANNOT_BE_EMPTY = "Text input cannot be empty"
            DB_URL_MUST_BE_TEXT = "Database URL must be text"
            DB_URL_CANNOT_BE_EMPTY = "Database URL cannot be empty"
            BOOM = "boom"
            ADAPTER_BOOM = "adapter boom"
            INVALID = "invalid"
            BAD_CFG = "bad cfg"
            DOWN = "down"
            LATE = "late"
            AUTH_FAIL = "auth fail"
            NOPE = "nope"
            MISSING = "missing"
            CONFLICT = "conflict"
            SLOW_DOWN = "slow down"
            OPEN = "open"
            WRONG_TYPE = "wrong type"
            FAILED_OP = "failed op"
            BAD_ATTR = "bad attr"
            FORCED_BOOM = "forced boom"

        class TriggerTokens(StrEnum):
            """Canonical string tokens used to trigger demo/test behaviour."""

            EXPLODE = "explode"
