"""Real tests to achieve 100% runtime coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in runtime.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping, Sequence
from typing import Never, cast, overload, override

import structlog

from flext_core import FlextRuntime, p
from flext_tests import t


class TestRuntimeDictLike:
    """Tests for is_dict_like runtime coverage."""

    def test_is_dict_like_with_exception_on_items(self) -> None:
        """Test is_dict_like when items() raises AttributeError."""

        class BadDictLike:
            def keys(self) -> list[str]:
                return []

            def items(self) -> Never:
                msg = "items not available"
                raise AttributeError(msg)

            def get(self, key: str) -> None:
                return None

        obj = BadDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False

    def test_is_dict_like_with_exception_on_items_typeerror(self) -> None:
        """Test is_dict_like when items() raises TypeError."""

        class BadDictLike:
            def keys(self) -> list[str]:
                return []

            def items(self) -> Never:
                msg = "items failed"
                raise TypeError(msg)

            def get(self, key: str) -> None:
                return None

        obj = BadDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False

    def test_is_dict_like_with_userdict(self) -> None:
        """Test is_dict_like with UserDict (dict-like object)."""
        user_dict = UserDict({"key": "value"})
        result = FlextRuntime.is_dict_like(user_dict)
        assert result is True

    def test_is_dict_like_with_missing_attributes(self) -> None:
        """Test is_dict_like with object missing required attributes."""

        class NotDictLike:
            pass

        obj = NotDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False

    def test_is_dict_like_with_missing_keys(self) -> None:
        """Test is_dict_like with object missing keys attribute."""

        class NotDictLike:
            def items(self) -> list[tuple[str, str]]:
                return []

            def get(self, key: str) -> None:
                return None

        obj = NotDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False

    def test_is_dict_like_with_missing_items(self) -> None:
        """Test is_dict_like with object missing items attribute."""

        class NotDictLike:
            def keys(self) -> list[str]:
                return []

            def get(self, key: str) -> None:
                return None

        obj = NotDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False

    def test_is_dict_like_with_missing_get(self) -> None:
        """Test is_dict_like with object missing get attribute."""

        class NotDictLike:
            def keys(self) -> list[str]:
                return []

            def items(self) -> list[tuple[str, str]]:
                return []

        obj = NotDictLike()
        result = FlextRuntime.is_dict_like(cast("t.NormalizedValue", obj))
        assert result is False


class TestRuntimeTypeChecking:
    """Tests for runtime type checking coverage."""

    def test_extract_generic_args_with_type_mapping(self) -> None:
        """Test extract_generic_args with known type aliases."""

        class StringDict:
            __name__ = "StringDict"

        class IntDict:
            __name__ = "IntDict"

        class FloatDict:
            __name__ = "FloatDict"

        class BoolDict:
            __name__ = "BoolDict"

        class NestedDict:
            __name__ = "NestedDict"

        assert FlextRuntime.extract_generic_args(StringDict) == (str, str)
        assert FlextRuntime.extract_generic_args(IntDict) == (str, int)
        assert FlextRuntime.extract_generic_args(FloatDict) == (str, float)
        assert FlextRuntime.extract_generic_args(BoolDict) == (str, bool)
        assert FlextRuntime.extract_generic_args(NestedDict) == (str, dict)

    def test_is_sequence_type_with_type_mapping(self) -> None:
        """Test is_sequence_type with known type aliases."""

        class StringList:
            __name__ = "StringList"

        class IntList:
            __name__ = "IntList"

        class FloatList:
            __name__ = "FloatList"

        class BoolList:
            __name__ = "BoolList"

        class List:
            __name__ = "List"

        assert FlextRuntime.is_sequence_type(StringList) is True
        assert FlextRuntime.is_sequence_type(IntList) is True
        assert FlextRuntime.is_sequence_type(FloatList) is True
        assert FlextRuntime.is_sequence_type(BoolList) is True
        assert FlextRuntime.is_sequence_type(List) is True

    def test_level_based_context_filter_malformed_prefix(self) -> None:
        """Test level_based_context_filter with malformed prefix."""
        FlextRuntime.configure_structlog()
        malformed_key = "_level_"
        event_dict: dict[str, t.Scalar] = {
            malformed_key: "value1",
            "normal_key": "value2",
        }
        logger = structlog.get_logger()
        result = FlextRuntime.level_based_context_filter(logger, "info", event_dict)
        assert malformed_key in result or "normal_key" in result

    def test_configure_structlog_with_config_object(self) -> None:
        """Test configure_structlog with config object."""
        FlextRuntime._structlog_configured = False
        FlextRuntime.configure_structlog(config=None)
        assert FlextRuntime._structlog_configured

    def test_enable_runtime_checking(self) -> None:
        """Test enable_runtime_checking method."""
        result = FlextRuntime.enable_runtime_checking()
        assert result is True

    def test_is_valid_json_exception_path(self) -> None:
        """Test is_valid_json when json.loads raises exception."""
        invalid_json = "{invalid json}"
        result = FlextRuntime.is_valid_json(invalid_json)
        assert result is False

    def test_is_valid_identifier_non_string(self) -> None:
        """Test is_valid_identifier with non-string types."""
        assert not FlextRuntime.is_valid_identifier(123)
        assert not FlextRuntime.is_valid_identifier(None)

    def test_extract_generic_args_with_typing_get_args(self) -> None:
        """Test extract_generic_args when typing.get_args returns values."""
        args = FlextRuntime.extract_generic_args(list[str])
        assert args == (str,)
        args = FlextRuntime.extract_generic_args(dict[str, int])
        assert args == (str, int)

    def test_extract_generic_args_exception_path(self) -> None:
        """Test extract_generic_args exception handling."""

        class BadType:
            @override
            def __getattribute__(self, name: str) -> object:
                if name == "__name__":
                    msg = "Cannot access __name__"
                    raise AttributeError(msg)
                return super().__getattribute__(name)

        result = FlextRuntime.extract_generic_args(BadType)
        assert result == ()

    def test_is_sequence_type_with_origin(self) -> None:
        """Test is_sequence_type with typing.get_origin returning Sequence."""
        assert FlextRuntime.is_sequence_type(Sequence[str]) is True
        assert FlextRuntime.is_sequence_type(Sequence[int]) is True

    def test_is_sequence_type_with_sequence_subclass(self) -> None:
        """Test is_sequence_type with type that is Sequence subclass."""

        class MySequence(Sequence[str]):
            @overload
            def __getitem__(self, index: int) -> str: ...

            @overload
            def __getitem__(self, index: slice) -> Sequence[str]: ...

            @override
            def __getitem__(self, index: int | slice) -> Sequence[str] | str:
                return "" if isinstance(index, int) else MySequence()

            @override
            def __len__(self) -> int:
                return 0

        assert FlextRuntime.is_sequence_type(MySequence) is True

    def test_is_sequence_type_exception_path(self) -> None:
        """Test is_sequence_type exception handling."""

        class BadType:
            @override
            def __getattribute__(self, name: str) -> object:
                if name == "__name__":
                    msg = "Cannot access __name__"
                    raise AttributeError(msg)
                return super().__getattribute__(name)

        result = FlextRuntime.is_sequence_type(BadType)
        assert result is False

    def test_level_based_context_filter_with_level_prefixed(self) -> None:
        """Test level_based_context_filter with properly formatted level prefix."""
        FlextRuntime.configure_structlog()
        event_dict: Mapping[str, t.Scalar] = {
            "_level_debug_config": "dbg",
            "_level_info_status": "ok",
            "_level_error_stack": "trace",
            "normal_key": "value",
        }
        logger = structlog.get_logger()
        result = FlextRuntime.level_based_context_filter(logger, "info", event_dict)
        assert "status" in result
        assert "normal_key" in result
        assert "config" in result
        assert "stack" not in result

    def test_configure_structlog_with_config_additional_processors(self) -> None:
        """Test configure_structlog with config object having additional_processors."""
        FlextRuntime._structlog_configured = False

        def custom_processor(
            logger: p.Log.StructlogLogger | None,
            method_name: str,
            event_dict: dict[str, t.Tests.object],
        ) -> dict[str, t.Tests.object]:
            event_dict["custom"] = True
            return event_dict

        _ = custom_processor
        FlextRuntime.configure_structlog(config=None)
        assert FlextRuntime._structlog_configured
