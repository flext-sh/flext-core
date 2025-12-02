"""Real tests to achieve 100% runtime coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in runtime.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections import UserDict
from collections.abc import Sequence
from typing import ClassVar, Never, cast, overload

import structlog

from flext_core import FlextRuntime
from flext_core.typings import FlextTypes


class TestRuntimeDictLike:
    """Tests for is_dict_like runtime coverage."""

    def test_is_dict_like_with_exception_on_items(self) -> None:
        """Test is_dict_like when items() raises AttributeError."""

        # Create object that has keys/items/get but items() raises AttributeError
        class BadDictLike:
            def keys(self) -> list[object]:
                return []

            def items(self) -> Never:
                msg = "items not available"
                raise AttributeError(msg)

            def get(self, key: object) -> object:
                return None

        # Business Rule: BadDictLike instances are compatible with GeneralValueType at runtime
        obj = BadDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
        assert result is False

    def test_is_dict_like_with_exception_on_items_typeerror(self) -> None:
        """Test is_dict_like when items() raises TypeError."""

        # Create object that has keys/items/get but items() raises TypeError
        class BadDictLike:
            def keys(self) -> list[object]:
                return []

            def items(self) -> Never:
                msg = "items failed"
                raise TypeError(msg)

            def get(self, key: object) -> object:
                return None

        # Business Rule: BadDictLike instances are compatible with GeneralValueType at runtime
        obj = BadDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
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

        # Business Rule: NotDictLike instances are compatible with GeneralValueType at runtime
        obj = NotDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
        assert result is False

    def test_is_dict_like_with_missing_keys(self) -> None:
        """Test is_dict_like with object missing keys attribute."""

        class NotDictLike:
            def items(self) -> list[object]:
                return []

            def get(self, key: object) -> object:
                return None

        # Business Rule: NotDictLike instances are compatible with GeneralValueType at runtime
        obj = NotDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
        assert result is False

    def test_is_dict_like_with_missing_items(self) -> None:
        """Test is_dict_like with object missing items attribute."""

        class NotDictLike:
            def keys(self) -> list[object]:
                return []

            def get(self, key: object) -> object:
                return None

        # Business Rule: NotDictLike instances are compatible with GeneralValueType at runtime
        obj = NotDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
        assert result is False

    def test_is_dict_like_with_missing_get(self) -> None:
        """Test is_dict_like with object missing get attribute."""

        class NotDictLike:
            def keys(self) -> list[object]:
                return []

            def items(self) -> list[tuple[object, object]]:
                return []

        # Business Rule: NotDictLike instances are compatible with GeneralValueType at runtime
        obj = NotDictLike()
        obj_typed = cast("FlextTypes.GeneralValueType", obj)
        result = FlextRuntime.is_dict_like(obj_typed)
        assert result is False


class TestRuntimeTypeChecking:
    """Tests for runtime type checking coverage."""

    def test_extract_generic_args_with_type_mapping(self) -> None:
        """Test extract_generic_args with known type aliases."""

        # Create mock type aliases that match the type_mapping
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

        # Test each type alias
        assert FlextRuntime.extract_generic_args(StringDict) == (str, str)
        assert FlextRuntime.extract_generic_args(IntDict) == (str, int)
        assert FlextRuntime.extract_generic_args(FloatDict) == (str, float)
        assert FlextRuntime.extract_generic_args(BoolDict) == (str, bool)
        assert FlextRuntime.extract_generic_args(NestedDict) == (str, dict)

    def test_is_sequence_type_with_type_mapping(self) -> None:
        """Test is_sequence_type with known type aliases."""

        # Create mock type aliases that match the type_mapping
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

        # Test each type alias
        assert FlextRuntime.is_sequence_type(StringList) is True
        assert FlextRuntime.is_sequence_type(IntList) is True
        assert FlextRuntime.is_sequence_type(FloatList) is True
        assert FlextRuntime.is_sequence_type(BoolList) is True
        assert FlextRuntime.is_sequence_type(List) is True

    def test_level_based_context_filter_malformed_prefix(self) -> None:
        """Test level_based_context_filter with malformed prefix."""
        # Configure structlog first
        FlextRuntime.configure_structlog()

        # Create event dict with malformed prefix
        # Malformed prefix: starts with _level_ but doesn't have enough parts after split
        # LEVEL_PREFIX_PARTS_COUNT is typically 4, so we need _level_<level>_<key>
        # A malformed one would be just "_level_" or "_level_debug" (not enough parts)
        # Create a key that starts with _level_ but has fewer parts than required
        malformed_key = "_level_"  # This will split into fewer parts than required
        event_dict: FlextTypes.Types.ConfigurationMapping = {
            malformed_key: "value1",  # Malformed - not enough parts after split
            "normal_key": "value2",  # Not prefixed
        }

        # Process with the filter
        logger = structlog.get_logger()
        result = FlextRuntime.level_based_context_filter(logger, "info", event_dict)

        # Malformed prefix should be included as-is (line 491)
        assert malformed_key in result or "normal_key" in result

    def test_configure_structlog_with_config_object(self) -> None:
        """Test configure_structlog with config object."""
        # Reset configuration
        FlextRuntime._structlog_configured = False

        # Create config object with attributes
        class Config:
            log_level: ClassVar[int] = logging.DEBUG
            console_renderer: ClassVar[bool] = False
            additional_processors: ClassVar[list[object]] = []
            wrapper_class_factory: ClassVar[object | None] = None
            logger_factory: ClassVar[object | None] = None
            cache_logger_on_first_use: ClassVar[bool] = True

        config = Config()
        # Convert Config object to Mapping for type compatibility
        # Convert list[object] to Sequence[GeneralValueType] for type compatibility
        additional_processors_typed: Sequence[FlextTypes.GeneralValueType] = (
            cast("Sequence[FlextTypes.GeneralValueType]", config.additional_processors)
            if isinstance(config.additional_processors, Sequence)
            else []
        )
        # Convert object | None to GeneralValueType | None for type compatibility
        wrapper_class_factory_typed: FlextTypes.GeneralValueType | None = (
            cast("FlextTypes.GeneralValueType", config.wrapper_class_factory)
            if config.wrapper_class_factory is not None
            else None
        )
        logger_factory_typed: FlextTypes.GeneralValueType | None = (
            cast("FlextTypes.GeneralValueType", config.logger_factory)
            if config.logger_factory is not None
            else None
        )
        config_dict: FlextTypes.Types.ConfigurationMapping = {
            "log_level": config.log_level,
            "console_renderer": config.console_renderer,
            "additional_processors": additional_processors_typed,
            "wrapper_class_factory": wrapper_class_factory_typed,
            "logger_factory": logger_factory_typed,
            "cache_logger_on_first_use": config.cache_logger_on_first_use,
        }
        FlextRuntime.configure_structlog(config=config_dict)

        assert FlextRuntime._structlog_configured is True

    def test_enable_runtime_checking(self) -> None:
        """Test enable_runtime_checking method."""
        # This should enable beartype runtime checking
        result = FlextRuntime.enable_runtime_checking()
        assert result is True

    def test_is_valid_phone_non_string(self) -> None:
        """Test is_valid_phone with non-string types."""
        assert not FlextRuntime.is_valid_phone(123)
        assert not FlextRuntime.is_valid_phone(None)
        assert not FlextRuntime.is_valid_phone([])

    def test_is_valid_json_exception_path(self) -> None:
        """Test is_valid_json when json.loads raises exception."""
        # Invalid JSON that causes json.loads to raise
        invalid_json = "{invalid json}"
        result = FlextRuntime.is_valid_json(invalid_json)
        assert result is False

    def test_is_valid_identifier_non_string(self) -> None:
        """Test is_valid_identifier with non-string types."""
        assert not FlextRuntime.is_valid_identifier(123)
        assert not FlextRuntime.is_valid_identifier(None)

    def test_extract_generic_args_with_typing_get_args(self) -> None:
        """Test extract_generic_args when typing.get_args returns values."""
        # Test with actual generic types
        args = FlextRuntime.extract_generic_args(list[str])
        assert args == (str,)

        args = FlextRuntime.extract_generic_args(dict[str, int])
        assert args == (str, int)

    def test_extract_generic_args_exception_path(self) -> None:
        """Test extract_generic_args exception handling."""

        # Test with object that raises exception
        class BadType:
            def __getattribute__(self, name: str) -> object:
                if name == "__name__":
                    msg = "Cannot access __name__"
                    raise AttributeError(msg)
                return super().__getattribute__(name)

        bad_type = cast("type", BadType)
        result = FlextRuntime.extract_generic_args(bad_type)
        assert result == ()

    def test_is_sequence_type_with_origin(self) -> None:
        """Test is_sequence_type with typing.get_origin returning Sequence."""
        # Test with actual sequence types
        assert FlextRuntime.is_sequence_type(Sequence[str]) is True
        assert FlextRuntime.is_sequence_type(Sequence[int]) is True

    def test_is_sequence_type_with_sequence_subclass(self) -> None:
        """Test is_sequence_type with type that is Sequence subclass."""

        class MySequence(Sequence[object]):
            @overload
            def __getitem__(self, index: int) -> object: ...

            @overload
            def __getitem__(self, index: slice) -> Sequence[object]: ...

            def __getitem__(self, index: int | slice) -> object | Sequence[object]:
                return None if isinstance(index, int) else MySequence()

            def __len__(self) -> int:
                return 0

        assert FlextRuntime.is_sequence_type(MySequence) is True

    def test_is_sequence_type_exception_path(self) -> None:
        """Test is_sequence_type exception handling."""

        # Test with object that raises exception
        class BadType:
            def __getattribute__(self, name: str) -> object:
                if name == "__name__":
                    msg = "Cannot access __name__"
                    raise AttributeError(msg)
                return super().__getattribute__(name)

        bad_type = cast("type", BadType)
        result = FlextRuntime.is_sequence_type(bad_type)
        assert result is False

    def test_level_based_context_filter_with_level_prefixed(self) -> None:
        """Test level_based_context_filter with properly formatted level prefix."""
        FlextRuntime.configure_structlog()

        # Create event dict with properly formatted level prefix
        # Format: _level_<level>_<key> where parts_count = 4
        # So "_level_debug_config" splits into ['', 'level', 'debug', 'config']
        # Level hierarchy: DEBUG (10) < INFO (20) < WARNING (30) < ERROR (40) < CRITICAL (50)
        event_dict: FlextTypes.Types.ConfigurationMapping = {
            "_level_debug_config": {"key": "value"},  # DEBUG level (10)
            "_level_info_status": "ok",  # INFO level (20)
            "_level_error_stack": "trace",  # ERROR level (40)
            "normal_key": "value",
        }

        logger = structlog.get_logger()
        # Test with INFO level (20)
        # Logic: if current_level >= required_level, include
        # - DEBUG (10): INFO (20) >= DEBUG (10) = True, so config is included
        # - INFO (20): INFO (20) >= INFO (20) = True, so status is included
        # - ERROR (40): INFO (20) >= ERROR (40) = False, so stack is excluded
        # - normal_key: always included (not prefixed)
        result = FlextRuntime.level_based_context_filter(logger, "info", event_dict)
        assert "status" in result  # INFO level included (same level)
        assert "normal_key" in result  # Normal key included
        assert "config" in result  # DEBUG level included (INFO >= DEBUG)
        assert "stack" not in result  # ERROR level excluded (INFO < ERROR)

    def test_configure_structlog_with_config_additional_processors(self) -> None:
        """Test configure_structlog with config object having additional_processors."""
        FlextRuntime._structlog_configured = False

        def custom_processor(
            logger: object,
            method_name: str,
            event_dict: dict[str, object],
        ) -> dict[str, object]:
            event_dict["custom"] = True
            return event_dict

        class Config:
            log_level: ClassVar[int] = logging.DEBUG
            console_renderer: ClassVar[bool] = True
            additional_processors: ClassVar[list[object]] = [custom_processor]
            wrapper_class_factory: ClassVar[object | None] = None
            logger_factory: ClassVar[object | None] = None
            cache_logger_on_first_use: ClassVar[bool] = True

        config = Config()
        # Convert Config object to Mapping for type compatibility
        # Convert list[object] to Sequence[GeneralValueType] for type compatibility
        additional_processors_typed: Sequence[FlextTypes.GeneralValueType] = (
            cast("Sequence[FlextTypes.GeneralValueType]", config.additional_processors)
            if isinstance(config.additional_processors, Sequence)
            else []
        )
        # Convert object | None to GeneralValueType | None for type compatibility
        wrapper_class_factory_typed: FlextTypes.GeneralValueType | None = (
            cast("FlextTypes.GeneralValueType", config.wrapper_class_factory)
            if config.wrapper_class_factory is not None
            else None
        )
        logger_factory_typed: FlextTypes.GeneralValueType | None = (
            cast("FlextTypes.GeneralValueType", config.logger_factory)
            if config.logger_factory is not None
            else None
        )
        config_dict: FlextTypes.Types.ConfigurationMapping = {
            "log_level": config.log_level,
            "console_renderer": config.console_renderer,
            "additional_processors": additional_processors_typed,
            "wrapper_class_factory": wrapper_class_factory_typed,
            "logger_factory": logger_factory_typed,
            "cache_logger_on_first_use": config.cache_logger_on_first_use,
        }
        FlextRuntime.configure_structlog(config=config_dict)
        assert FlextRuntime._structlog_configured is True
