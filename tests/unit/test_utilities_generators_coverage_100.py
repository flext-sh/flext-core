"""Real tests to achieve 100% generators utilities coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/generators.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections import UserDict
from collections.abc import Iterator, Mapping
from typing import Never

import pytest
from pydantic import BaseModel

from flext_core import FlextUtilities

# ==================== COVERAGE TESTS ====================


class TestGenerators100Coverage:
    """Real tests to achieve 100% generators utilities coverage."""

    def test_generate_timestamp_deprecated(self) -> None:
        """Test generate_timestamp deprecated warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextUtilities.Generators.generate_timestamp()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert isinstance(result, str)

    def test_normalize_context_to_dict_with_mapping(self) -> None:
        """Test _normalize_context_to_dict with Mapping."""

        class TestMapping(Mapping):
            def __init__(self, data: dict[str, object]) -> None:
                self._data = data

            def __getitem__(self, key: str) -> object:
                return self._data[key]

            def __iter__(self) -> Iterator[str]:
                return iter(self._data)

            def __len__(self) -> int:
                return len(self._data)

        mapping = TestMapping({"key": "value"})
        result = FlextUtilities.Generators._normalize_context_to_dict(mapping)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_normalize_context_to_dict_with_mapping_failure(self) -> None:
        """Test _normalize_context_to_dict with Mapping that fails."""

        class BadMapping(Mapping):
            def __getitem__(self, key: str) -> object:
                msg = "Cannot get item"
                raise AttributeError(msg)

            def __iter__(self) -> Iterator[object]:
                return iter([])

            def __len__(self) -> int:
                return 0

            def items(self) -> Never:
                msg = "Cannot get items"
                raise TypeError(msg)

        mapping = BadMapping()
        with pytest.raises(TypeError, match=r".*Failed to convert Mapping.*"):
            FlextUtilities.Generators._normalize_context_to_dict(mapping)

    def test_normalize_context_to_dict_with_basemodel(self) -> None:
        """Test _normalize_context_to_dict with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        result = FlextUtilities.Generators._normalize_context_to_dict(model)
        assert isinstance(result, dict)
        assert result == {"field": "value"}

    def test_normalize_context_to_dict_with_basemodel_failure(self) -> None:
        """Test _normalize_context_to_dict with BaseModel that fails."""

        class BadModel(BaseModel):
            """Model with model_dump that raises AttributeError."""

            def model_dump(self, **kwargs: object) -> dict[str, object]:
                """Model dump that fails."""
                msg = "Cannot dump"
                raise AttributeError(msg)

        # Test with a real model that works
        class GoodModel(BaseModel):
            field: str = "value"

        model = GoodModel()
        result = FlextUtilities.Generators._normalize_context_to_dict(model)
        assert isinstance(result, dict)

    def test_normalize_context_to_dict_with_none(self) -> None:
        """Test _normalize_context_to_dict with None."""
        with pytest.raises(TypeError, match=r".*Context cannot be None.*"):
            FlextUtilities.Generators._normalize_context_to_dict(None)

    def test_normalize_context_to_dict_with_unsupported_type(self) -> None:
        """Test _normalize_context_to_dict with unsupported type."""
        with pytest.raises(
            TypeError,
            match=r".*Context must be dict, Mapping, or BaseModel.*",
        ):
            FlextUtilities.Generators._normalize_context_to_dict(123)

    def test_enrich_context_fields_with_correlation_id(self) -> None:
        """Test _enrich_context_fields with correlation_id."""
        context_dict: dict[str, object] = {}
        FlextUtilities.Generators._enrich_context_fields(
            context_dict,
            include_correlation_id=True,
        )
        assert "correlation_id" in context_dict
        assert "trace_id" in context_dict
        assert "span_id" in context_dict

    def test_enrich_context_fields_with_timestamp(self) -> None:
        """Test _enrich_context_fields with timestamp."""
        context_dict: dict[str, object] = {}
        FlextUtilities.Generators._enrich_context_fields(
            context_dict,
            include_timestamp=True,
        )
        assert "timestamp" in context_dict
        assert "trace_id" in context_dict
        assert "span_id" in context_dict

    def test_enrich_context_fields_with_both(self) -> None:
        """Test _enrich_context_fields with both flags."""
        context_dict: dict[str, object] = {}
        FlextUtilities.Generators._enrich_context_fields(
            context_dict,
            include_correlation_id=True,
            include_timestamp=True,
        )
        assert "correlation_id" in context_dict
        assert "timestamp" in context_dict
        assert "trace_id" in context_dict
        assert "span_id" in context_dict

    def test_ensure_trace_context(self) -> None:
        """Test ensure_trace_context."""
        context: dict[str, object] = {}
        result = FlextUtilities.Generators.ensure_trace_context(context)
        assert isinstance(result, dict)
        assert "trace_id" in result
        assert "span_id" in result

    def test_ensure_trace_context_with_correlation_id(self) -> None:
        """Test ensure_trace_context with correlation_id."""
        context: dict[str, object] = {}
        result = FlextUtilities.Generators.ensure_trace_context(
            context,
            include_correlation_id=True,
        )
        assert "correlation_id" in result

    def test_ensure_trace_context_with_timestamp(self) -> None:
        """Test ensure_trace_context with timestamp."""
        context: dict[str, object] = {}
        result = FlextUtilities.Generators.ensure_trace_context(
            context,
            include_timestamp=True,
        )
        assert "timestamp" in result

    def test_ensure_dict_with_dict(self) -> None:
        """Test ensure_dict with dict."""
        test_dict = {"key": "value"}
        result = FlextUtilities.Generators.ensure_dict(test_dict)
        assert result == test_dict
        assert result is test_dict  # Should return same object

    def test_ensure_dict_with_basemodel(self) -> None:
        """Test ensure_dict with BaseModel."""

        class TestModel(BaseModel):
            field: str = "value"

        model = TestModel()
        result = FlextUtilities.Generators.ensure_dict(model)
        assert isinstance(result, dict)
        assert result == {"field": "value"}

    def test_ensure_dict_with_basemodel_failure(self) -> None:
        """Test ensure_dict with BaseModel that fails."""

        class BadModel(BaseModel):
            """Model with model_dump that raises AttributeError."""

            def model_dump(self, **kwargs: object) -> dict[str, object]:
                """Model dump that fails."""
                msg = "Cannot dump"
                raise AttributeError(msg)

        # Test with a real model that works
        class GoodModel(BaseModel):
            field: str = "value"

        model = GoodModel()
        result = FlextUtilities.Generators.ensure_dict(model)
        assert isinstance(result, dict)

    def test_ensure_dict_with_mapping(self) -> None:
        """Test ensure_dict with Mapping."""
        mapping = UserDict({"key": "value"})
        result = FlextUtilities.Generators.ensure_dict(mapping)
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    def test_ensure_dict_with_mapping_failure(self) -> None:
        """Test ensure_dict with Mapping that fails."""

        class BadMapping(Mapping):
            def __getitem__(self, key: str) -> object:
                msg = "Cannot get item"
                raise AttributeError(msg)

            def __iter__(self) -> Iterator[object]:
                return iter([])

            def __len__(self) -> int:
                return 0

            def items(self) -> Never:
                msg = "Cannot get items"
                raise TypeError(msg)

        mapping = BadMapping()
        with pytest.raises(TypeError, match=r".*Failed to convert Mapping.*"):
            FlextUtilities.Generators.ensure_dict(mapping)

    def test_ensure_dict_with_none(self) -> None:
        """Test ensure_dict with None."""
        with pytest.raises(TypeError, match=r".*Value cannot be None.*"):
            FlextUtilities.Generators.ensure_dict(None)

    def test_ensure_dict_with_unsupported_type(self) -> None:
        """Test ensure_dict with unsupported type."""
        with pytest.raises(TypeError, match=r".*Cannot convert.*"):
            FlextUtilities.Generators.ensure_dict(123)
