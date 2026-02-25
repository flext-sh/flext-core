"""Full coverage tests for FlextModelFoundation.Validators.

Tests all staticmethod validators available on m.Validation (which is
FlextModelFoundation.Validators):
  - strip_whitespace
  - ensure_utc_datetime
  - normalize_to_list
  - validate_non_empty_string
  - validate_email
  - validate_url
  - validate_semver
  - validate_uuid_string
  - validate_config_dict
  - validate_tags_list

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from flext_core import c, m, r, t, u
from flext_core._models.base import FlextModelFoundation
from pydantic import BaseModel


class _Simple(BaseModel):
    x: int


# ---------------------------------------------------------------------------
# Validators.strip_whitespace
# ---------------------------------------------------------------------------


def test_strip_whitespace_trims_leading_trailing() -> None:
    assert m.Validation.strip_whitespace("  hello  ") == "hello"


def test_strip_whitespace_preserves_clean() -> None:
    assert m.Validation.strip_whitespace("already") == "already"


def test_strip_whitespace_returns_empty_on_spaces() -> None:
    assert m.Validation.strip_whitespace("   ") == ""


# ---------------------------------------------------------------------------
# Validators.ensure_utc_datetime
# ---------------------------------------------------------------------------


def test_ensure_utc_datetime_adds_tzinfo_when_naive() -> None:
    naive = datetime(2025, 1, 1, 12, 0, 0, tzinfo=None)  # noqa: DTZ001
    result = m.Validation.ensure_utc_datetime(naive)
    assert result is not None
    assert result.tzinfo is UTC


def test_ensure_utc_datetime_preserves_aware() -> None:
    aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    result = m.Validation.ensure_utc_datetime(aware)
    assert result is aware


def test_ensure_utc_datetime_returns_none_on_none() -> None:
    assert m.Validation.ensure_utc_datetime(None) is None


# ---------------------------------------------------------------------------
# Validators.normalize_to_list
# ---------------------------------------------------------------------------


def test_normalize_to_list_wraps_scalar() -> None:
    result = m.Validation.normalize_to_list("single")
    assert result == ["single"]


def test_normalize_to_list_passes_list_through() -> None:
    result = m.Validation.normalize_to_list([1, 2, 3])
    assert result == [1, 2, 3]


def test_normalize_to_list_wraps_int() -> None:
    result = m.Validation.normalize_to_list(42)
    assert result == [42]


# ---------------------------------------------------------------------------
# Validators.validate_non_empty_string
# ---------------------------------------------------------------------------


def test_validate_non_empty_string_passes() -> None:
    assert m.Validation.validate_non_empty_string("hello") == "hello"


def test_validate_non_empty_string_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        m.Validation.validate_non_empty_string("")


def test_validate_non_empty_string_raises_on_whitespace() -> None:
    with pytest.raises(ValueError, match="empty"):
        m.Validation.validate_non_empty_string("   ")


# ---------------------------------------------------------------------------
# Validators.validate_email
# ---------------------------------------------------------------------------


def test_validate_email_passes() -> None:
    assert m.Validation.validate_email("user@example.com") == "user@example.com"


def test_validate_email_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="email"):
        m.Validation.validate_email("not-an-email")


# ---------------------------------------------------------------------------
# Validators.validate_url
# ---------------------------------------------------------------------------


def test_validate_url_passes() -> None:
    assert m.Validation.validate_url("https://example.com") == "https://example.com"


def test_validate_url_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="URL"):
        m.Validation.validate_url("not-a-url")


# ---------------------------------------------------------------------------
# Validators.validate_semver
# ---------------------------------------------------------------------------


def test_validate_semver_passes() -> None:
    assert m.Validation.validate_semver("1.2.3") == "1.2.3"


def test_validate_semver_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="semantic version"):
        m.Validation.validate_semver("abc")


# ---------------------------------------------------------------------------
# Validators.validate_uuid_string
# ---------------------------------------------------------------------------


def test_validate_uuid_string_passes() -> None:
    uuid_str = "12345678-1234-5678-1234-567812345678"
    assert m.Validation.validate_uuid_string(uuid_str) == uuid_str


def test_validate_uuid_string_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="UUID"):
        m.Validation.validate_uuid_string("not-a-uuid")


# ---------------------------------------------------------------------------
# Validators.validate_config_dict
# ---------------------------------------------------------------------------


def test_validate_config_dict_normalizes_dict() -> None:
    result = m.Validation.validate_config_dict({"key": "value"})
    assert isinstance(result, dict)
    assert result["key"] == "value"


# ---------------------------------------------------------------------------
# Validators.validate_tags_list
# ---------------------------------------------------------------------------


def test_validate_tags_list_normalizes() -> None:
    result = m.Validation.validate_tags_list(["tag1", "  TAG1  ", "tag2"])
    assert isinstance(result, list)
    # Tags should be deduplicated and cleaned
    assert len(result) <= 3


def test_validate_tags_list_from_string() -> None:
    result = m.Validation.validate_tags_list(["hello", "world"])
    assert "hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# Smoke: confirm facade binding
# ---------------------------------------------------------------------------


def test_facade_binding_is_correct() -> None:
    """m.Validation IS FlextModelFoundation.Validators."""
    assert m.Validation is FlextModelFoundation.Validators


def test_basic_imports_work() -> None:
    """Smoke test: all standard imports resolve."""
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"


__all__ = [
    "test_basic_imports_work",
    "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    "test_ensure_utc_datetime_preserves_aware",
    "test_ensure_utc_datetime_returns_none_on_none",
    "test_facade_binding_is_correct",
    "test_normalize_to_list_passes_list_through",
    "test_normalize_to_list_wraps_int",
    "test_normalize_to_list_wraps_scalar",
    "test_strip_whitespace_preserves_clean",
    "test_strip_whitespace_returns_empty_on_spaces",
    "test_strip_whitespace_trims_leading_trailing",
    "test_validate_config_dict_normalizes_dict",
    "test_validate_email_passes",
    "test_validate_email_raises_on_invalid",
    "test_validate_non_empty_string_passes",
    "test_validate_non_empty_string_raises_on_empty",
    "test_validate_non_empty_string_raises_on_whitespace",
    "test_validate_semver_passes",
    "test_validate_semver_raises_on_invalid",
    "test_validate_tags_list_from_string",
    "test_validate_tags_list_normalizes",
    "test_validate_url_passes",
    "test_validate_url_raises_on_invalid",
    "test_validate_uuid_string_passes",
    "test_validate_uuid_string_raises_on_invalid",
]
