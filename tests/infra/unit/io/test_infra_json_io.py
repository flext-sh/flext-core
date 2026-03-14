"""Tests for FlextInfraJsonService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesIo
from flext_tests import tf, tm
from tests.unit._models import SampleModel


class TestFlextInfraJsonService:
    """Test suite for FlextInfraJsonService."""

    @pytest.mark.parametrize(
        ("name", "content", "should_succeed", "expected_fragment"),
        [
            ("test.json", '{"key": "value", "number": 42}', True, "key"),
            ("missing.json", None, True, "{}"),
            ("invalid.json", "{invalid json}", False, "JSON read error"),
            ("array.json", "[1, 2, 3]", False, "must be object"),
        ],
        ids=["read-existing", "read-missing", "read-invalid", "read-non-object"],
    )
    def test_read_json_scenarios(
        self,
        tmp_path: Path,
        name: str,
        content: str | None,
        should_succeed: bool,
        expected_fragment: str,
    ) -> None:
        json_file = tmp_path / name
        if content is not None:
            json_file.write_text(content, encoding="utf-8")
        service = FlextInfraUtilitiesIo()
        result = service.read_json(json_file)
        if should_succeed:
            tm.ok(result)
            tm.that(str(result.value), has=expected_fragment)
            return
        tm.fail(result, has=expected_fragment)

    @pytest.mark.parametrize(
        ("path_parts", "payload", "sort_keys", "ensure_ascii", "expected"),
        [
            (("output.json",), {"key": "value", "number": 42}, False, False, '"key"'),
            (
                ("model.json",),
                SampleModel(name="test", value=123),
                False,
                False,
                '"name"',
            ),
            (("nested", "deep", "file.json"), {"key": "value"}, False, False, '"key"'),
            (("sorted.json",), {"z": 1, "a": 2, "m": 3}, True, False, '"a"'),
            (("ascii.json",), {"text": "café"}, False, True, "\\u"),
        ],
        ids=[
            "write-dict",
            "write-model",
            "write-nested",
            "write-sorted-keys",
            "write-ensure-ascii",
        ],
    )
    def test_write_json_scenarios(
        self,
        tmp_path: Path,
        path_parts: tuple[str, ...],
        payload: dict[str, str | int] | SampleModel,
        sort_keys: bool,
        ensure_ascii: bool,
        expected: str,
    ) -> None:
        json_file = tmp_path.joinpath(*path_parts)
        service = FlextInfraUtilitiesIo()
        result = service.write_json(
            json_file,
            payload,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )
        tm.ok(result)
        tm.that(json_file.exists(), eq=True)
        tm.that(json_file.read_text(encoding="utf-8"), has=expected)
        if sort_keys:
            content = json_file.read_text(encoding="utf-8")
            tm.that(content.index('"a"') < content.index('"z"'), eq=True)

    def test_write_permission_error(self, tmp_path: Path) -> None:
        """Test write failure on permission error."""
        json_file = tmp_path / "readonly.json"
        _ = tf.create_in("{}", "readonly.json", tmp_path)
        json_file.chmod(292)
        service = FlextInfraUtilitiesIo()
        try:
            result = service.write_json(json_file, {"key": "value"})
            tm.fail(result)
        finally:
            json_file.chmod(420)

    def test_write_returns_true_on_success(self, tmp_path: Path) -> None:
        """Test write returns True on success."""
        json_file = tmp_path / "test.json"
        service = FlextInfraUtilitiesIo()
        result = service.write_json(json_file, {"key": "value"})
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_removed_direct_api_methods_raise_attribute_error(self) -> None:
        """Direct-return compatibility methods are no longer available."""
        service = FlextInfraUtilitiesIo()
        with pytest.raises(AttributeError):
            _ = getattr(service, "load")
        with pytest.raises(AttributeError):
            _ = getattr(service, "dump")
        with pytest.raises(AttributeError):
            _ = getattr(service, "loads")
        with pytest.raises(AttributeError):
            _ = getattr(service, "dumps")
        with pytest.raises(AttributeError):
            _ = getattr(service, "is_json")
