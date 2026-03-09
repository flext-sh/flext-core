"""Tests for FlextInfraPrManager core operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from flext_core import r
from flext_infra.github.pr import FlextInfraPrManager

from tests.infra.unit.github._stubs import StubRunner, StubVersioning


def _mgr(
    runner: StubRunner | None = None,
    versioning: StubVersioning | None = None,
) -> FlextInfraPrManager:
    return FlextInfraPrManager(
        runner=runner or StubRunner(),
        versioning=versioning or StubVersioning(),
    )


class TestFlextInfraPrManager:
    """Test suite for FlextInfraPrManager."""

    def test_open_pr_for_head_found(self, tmp_path: Path) -> None:
        pr_data = {"number": 42, "title": "Feature", "state": "OPEN",
                   "baseRefName": "main", "headRefName": "feature/new",
                   "url": "https://github.com/org/repo/pull/42", "isDraft": False}
        runner = StubRunner(capture_returns=[r[str].ok(json.dumps([pr_data]))])
        result = _mgr(runner=runner).open_pr_for_head(tmp_path, "feature/new")
        assert result.is_success
        assert result.value["number"] == 42

    def test_open_pr_for_head_not_found(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("[]")])
        result = _mgr(runner=runner).open_pr_for_head(tmp_path, "feature/x")
        assert result.is_success
        assert result.value == {}

    def test_open_pr_for_head_json_error(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("invalid json")])
        result = _mgr(runner=runner).open_pr_for_head(tmp_path, "feature/test")
        assert result.is_failure
        assert result.error

    def test_open_pr_for_head_command_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].fail("gh command failed")])
        result = _mgr(runner=runner).open_pr_for_head(tmp_path, "feature/test")
        assert result.is_failure
        assert result.error

    def test_default_initialization(self) -> None:
        manager = FlextInfraPrManager()
        assert manager._runner is not None

    def test_open_pr_for_head_non_dict_first(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok(json.dumps(["not-a-dict"]))])
        result = _mgr(runner=runner).open_pr_for_head(tmp_path, "head")
        assert result.is_success
        assert result.value == {}


class TestStatus:
    """Test status method."""

    def test_status_open_pr(self, tmp_path: Path) -> None:
        pr_data = {"number": 10, "title": "Test PR", "state": "OPEN",
                   "url": "https://github.com/o/r/pull/10", "isDraft": False}
        runner = StubRunner(capture_returns=[r[str].ok(json.dumps([pr_data]))])
        result = _mgr(runner=runner).status(tmp_path, "main", "feature")
        assert result.is_success
        assert result.value["status"] == "open"
        assert result.value["pr_number"] == 10

    def test_status_no_pr(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].ok("[]")])
        result = _mgr(runner=runner).status(tmp_path, "main", "feature")
        assert result.is_success
        assert result.value["status"] == "no-open-pr"

    def test_status_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].fail("gh error")])
        result = _mgr(runner=runner).status(tmp_path, "main", "feature")
        assert result.is_failure


class TestCreate:
    """Test create method."""

    def test_create_new(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[
            r[str].ok("[]"),
            r[str].ok("https://github.com/o/r/pull/99"),
        ])
        result = _mgr(runner=runner).create(
            tmp_path, "main", "feature", "title", "body",
        )
        assert result.is_success
        assert result.value["status"] == "created"

    def test_create_already_open(self, tmp_path: Path) -> None:
        pr_data = {"url": "https://github.com/o/r/pull/10"}
        runner = StubRunner(capture_returns=[r[str].ok(json.dumps([pr_data]))])
        result = _mgr(runner=runner).create(
            tmp_path, "main", "feature", "title", "body",
        )
        assert result.is_success
        assert result.value["status"] == "already-open"

    def test_create_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[
            r[str].ok("[]"),
            r[str].fail("create failed"),
        ])
        result = _mgr(runner=runner).create(
            tmp_path, "main", "feature", "title", "body",
        )
        assert result.is_failure

    def test_create_with_draft(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[
            r[str].ok("[]"),
            r[str].ok("https://github.com/o/r/pull/100"),
        ])
        result = _mgr(runner=runner).create(
            tmp_path, "main", "feature", "title", "body", draft=True,
        )
        assert result.is_success
        assert "--draft" in runner.capture_calls[1]

    def test_create_check_existing_failure(self, tmp_path: Path) -> None:
        runner = StubRunner(capture_returns=[r[str].fail("gh error")])
        result = _mgr(runner=runner).create(
            tmp_path, "main", "feature", "title", "body",
        )
        assert result.is_failure
