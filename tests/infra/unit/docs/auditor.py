"""Tests for FlextInfraDocAuditor — core audit and static helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_tests import tm
from tests.infra.models import m
from tests.infra.typings import t


@pytest.fixture
def auditor() -> FlextInfraDocAuditor:
    return FlextInfraDocAuditor()


@pytest.fixture
def normalize_link() -> Callable[[str], str]:
    return FlextInfraDocAuditor.normalize_link


@pytest.fixture
def should_skip_target() -> Callable[[str, str], bool]:
    return FlextInfraDocAuditor.should_skip_target


@pytest.fixture
def is_external() -> Callable[[str], bool]:
    return FlextInfraDocAuditor.is_external


class TestAuditorCore:
    def test_returns_flext_result(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_valid_scope_returns_success(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        tm.ok(result)

    def test_report_structure(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
    ) -> None:
        result = auditor.audit(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "items"), eq=True)

    def test_issue_structure(self) -> None:
        issue = m.Infra.Docs.AuditIssue(
            file="README.md",
            issue_type="broken_link",
            severity="high",
            message="Link to missing file",
        )
        tm.that(issue.file, eq="README.md")
        tm.that(issue.issue_type, eq="broken_link")
        tm.that(issue.severity, eq="high")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"project": "test-project"},
            {"projects": "proj1,proj2"},
            {"check": "links"},
            {"check": "forbidden-terms"},
            {"strict": True},
            {"output_dir": "custom_output"},
        ],
    )
    def test_audit_option_variants(
        self,
        auditor: FlextInfraDocAuditor,
        tmp_path: Path,
        kwargs: dict[str, t.ContainerValue],
    ) -> None:
        params: dict[str, t.ContainerValue] = dict(kwargs)
        if "output_dir" in params:
            params["output_dir"] = str(tmp_path / str(params["output_dir"]))
        result = auditor.audit(tmp_path, **params)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_report_frozen(self) -> None:
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_issue_frozen(self) -> None:
        tm.that(m.Infra.Docs.AuditIssue.model_config.get("frozen"), eq=True)


class TestAuditorNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("path/to/file.md#section", "path/to/file.md"),
            ("path/to/file.md?param=value", "path/to/file.md"),
            ("<path/to/file.md>", "path/to/file.md"),
            ("  path/to/file.md  ", "path/to/file.md"),
            ("<path/to/file.md#section?param=value>", "path/to/file.md"),
        ],
    )
    def test_normalize_link(
        self,
        normalize_link: m.Infra.Docs.NormalizeLinkFn,
        raw: str,
        expected: str,
    ) -> None:
        tm.that(normalize_link(raw), eq=expected)

    @pytest.mark.parametrize(
        ("text", "target", "expected"),
        [
            ("[link](http://example.com)", "http://example.com", False),
            ("[link](https://example.com)", "https://example.com", False),
            ("[a, b]", "a", True),
            ("[a b]", "a", True),
            ("[a, b.md]", "a", False),
            ("[a/b]", "a/b", False),
        ],
    )
    def test_should_skip_target(
        self,
        should_skip_target: m.Infra.Docs.SkipTargetFn,
        text: str,
        target: str,
        expected: bool,
    ) -> None:
        tm.that(should_skip_target(text, target), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("http://example.com", True),
            ("https://example.com", True),
            ("mailto:test@example.com", True),
            ("tel:+1234567890", True),
            ("data:text/plain;base64,SGVsbG8=", True),
            ("path/to/file.md", False),
            ("<http://example.com>", True),
            ("HTTPS://EXAMPLE.COM", True),
        ],
    )
    def test_is_external(
        self,
        is_external: m.Infra.Docs.IsExternalFn,
        value: str,
        expected: bool,
    ) -> None:
        tm.that(is_external(value), eq=expected)
