"""Tests for FlextInfraVersioningService.

Tests cover semantic version parsing, bumping, and branch tag extraction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import FlextInfraUtilitiesVersioning
from flext_tests import tf, tm


@pytest.fixture
def service() -> FlextInfraUtilitiesVersioning:
    return FlextInfraUtilitiesVersioning()


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.2.3", (1, 2, 3)),
        ("1.2.3-dev", (1, 2, 3)),
        ("0.0.0", (0, 0, 0)),
        ("999.888.777", (999, 888, 777)),
    ],
    ids=["standard", "dev-suffix", "zero", "large"],
)
def test_parse_semver_valid(
    service: FlextInfraUtilitiesVersioning,
    version: str,
    expected: tuple[int, int, int],
) -> None:
    tm.ok(service.parse_semver(version), eq=expected)


@pytest.mark.parametrize(
    "version",
    ["1.2", "a.b.c", "1.2.3-beta"],
    ids=["invalid-format", "non-numeric", "unsupported-suffix"],
)
def test_parse_semver_invalid(
    service: FlextInfraUtilitiesVersioning, version: str
) -> None:
    tm.fail(service.parse_semver(version), has="invalid semver")


def test_parse_semver_result_type(service: FlextInfraUtilitiesVersioning) -> None:
    tm.ok(service.parse_semver("1.2.3"), is_=tuple)


@pytest.mark.parametrize(
    ("version", "bump_type", "expected"),
    [
        ("1.2.3", "major", "2.0.0"),
        ("1.2.3", "minor", "1.3.0"),
        ("1.2.3", "patch", "1.2.4"),
        ("0.0.0", "major", "1.0.0"),
    ],
    ids=["major", "minor", "patch", "from-zero"],
)
def test_bump_version_valid(
    service: FlextInfraUtilitiesVersioning,
    version: str,
    bump_type: str,
    expected: str,
) -> None:
    tm.ok(service.bump_version(version, bump_type), eq=expected)


@pytest.mark.parametrize(
    ("version", "bump_type", "error"),
    [
        ("1.2.3", "invalid", "invalid bump type"),
        ("not.a.version", "major", "invalid semver"),
    ],
    ids=["invalid-bump-type", "invalid-version"],
)
def test_bump_version_invalid(
    service: FlextInfraUtilitiesVersioning,
    version: str,
    bump_type: str,
    error: str,
) -> None:
    tm.fail(service.bump_version(version, bump_type), has=error)


def test_bump_version_result_type(service: FlextInfraUtilitiesVersioning) -> None:
    tm.ok(service.bump_version("1.2.3", "major"), is_=str)


@pytest.mark.parametrize(
    ("branch", "expected"),
    [("release/1.2.3", "v1.2.3"), ("1.2.3-dev", "v1.2.3")],
    ids=["release-pattern", "dev-pattern"],
)
def test_release_tag_from_branch_valid(
    service: FlextInfraUtilitiesVersioning,
    branch: str,
    expected: str,
) -> None:
    tm.ok(service.release_tag_from_branch(branch), eq=expected)


@pytest.mark.parametrize(
    "branch",
    ["feature/my-feature", ""],
    ids=["invalid-pattern", "empty"],
)
def test_release_tag_from_branch_invalid(
    service: FlextInfraUtilitiesVersioning,
    branch: str,
) -> None:
    tm.fail(
        service.release_tag_from_branch(branch), has="does not match release pattern"
    )


def test_release_tag_from_branch_result_type(
    service: FlextInfraUtilitiesVersioning,
) -> None:
    tm.ok(service.release_tag_from_branch("release/1.2.3"), is_=str)


@pytest.mark.parametrize(
    ("content", "expected", "error"),
    [
        ('[project]\nversion = "1.2.3"\n', "1.2.3", ""),
        (None, "", ""),
        ('[tool]\nname = "test"\n', "", "version not found"),
        ('[project]\nname = "test"\n', "", "version not found"),
        ('[project]\nversion = ""\n', "", "version not found"),
    ],
    ids=[
        "success",
        "missing-file",
        "missing-project-table",
        "missing-version",
        "empty-version",
    ],
)
def test_current_workspace_version(
    service: FlextInfraUtilitiesVersioning,
    tmp_path: Path,
    content: str | None,
    expected: str,
    error: str,
) -> None:
    if content is not None:
        tf.create_in(content, "pyproject.toml", tmp_path)
    result = service.current_workspace_version(tmp_path)
    if expected:
        tm.ok(result, eq=expected)
        return
    if error:
        tm.fail(result, has=error)
        return
    tm.fail(result)


@pytest.mark.parametrize(
    ("content", "expect_success", "error"),
    [
        ('[project]\nversion = "1.0.0"\n', True, ""),
        (None, False, ""),
        ('[tool]\nname = "test"\n', False, "missing [project] table"),
    ],
    ids=["success", "missing-file", "missing-project-table"],
)
def test_replace_project_version(
    service: FlextInfraUtilitiesVersioning,
    tmp_path: Path,
    content: str | None,
    expect_success: bool,
    error: str,
) -> None:
    if content is not None:
        tf.create_in(content, "pyproject.toml", tmp_path)
    result = service.replace_project_version(tmp_path, "2.0.0")
    if error:
        tm.fail(result, has=error)
        return
    if not expect_success:
        tm.fail(result)
        return
    tm.ok(result, eq=True)
    if content is not None:
        tm.ok(service.current_workspace_version(tmp_path), eq="2.0.0")
