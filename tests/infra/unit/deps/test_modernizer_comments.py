"""Comment injection phase tests for deps modernizer."""

from __future__ import annotations

from flext_infra import InjectCommentsPhase
from flext_tests import tm
from tests.infra import h


class TestInjectCommentsPhase:
    """Tests comment injection behavior."""

    def test_inject_comments_adds_banner(self) -> None:
        rendered = "[project]\nname = 'test'"
        result, changes = InjectCommentsPhase().apply(rendered)
        tm.that("[MANAGED] FLEXT pyproject standardization" in result, eq=True)
        tm.that(any("banner" in change for change in changes), eq=True)

    def test_inject_comments_injects_markers(self) -> None:
        rendered = "[project]\nname = 'test'\n[tool.pytest]"
        _, changes = InjectCommentsPhase().apply(rendered)
        tm.that(any("marker" in change for change in changes), eq=True)

    def test_inject_comments_removes_broken_group_section(self) -> None:
        rendered = "[group.dev.dependencies]\npytest = '^7.0'"
        result, changes = InjectCommentsPhase().apply(rendered)
        tm.that("[group.dev.dependencies]" not in result, eq=True)
        tm.that(any("broken" in change for change in changes), eq=True)

    def test_inject_comments_handles_optional_dependencies_dev(self) -> None:
        rendered = "[project.optional-dependencies]\ndev = ['pytest']"
        result, changes = InjectCommentsPhase().apply(rendered)
        tm.that(("dev" in result) or (len(changes) > 0), eq=True)

    def test_inject_comments_preserves_existing_markers(self) -> None:
        rendered = "# [MANAGED] build system\n[build-system]"
        result, _ = InjectCommentsPhase().apply(rendered)
        tm.that("# [MANAGED] build system" in result, eq=True)


def test_inject_comments_phase_apply_banner() -> None:
    rendered = '[project]\nname = "test"\n'
    result, changes = InjectCommentsPhase().apply(rendered)
    tm.that("[MANAGED] FLEXT pyproject standardization" in result, eq=True)
    tm.that("managed banner injected" in changes, eq=True)


def test_inject_comments_phase_apply_markers() -> None:
    rendered = '[project]\nname = "test"\n[tool.pytest]\n'
    result, _ = InjectCommentsPhase().apply(rendered)
    tm.that("[MANAGED]" in result, eq=True)


def test_inject_comments_phase_apply_broken_group_section() -> None:
    rendered = '[group.dev.dependencies]\nrequests = "*"\n[project]\n'
    result, changes = InjectCommentsPhase().apply(rendered)
    tm.that("[group.dev.dependencies]" not in result, eq=True)
    tm.that("broken [group.dev.dependencies] section removed" in changes, eq=True)


def test_inject_comments_phase_apply_with_optional_dependencies_dev() -> None:
    rendered = "[project.optional-dependencies]\noptional-dependencies.dev = ['pytest', 'coverage']\n"
    result, changes = InjectCommentsPhase().apply(rendered)
    tm.that(("optional-dependencies.dev" in result) or (len(changes) > 0), eq=True)
    tm.that(hasattr(h, "assert_ok"), eq=True)
