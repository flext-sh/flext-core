from __future__ import annotations

from flext_core import r
from flext_infra.deps.path_sync import (
    _extract_requirement_name,
    _target_path,
    extract_dep_name,
)
from flext_tests import tm


class TestExtractDepName:
    def test_extract_dep_name_simple(self) -> None:
        tm.that(extract_dep_name("flext-core"), eq="flext-core")

    def test_extract_dep_name_with_prefix(self) -> None:
        tm.that(extract_dep_name(".flext-deps/flext-core"), eq="flext-core")

    def test_extract_dep_name_with_parent_ref(self) -> None:
        tm.that(extract_dep_name("../flext-core"), eq="flext-core")

    def test_extract_dep_name_with_slash(self) -> None:
        tm.that(extract_dep_name("/flext-core"), eq="flext-core")

    def test_extract_dep_name_with_whitespace(self) -> None:
        tm.that(extract_dep_name("  flext-core  "), eq="flext-core")

    def test_extract_dep_name_with_dot_prefix(self) -> None:
        tm.that(extract_dep_name("./flext-core"), eq="flext-core")

    def test_extract_dep_name_complex(self) -> None:
        tm.that(extract_dep_name(".flext-deps/flext-core"), eq="flext-core")

    def test_extract_dep_name_parent_and_slash(self) -> None:
        tm.that(extract_dep_name("/../flext-core"), eq="flext-core")

    def test_extract_dep_name_with_empty_string(self) -> None:
        tm.that(extract_dep_name(""), eq="")


class TestTargetPath:
    def test_target_path_workspace_root(self) -> None:
        tm.that(
            _target_path("flext-core", is_root=True, mode="workspace"), eq="flext-core"
        )

    def test_target_path_workspace_subproject(self) -> None:
        tm.that(
            _target_path("flext-core", is_root=False, mode="workspace"),
            eq="../flext-core",
        )

    def test_target_path_standalone_root(self) -> None:
        tm.that(
            _target_path("flext-core", is_root=True, mode="standalone"),
            eq=".flext-deps/flext-core",
        )

    def test_target_path_standalone_subproject(self) -> None:
        tm.that(
            _target_path("flext-core", is_root=False, mode="standalone"),
            eq=".flext-deps/flext-core",
        )


class TestExtractRequirementName:
    def test_extract_requirement_name_pep621_path(self) -> None:
        tm.that(
            _extract_requirement_name("flext-core @ file://.flext-deps/flext-core"),
            eq="flext-core",
        )

    def test_extract_requirement_name_simple(self) -> None:
        tm.that(_extract_requirement_name("flext-core"), eq="flext-core")

    def test_extract_requirement_name_with_version(self) -> None:
        tm.that(_extract_requirement_name("flext-core>=1.0.0"), eq="flext-core")

    def test_extract_requirement_name_invalid(self) -> None:
        tm.that(_extract_requirement_name("@invalid"), eq=None)

    def test_extract_requirement_name_empty(self) -> None:
        tm.that(_extract_requirement_name(""), eq=None)

    def test_extract_requirement_name_with_marker(self) -> None:
        tm.that(
            _extract_requirement_name(
                'flext-core @ file://.flext-deps/flext-core ; python_version >= "3.8"',
            ),
            eq="flext-core",
        )


def test_extract_requirement_name_with_path_dep() -> None:
    tm.that(
        _extract_requirement_name("flext-core @ file:../flext-core"), eq="flext-core"
    )


def test_extract_requirement_name_simple() -> None:
    tm.that(_extract_requirement_name("requests>=2.0"), eq="requests")


def test_extract_requirement_name_invalid() -> None:
    tm.that(_extract_requirement_name(""), eq=None)


def test_target_path_workspace_root() -> None:
    tm.that(_target_path("flext-core", is_root=True, mode="workspace"), eq="flext-core")


def test_target_path_workspace_subproject() -> None:
    tm.that(
        _target_path("flext-core", is_root=False, mode="workspace"), eq="../flext-core"
    )


def test_target_path_standalone() -> None:
    tm.that(
        _target_path("flext-core", is_root=False, mode="standalone"),
        eq=".flext-deps/flext-core",
    )


def test_helpers_alias_is_reachable_helpers() -> None:
    tm.fail(r[bool].fail("x"), has="x")
