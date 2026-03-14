"""Tests for flext_infra.constants — core namespace constants.

Tests cover Paths, Files, Gates, Status, and Excluded namespaces.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_infra import c as infra_c
from flext_tests import t, tm


class TestFlextInfraConstantsPathsNamespace:
    """Tests for Paths namespace constants."""

    def test_venv_bin_rel_constant(self) -> None:
        tm.that(infra_c.Infra.Paths.VENV_BIN_REL, eq=".venv/bin")

    def test_default_src_dir_constant(self) -> None:
        tm.that(infra_c.Infra.Paths.DEFAULT_SRC_DIR, eq="src")

    def test_paths_constants_are_strings(self) -> None:
        tm.that(infra_c.Infra.Paths.VENV_BIN_REL, is_=str)
        tm.that(infra_c.Infra.Paths.DEFAULT_SRC_DIR, is_=str)


class TestFlextInfraConstantsFilesNamespace:
    """Tests for Files namespace constants."""

    def test_pyproject_filename_constant(self) -> None:
        tm.that(infra_c.Infra.Files.PYPROJECT_FILENAME, eq="pyproject.toml")

    def test_makefile_filename_constant(self) -> None:
        tm.that(infra_c.Infra.Files.MAKEFILE_FILENAME, eq="Makefile")

    def test_base_mk_constant(self) -> None:
        tm.that(infra_c.Infra.Files.BASE_MK, eq="base.mk")

    def test_go_mod_constant(self) -> None:
        tm.that(infra_c.Infra.Files.GO_MOD, eq="go.mod")

    def test_files_constants_are_strings(self) -> None:
        tm.that(infra_c.Infra.Files.PYPROJECT_FILENAME, is_=str)
        tm.that(infra_c.Infra.Files.MAKEFILE_FILENAME, is_=str)
        tm.that(infra_c.Infra.Files.BASE_MK, is_=str)
        tm.that(infra_c.Infra.Files.GO_MOD, is_=str)


class TestFlextInfraConstantsGatesNamespace:
    """Tests for Gates namespace constants."""

    def test_gate_constants_exist(self) -> None:
        tm.that(infra_c.Infra.Gates.LINT, eq="lint")
        tm.that(infra_c.Infra.Gates.FORMAT, eq="format")
        tm.that(infra_c.Infra.Gates.PYREFLY, eq="pyrefly")
        tm.that(infra_c.Infra.Gates.MYPY, eq="mypy")
        tm.that(infra_c.Infra.Gates.PYRIGHT, eq="pyright")
        tm.that(infra_c.Infra.Gates.SECURITY, eq="security")
        tm.that(infra_c.Infra.Gates.MARKDOWN, eq="markdown")
        tm.that(infra_c.Infra.Gates.GO, eq="go")

    def test_type_alias_gate(self) -> None:
        tm.that(infra_c.Infra.Gates.TYPE_ALIAS, eq="type")

    def test_default_csv_contains_gates(self) -> None:
        csv = infra_c.Infra.Gates.DEFAULT_CSV
        tm.that(csv, contains="lint")
        tm.that(csv, contains="format")
        tm.that(csv, contains="mypy")
        tm.that(csv, contains="pyright")

    def test_default_csv_is_comma_separated(self) -> None:
        csv = infra_c.Infra.Gates.DEFAULT_CSV
        gates = csv.split(",")
        tm.that(gates, length_gt=0)
        for g in gates:
            tm.that(g, is_=str)


class TestFlextInfraConstantsStatusNamespace:
    """Tests for Status namespace constants."""

    def test_pass_status_constant(self) -> None:
        tm.that(infra_c.Infra.Status.PASS, eq="PASS")

    def test_fail_status_constant(self) -> None:
        tm.that(infra_c.Infra.Status.FAIL, eq="FAIL")

    def test_ok_status_constant(self) -> None:
        tm.that(infra_c.Infra.Status.OK, eq="OK")

    def test_warn_status_constant(self) -> None:
        tm.that(infra_c.Infra.Status.WARN, eq="WARN")

    def test_status_constants_are_strings(self) -> None:
        tm.that(infra_c.Infra.Status.PASS, is_=str)
        tm.that(infra_c.Infra.Status.FAIL, is_=str)
        tm.that(infra_c.Infra.Status.OK, is_=str)
        tm.that(infra_c.Infra.Status.WARN, is_=str)


class TestFlextInfraConstantsExcludedNamespace:
    """Tests for Excluded namespace constants."""

    def test_common_excluded_dirs_is_string(self) -> None:
        excluded = cast(
            "t.Tests.Matcher.MatcherKwargValue",
            infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS,
        )
        tm.that(excluded, is_=str)

    def test_common_excluded_dirs_contains_standard_dirs(self) -> None:
        excluded = cast(
            "t.Tests.Matcher.MatcherKwargValue",
            infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS,
        )
        tm.that(excluded, contains=".git")
        tm.that(excluded, contains=".venv")
        tm.that(excluded, contains="__pycache__")
        tm.that(excluded, contains="dist")
        tm.that(excluded, contains="build")

    def test_doc_excluded_dirs_includes_common(self) -> None:
        doc_excluded = infra_c.Infra.Excluded.DOC_EXCLUDED_DIRS
        common = infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS
        tm.that(doc_excluded.issuperset(common), eq=True)

    def test_doc_excluded_dirs_includes_site(self) -> None:
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                infra_c.Infra.Excluded.DOC_EXCLUDED_DIRS,
            ),
            contains="site",
        )

    def test_pyproject_skip_dirs_includes_common(self) -> None:
        skip_dirs = infra_c.Infra.Excluded.PYPROJECT_SKIP_DIRS
        common = infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS
        tm.that(skip_dirs.issuperset(common), eq=True)

    def test_pyproject_skip_dirs_includes_flext_dirs(self) -> None:
        skip_dirs = cast(
            "t.Tests.Matcher.MatcherKwargValue",
            infra_c.Infra.Excluded.PYPROJECT_SKIP_DIRS,
        )
        tm.that(skip_dirs, contains=".flext-deps")
        tm.that(skip_dirs, contains=".sisyphus")

    def test_check_excluded_dirs_includes_common(self) -> None:
        check_excluded = infra_c.Infra.Excluded.CHECK_EXCLUDED_DIRS
        common = infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS
        tm.that(check_excluded.issuperset(common), eq=True)

    def test_check_excluded_dirs_includes_flext_deps(self) -> None:
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                infra_c.Infra.Excluded.CHECK_EXCLUDED_DIRS,
            ),
            contains=".flext-deps",
        )

    def test_excluded_dirs_are_strings(self) -> None:
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                infra_c.Infra.Excluded.DOC_EXCLUDED_DIRS,
            ),
            is_=str,
        )
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                infra_c.Infra.Excluded.PYPROJECT_SKIP_DIRS,
            ),
            is_=str,
        )
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                infra_c.Infra.Excluded.CHECK_EXCLUDED_DIRS,
            ),
            is_=str,
        )
