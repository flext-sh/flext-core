"""Tests for flext_infra.constants — Check, Github, Encoding, alias, and consistency.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import c as infra_c
from flext_tests import tm


class TestFlextInfraConstantsCheckNamespace:
    """Tests for Check namespace constants."""

    def test_default_check_dirs_is_list(self) -> None:
        tm.that(infra_c.Infra.Check.DEFAULT_CHECK_DIRS, is_=list)

    def test_default_check_dirs_contains_standard_dirs(self) -> None:
        dirs = infra_c.Infra.Check.DEFAULT_CHECK_DIRS
        tm.that(dirs, contains="src")
        tm.that(dirs, contains="tests")
        tm.that(dirs, contains="examples")
        tm.that(dirs, contains="scripts")

    def test_check_dirs_subproject_is_list(self) -> None:
        tm.that(infra_c.Infra.Check.CHECK_DIRS_SUBPROJECT, is_=list)

    def test_check_dirs_subproject_excludes_scripts(self) -> None:
        dirs = infra_c.Infra.Check.CHECK_DIRS_SUBPROJECT
        tm.that(dirs, contains="src")
        tm.that(dirs, contains="tests")
        tm.that(dirs, contains="examples")
        assert "scripts" not in dirs

    def test_check_dirs_are_strings(self) -> None:
        for d in infra_c.Infra.Check.DEFAULT_CHECK_DIRS:
            tm.that(d, is_=str)
        for d in infra_c.Infra.Check.CHECK_DIRS_SUBPROJECT:
            tm.that(d, is_=str)


class TestFlextInfraConstantsGithubNamespace:
    """Tests for Github namespace constants."""

    def test_github_repo_url_constant(self) -> None:
        tm.that(
            infra_c.Infra.Github.GITHUB_REPO_URL, eq="https://github.com/flext-sh/flext"
        )

    def test_github_repo_name_constant(self) -> None:
        tm.that(infra_c.Infra.Github.GITHUB_REPO_NAME, eq="flext-sh/flext")

    def test_github_constants_are_strings(self) -> None:
        tm.that(infra_c.Infra.Github.GITHUB_REPO_URL, is_=str)
        tm.that(infra_c.Infra.Github.GITHUB_REPO_NAME, is_=str)


class TestFlextInfraConstantsEncodingNamespace:
    """Tests for Encoding namespace constants."""

    def test_default_encoding_constant(self) -> None:
        tm.that(infra_c.Infra.Encoding.DEFAULT, eq="utf-8")

    def test_encoding_constant_is_string(self) -> None:
        tm.that(infra_c.Infra.Encoding.DEFAULT, is_=str)


class TestFlextInfraConstantsAlias:
    """Tests for module-level alias."""

    def test_c_alias_is_string(self) -> None:
        tm.that(infra_c, is_=str)

    def test_c_alias_provides_access_to_namespaces(self) -> None:
        tm.that(hasattr(infra_c, "Infra"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Paths"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Files"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Gates"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Status"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Excluded"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Check"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Github"), eq=True)
        tm.that(hasattr(infra_c.Infra, "Encoding"), eq=True)

    def test_c_alias_access_to_constants(self) -> None:
        tm.that(infra_c.Infra.Paths.VENV_BIN_REL, eq=".venv/bin")
        tm.that(infra_c.Infra.Status.PASS, eq="PASS")
        tm.that(infra_c.Infra.Files.PYPROJECT_FILENAME, eq="pyproject.toml")


class TestFlextInfraConstantsImmutability:
    """Tests for constant immutability."""

    def test_excluded_dirs_are_immutable(self) -> None:
        excluded = infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS
        tm.that(hasattr(excluded, "add"), eq=False)

    def test_check_dirs_are_immutable(self) -> None:
        dirs = infra_c.Infra.Check.DEFAULT_CHECK_DIRS
        tm.that(hasattr(dirs, "append"), eq=False)


class TestFlextInfraConstantsConsistency:
    """Tests for consistency across namespaces."""

    def test_all_status_values_are_uppercase(self) -> None:
        tm.that(infra_c.Infra.Status.PASS.isupper(), eq=True)
        tm.that(infra_c.Infra.Status.FAIL.isupper(), eq=True)
        tm.that(infra_c.Infra.Status.OK.isupper(), eq=True)
        tm.that(infra_c.Infra.Status.WARN.isupper(), eq=True)

    def test_all_gate_values_are_lowercase(self) -> None:
        gates = [
            infra_c.Infra.Gates.LINT,
            infra_c.Infra.Gates.FORMAT,
            infra_c.Infra.Gates.PYREFLY,
            infra_c.Infra.Gates.MYPY,
            infra_c.Infra.Gates.PYRIGHT,
            infra_c.Infra.Gates.SECURITY,
            infra_c.Infra.Gates.MARKDOWN,
            infra_c.Infra.Gates.GO,
        ]
        for gate in gates:
            tm.that(gate.islower(), eq=True, msg=f"Gate {gate} should be lowercase")

    def test_excluded_dirs_no_duplicates(self) -> None:
        common = infra_c.Infra.Excluded.COMMON_EXCLUDED_DIRS
        doc = infra_c.Infra.Excluded.DOC_EXCLUDED_DIRS
        tm.that(len(common), eq=len(set(common)))
        tm.that(len(doc), eq=len(set(doc)))
