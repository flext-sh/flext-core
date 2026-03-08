"""Tests for flext_infra.constants — infrastructure constants."""

from __future__ import annotations

from flext_infra.constants import FlextInfraConstants, c


class TestFlextInfraConstantsPathsNamespace:
    """Tests for Paths namespace constants."""

    def test_venv_bin_rel_constant(self) -> None:
        assert FlextInfraConstants.Infra.Paths.VENV_BIN_REL == ".venv/bin"

    def test_default_src_dir_constant(self) -> None:
        assert FlextInfraConstants.Infra.Paths.DEFAULT_SRC_DIR == "src"

    def test_paths_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Paths.VENV_BIN_REL, str)
        assert isinstance(FlextInfraConstants.Infra.Paths.DEFAULT_SRC_DIR, str)


class TestFlextInfraConstantsFilesNamespace:
    """Tests for Files namespace constants."""

    def test_pyproject_filename_constant(self) -> None:
        assert FlextInfraConstants.Infra.Files.PYPROJECT_FILENAME == "pyproject.toml"

    def test_makefile_filename_constant(self) -> None:
        assert FlextInfraConstants.Infra.Files.MAKEFILE_FILENAME == "Makefile"

    def test_base_mk_constant(self) -> None:
        assert FlextInfraConstants.Infra.Files.BASE_MK == "base.mk"

    def test_go_mod_constant(self) -> None:
        assert FlextInfraConstants.Infra.Files.GO_MOD == "go.mod"

    def test_files_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Files.PYPROJECT_FILENAME, str)
        assert isinstance(FlextInfraConstants.Infra.Files.MAKEFILE_FILENAME, str)
        assert isinstance(FlextInfraConstants.Infra.Files.BASE_MK, str)
        assert isinstance(FlextInfraConstants.Infra.Files.GO_MOD, str)


class TestFlextInfraConstantsGatesNamespace:
    """Tests for Gates namespace constants."""

    def test_gate_constants_exist(self) -> None:
        assert FlextInfraConstants.Infra.Gates.LINT == "lint"
        assert FlextInfraConstants.Infra.Gates.FORMAT == "format"
        assert FlextInfraConstants.Infra.Gates.PYREFLY == "pyrefly"
        assert FlextInfraConstants.Infra.Gates.MYPY == "mypy"
        assert FlextInfraConstants.Infra.Gates.PYRIGHT == "pyright"
        assert FlextInfraConstants.Infra.Gates.SECURITY == "security"
        assert FlextInfraConstants.Infra.Gates.MARKDOWN == "markdown"
        assert FlextInfraConstants.Infra.Gates.GO == "go"

    def test_type_alias_gate(self) -> None:
        assert FlextInfraConstants.Infra.Gates.TYPE_ALIAS == "type"

    def test_default_csv_contains_gates(self) -> None:
        csv = FlextInfraConstants.Infra.Gates.DEFAULT_CSV
        assert "lint" in csv
        assert "format" in csv
        assert "mypy" in csv
        assert "pyright" in csv

    def test_default_csv_is_comma_separated(self) -> None:
        csv = FlextInfraConstants.Infra.Gates.DEFAULT_CSV
        gates = csv.split(",")
        assert len(gates) > 0
        assert all(isinstance(g, str) for g in gates)


class TestFlextInfraConstantsStatusNamespace:
    """Tests for Status namespace constants."""

    def test_pass_status_constant(self) -> None:
        assert FlextInfraConstants.Infra.Status.PASS == "PASS"

    def test_fail_status_constant(self) -> None:
        assert FlextInfraConstants.Infra.Status.FAIL == "FAIL"

    def test_ok_status_constant(self) -> None:
        assert FlextInfraConstants.Infra.Status.OK == "OK"

    def test_warn_status_constant(self) -> None:
        assert FlextInfraConstants.Infra.Status.WARN == "WARN"

    def test_status_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Status.PASS, str)
        assert isinstance(FlextInfraConstants.Infra.Status.FAIL, str)
        assert isinstance(FlextInfraConstants.Infra.Status.OK, str)
        assert isinstance(FlextInfraConstants.Infra.Status.WARN, str)


class TestFlextInfraConstantsExcludedNamespace:
    """Tests for Excluded namespace constants."""

    def test_common_excluded_dirs_is_frozenset(self) -> None:
        assert isinstance(
            FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS, frozenset,
        )

    def test_common_excluded_dirs_contains_standard_dirs(self) -> None:
        excluded = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        assert ".git" in excluded
        assert ".venv" in excluded
        assert "__pycache__" in excluded
        assert "dist" in excluded
        assert "build" in excluded

    def test_doc_excluded_dirs_includes_common(self) -> None:
        doc_excluded = FlextInfraConstants.Infra.Excluded.DOC_EXCLUDED_DIRS
        common = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(doc_excluded)

    def test_doc_excluded_dirs_includes_site(self) -> None:
        assert "site" in FlextInfraConstants.Infra.Excluded.DOC_EXCLUDED_DIRS

    def test_pyproject_skip_dirs_includes_common(self) -> None:
        skip_dirs = FlextInfraConstants.Infra.Excluded.PYPROJECT_SKIP_DIRS
        common = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(skip_dirs)

    def test_pyproject_skip_dirs_includes_flext_dirs(self) -> None:
        skip_dirs = FlextInfraConstants.Infra.Excluded.PYPROJECT_SKIP_DIRS
        assert ".flext-deps" in skip_dirs
        assert ".sisyphus" in skip_dirs

    def test_check_excluded_dirs_includes_common(self) -> None:
        check_excluded = FlextInfraConstants.Infra.Excluded.CHECK_EXCLUDED_DIRS
        common = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(check_excluded)

    def test_check_excluded_dirs_includes_flext_deps(self) -> None:
        assert ".flext-deps" in FlextInfraConstants.Infra.Excluded.CHECK_EXCLUDED_DIRS

    def test_excluded_dirs_are_frozensets(self) -> None:
        assert isinstance(
            FlextInfraConstants.Infra.Excluded.DOC_EXCLUDED_DIRS, frozenset,
        )
        assert isinstance(
            FlextInfraConstants.Infra.Excluded.PYPROJECT_SKIP_DIRS, frozenset,
        )
        assert isinstance(
            FlextInfraConstants.Infra.Excluded.CHECK_EXCLUDED_DIRS, frozenset,
        )


class TestFlextInfraConstantsCheckNamespace:
    """Tests for Check namespace constants."""

    def test_default_check_dirs_is_tuple(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Check.DEFAULT_CHECK_DIRS, tuple)

    def test_default_check_dirs_contains_standard_dirs(self) -> None:
        dirs = FlextInfraConstants.Infra.Check.DEFAULT_CHECK_DIRS
        assert "src" in dirs
        assert "tests" in dirs
        assert "examples" in dirs
        assert "scripts" in dirs

    def test_check_dirs_subproject_is_tuple(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Check.CHECK_DIRS_SUBPROJECT, tuple)

    def test_check_dirs_subproject_excludes_scripts(self) -> None:
        dirs = FlextInfraConstants.Infra.Check.CHECK_DIRS_SUBPROJECT
        assert "src" in dirs
        assert "tests" in dirs
        assert "examples" in dirs
        assert "scripts" not in dirs

    def test_check_dirs_are_strings(self) -> None:
        for d in FlextInfraConstants.Infra.Check.DEFAULT_CHECK_DIRS:
            assert isinstance(d, str)
        for d in FlextInfraConstants.Infra.Check.CHECK_DIRS_SUBPROJECT:
            assert isinstance(d, str)


class TestFlextInfraConstantsGithubNamespace:
    """Tests for Github namespace constants."""

    def test_github_repo_url_constant(self) -> None:
        assert (
            FlextInfraConstants.Infra.Github.GITHUB_REPO_URL
            == "https://github.com/flext-sh/flext"
        )

    def test_github_repo_name_constant(self) -> None:
        assert FlextInfraConstants.Infra.Github.GITHUB_REPO_NAME == "flext-sh/flext"

    def test_github_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Github.GITHUB_REPO_URL, str)
        assert isinstance(FlextInfraConstants.Infra.Github.GITHUB_REPO_NAME, str)


class TestFlextInfraConstantsEncodingNamespace:
    """Tests for Encoding namespace constants."""

    def test_default_encoding_constant(self) -> None:
        assert FlextInfraConstants.Infra.Encoding.DEFAULT == "utf-8"

    def test_encoding_constant_is_string(self) -> None:
        assert isinstance(FlextInfraConstants.Infra.Encoding.DEFAULT, str)


class TestFlextInfraConstantsAlias:
    """Tests for module-level alias."""

    def test_c_alias_points_to_class(self) -> None:
        assert c is FlextInfraConstants

    def test_c_alias_provides_access_to_namespaces(self) -> None:
        assert hasattr(c, "Infra")
        assert hasattr(c.Infra, "Paths")
        assert hasattr(c.Infra, "Files")
        assert hasattr(c.Infra, "Gates")
        assert hasattr(c.Infra, "Status")
        assert hasattr(c.Infra, "Excluded")
        assert hasattr(c.Infra, "Check")
        assert hasattr(c.Infra, "Github")
        assert hasattr(c.Infra, "Encoding")

    def test_c_alias_access_to_constants(self) -> None:
        assert c.Infra.Paths.VENV_BIN_REL == ".venv/bin"
        assert c.Infra.Status.PASS == "PASS"
        assert c.Infra.Files.PYPROJECT_FILENAME == "pyproject.toml"


class TestFlextInfraConstantsImmutability:
    """Tests for constant immutability."""

    def test_excluded_dirs_are_immutable(self) -> None:
        excluded = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        assert not hasattr(excluded, "add")

    def test_check_dirs_are_immutable(self) -> None:
        dirs = FlextInfraConstants.Infra.Check.DEFAULT_CHECK_DIRS
        assert not hasattr(dirs, "append")


class TestFlextInfraConstantsConsistency:
    """Tests for consistency across namespaces."""

    def test_all_status_values_are_uppercase(self) -> None:
        assert FlextInfraConstants.Infra.Status.PASS.isupper()
        assert FlextInfraConstants.Infra.Status.FAIL.isupper()
        assert FlextInfraConstants.Infra.Status.OK.isupper()
        assert FlextInfraConstants.Infra.Status.WARN.isupper()

    def test_all_gate_values_are_lowercase(self) -> None:
        gates = [
            FlextInfraConstants.Infra.Gates.LINT,
            FlextInfraConstants.Infra.Gates.FORMAT,
            FlextInfraConstants.Infra.Gates.PYREFLY,
            FlextInfraConstants.Infra.Gates.MYPY,
            FlextInfraConstants.Infra.Gates.PYRIGHT,
            FlextInfraConstants.Infra.Gates.SECURITY,
            FlextInfraConstants.Infra.Gates.MARKDOWN,
            FlextInfraConstants.Infra.Gates.GO,
        ]
        for gate in gates:
            assert gate.islower(), f"Gate {gate} should be lowercase"

    def test_excluded_dirs_no_duplicates(self) -> None:
        common = FlextInfraConstants.Infra.Excluded.COMMON_EXCLUDED_DIRS
        doc = FlextInfraConstants.Infra.Excluded.DOC_EXCLUDED_DIRS
        assert len(common) == len(set(common))
        assert len(doc) == len(set(doc))
