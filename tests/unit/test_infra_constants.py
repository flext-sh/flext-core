"""Tests for flext_infra.constants — infrastructure constants."""

from __future__ import annotations

from flext_infra.constants import FlextInfraConstants, c


class TestFlextInfraConstantsPathsNamespace:
    """Tests for Paths namespace constants."""

    def test_venv_bin_rel_constant(self) -> None:
        assert FlextInfraConstants.Paths.VENV_BIN_REL == ".venv/bin"

    def test_default_src_dir_constant(self) -> None:
        assert FlextInfraConstants.Paths.DEFAULT_SRC_DIR == "src"

    def test_paths_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Paths.VENV_BIN_REL, str)
        assert isinstance(FlextInfraConstants.Paths.DEFAULT_SRC_DIR, str)


class TestFlextInfraConstantsFilesNamespace:
    """Tests for Files namespace constants."""

    def test_pyproject_filename_constant(self) -> None:
        assert FlextInfraConstants.Files.PYPROJECT_FILENAME == "pyproject.toml"

    def test_makefile_filename_constant(self) -> None:
        assert FlextInfraConstants.Files.MAKEFILE_FILENAME == "Makefile"

    def test_base_mk_constant(self) -> None:
        assert FlextInfraConstants.Files.BASE_MK == "base.mk"

    def test_go_mod_constant(self) -> None:
        assert FlextInfraConstants.Files.GO_MOD == "go.mod"

    def test_files_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Files.PYPROJECT_FILENAME, str)
        assert isinstance(FlextInfraConstants.Files.MAKEFILE_FILENAME, str)
        assert isinstance(FlextInfraConstants.Files.BASE_MK, str)
        assert isinstance(FlextInfraConstants.Files.GO_MOD, str)


class TestFlextInfraConstantsGatesNamespace:
    """Tests for Gates namespace constants."""

    def test_gate_constants_exist(self) -> None:
        assert FlextInfraConstants.Gates.LINT == "lint"
        assert FlextInfraConstants.Gates.FORMAT == "format"
        assert FlextInfraConstants.Gates.PYREFLY == "pyrefly"
        assert FlextInfraConstants.Gates.MYPY == "mypy"
        assert FlextInfraConstants.Gates.PYRIGHT == "pyright"
        assert FlextInfraConstants.Gates.SECURITY == "security"
        assert FlextInfraConstants.Gates.MARKDOWN == "markdown"
        assert FlextInfraConstants.Gates.GO == "go"

    def test_type_alias_gate(self) -> None:
        assert FlextInfraConstants.Gates.TYPE_ALIAS == "type"

    def test_default_csv_contains_gates(self) -> None:
        csv = FlextInfraConstants.Gates.DEFAULT_CSV
        assert "lint" in csv
        assert "format" in csv
        assert "mypy" in csv
        assert "pyright" in csv

    def test_default_csv_is_comma_separated(self) -> None:
        csv = FlextInfraConstants.Gates.DEFAULT_CSV
        gates = csv.split(",")
        assert len(gates) > 0
        assert all(isinstance(g, str) for g in gates)


class TestFlextInfraConstantsStatusNamespace:
    """Tests for Status namespace constants."""

    def test_pass_status_constant(self) -> None:
        assert FlextInfraConstants.Status.PASS == "PASS"

    def test_fail_status_constant(self) -> None:
        assert FlextInfraConstants.Status.FAIL == "FAIL"

    def test_ok_status_constant(self) -> None:
        assert FlextInfraConstants.Status.OK == "OK"

    def test_warn_status_constant(self) -> None:
        assert FlextInfraConstants.Status.WARN == "WARN"

    def test_status_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Status.PASS, str)
        assert isinstance(FlextInfraConstants.Status.FAIL, str)
        assert isinstance(FlextInfraConstants.Status.OK, str)
        assert isinstance(FlextInfraConstants.Status.WARN, str)


class TestFlextInfraConstantsExcludedNamespace:
    """Tests for Excluded namespace constants."""

    def test_common_excluded_dirs_is_frozenset(self) -> None:
        assert isinstance(FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS, frozenset)

    def test_common_excluded_dirs_contains_standard_dirs(self) -> None:
        excluded = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        assert ".git" in excluded
        assert ".venv" in excluded
        assert "__pycache__" in excluded
        assert "dist" in excluded
        assert "build" in excluded

    def test_doc_excluded_dirs_includes_common(self) -> None:
        doc_excluded = FlextInfraConstants.Excluded.DOC_EXCLUDED_DIRS
        common = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(doc_excluded)

    def test_doc_excluded_dirs_includes_site(self) -> None:
        assert "site" in FlextInfraConstants.Excluded.DOC_EXCLUDED_DIRS

    def test_pyproject_skip_dirs_includes_common(self) -> None:
        skip_dirs = FlextInfraConstants.Excluded.PYPROJECT_SKIP_DIRS
        common = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(skip_dirs)

    def test_pyproject_skip_dirs_includes_flext_dirs(self) -> None:
        skip_dirs = FlextInfraConstants.Excluded.PYPROJECT_SKIP_DIRS
        assert ".flext-deps" in skip_dirs
        assert ".sisyphus" in skip_dirs

    def test_check_excluded_dirs_includes_common(self) -> None:
        check_excluded = FlextInfraConstants.Excluded.CHECK_EXCLUDED_DIRS
        common = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        assert common.issubset(check_excluded)

    def test_check_excluded_dirs_includes_flext_deps(self) -> None:
        assert ".flext-deps" in FlextInfraConstants.Excluded.CHECK_EXCLUDED_DIRS

    def test_excluded_dirs_are_frozensets(self) -> None:
        assert isinstance(FlextInfraConstants.Excluded.DOC_EXCLUDED_DIRS, frozenset)
        assert isinstance(FlextInfraConstants.Excluded.PYPROJECT_SKIP_DIRS, frozenset)
        assert isinstance(FlextInfraConstants.Excluded.CHECK_EXCLUDED_DIRS, frozenset)


class TestFlextInfraConstantsCheckNamespace:
    """Tests for Check namespace constants."""

    def test_default_check_dirs_is_tuple(self) -> None:
        assert isinstance(FlextInfraConstants.Check.DEFAULT_CHECK_DIRS, tuple)

    def test_default_check_dirs_contains_standard_dirs(self) -> None:
        dirs = FlextInfraConstants.Check.DEFAULT_CHECK_DIRS
        assert "src" in dirs
        assert "tests" in dirs
        assert "examples" in dirs
        assert "scripts" in dirs

    def test_check_dirs_subproject_is_tuple(self) -> None:
        assert isinstance(FlextInfraConstants.Check.CHECK_DIRS_SUBPROJECT, tuple)

    def test_check_dirs_subproject_excludes_scripts(self) -> None:
        dirs = FlextInfraConstants.Check.CHECK_DIRS_SUBPROJECT
        assert "src" in dirs
        assert "tests" in dirs
        assert "examples" in dirs
        assert "scripts" not in dirs

    def test_check_dirs_are_strings(self) -> None:
        for d in FlextInfraConstants.Check.DEFAULT_CHECK_DIRS:
            assert isinstance(d, str)
        for d in FlextInfraConstants.Check.CHECK_DIRS_SUBPROJECT:
            assert isinstance(d, str)


class TestFlextInfraConstantsGithubNamespace:
    """Tests for Github namespace constants."""

    def test_github_repo_url_constant(self) -> None:
        assert (
            FlextInfraConstants.Github.GITHUB_REPO_URL
            == "https://github.com/flext-sh/flext"
        )

    def test_github_repo_name_constant(self) -> None:
        assert FlextInfraConstants.Github.GITHUB_REPO_NAME == "flext-sh/flext"

    def test_github_constants_are_strings(self) -> None:
        assert isinstance(FlextInfraConstants.Github.GITHUB_REPO_URL, str)
        assert isinstance(FlextInfraConstants.Github.GITHUB_REPO_NAME, str)


class TestFlextInfraConstantsEncodingNamespace:
    """Tests for Encoding namespace constants."""

    def test_default_encoding_constant(self) -> None:
        assert FlextInfraConstants.Encoding.DEFAULT == "utf-8"

    def test_encoding_constant_is_string(self) -> None:
        assert isinstance(FlextInfraConstants.Encoding.DEFAULT, str)


class TestFlextInfraConstantsAlias:
    """Tests for module-level alias."""

    def test_c_alias_points_to_class(self) -> None:
        assert c is FlextInfraConstants

    def test_c_alias_provides_access_to_namespaces(self) -> None:
        assert hasattr(c, "Paths")
        assert hasattr(c, "Files")
        assert hasattr(c, "Gates")
        assert hasattr(c, "Status")
        assert hasattr(c, "Excluded")
        assert hasattr(c, "Check")
        assert hasattr(c, "Github")
        assert hasattr(c, "Encoding")

    def test_c_alias_access_to_constants(self) -> None:
        assert c.Paths.VENV_BIN_REL == ".venv/bin"
        assert c.Status.PASS == "PASS"
        assert c.Files.PYPROJECT_FILENAME == "pyproject.toml"


class TestFlextInfraConstantsImmutability:
    """Tests for constant immutability."""

    def test_excluded_dirs_are_immutable(self) -> None:
        excluded = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        # frozenset should not have add method
        assert not hasattr(excluded, "add")

    def test_check_dirs_are_immutable(self) -> None:
        dirs = FlextInfraConstants.Check.DEFAULT_CHECK_DIRS
        # tuple should not have append method
        assert not hasattr(dirs, "append")


class TestFlextInfraConstantsConsistency:
    """Tests for consistency across namespaces."""

    def test_all_status_values_are_uppercase(self) -> None:
        assert FlextInfraConstants.Status.PASS.isupper()
        assert FlextInfraConstants.Status.FAIL.isupper()
        assert FlextInfraConstants.Status.OK.isupper()
        assert FlextInfraConstants.Status.WARN.isupper()

    def test_all_gate_values_are_lowercase(self) -> None:
        gates = [
            FlextInfraConstants.Gates.LINT,
            FlextInfraConstants.Gates.FORMAT,
            FlextInfraConstants.Gates.PYREFLY,
            FlextInfraConstants.Gates.MYPY,
            FlextInfraConstants.Gates.PYRIGHT,
            FlextInfraConstants.Gates.SECURITY,
            FlextInfraConstants.Gates.MARKDOWN,
            FlextInfraConstants.Gates.GO,
        ]
        for gate in gates:
            assert gate.islower(), f"Gate {gate} should be lowercase"

    def test_excluded_dirs_no_duplicates(self) -> None:
        common = FlextInfraConstants.Excluded.COMMON_EXCLUDED_DIRS
        doc = FlextInfraConstants.Excluded.DOC_EXCLUDED_DIRS
        # frozenset automatically handles uniqueness
        assert len(common) == len(set(common))
        assert len(doc) == len(set(doc))
