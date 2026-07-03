"""Beartype enforcement import hook tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests.unit._beartype_engine_support import (
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineImportHooks(TestsFlextBeartypeEngine):
    def test_enforce_055_detects_wrapper_submodule_alias_import(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-055 must detect alias imports from wrapper submodules."""
        package_dir = tmp_path / "importprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                from tests.constants import c


                class Probe:
                    value = c
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                from importprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_wrapper_root_alias_import"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "from tests.constants import c" in result.stdout

    def test_enforce_055_ignores_string_literals_in_test_modules(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-055 must ignore string literals that only mention wrapper imports."""
        package_dir = tmp_path / "importstringprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                class Probe:
                    value = "from tests.constants import c"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                from importstringprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_wrapper_root_alias_import"),
                    Probe,
                )))
                """
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout
