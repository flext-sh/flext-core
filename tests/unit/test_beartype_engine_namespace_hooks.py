"""Beartype enforcement namespace hook tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests.unit._beartype_engine_support import (
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineNamespaceHooks(TestsFlextBeartypeEngine):
    def test_enforce_054_ignores_tests_init_modules(self, tmp_path: Path) -> None:
        """ENFORCE-054 must ignore legitimate wrapper-root ``tests/__init__.py`` files."""
        package_dir = tmp_path / "hookprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text(
            textwrap.dedent(
                """
                class c:
                    class Core:
                        class Tests:
                            ERR_OK_FAILED = "ok"


                class Probe:
                    value = c.Core.Tests.ERR_OK_FAILED
                """,
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                from hookprobe.tests import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_core_tests_namespace"),
                    Probe,
                )))
                """,
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout

    def test_enforce_054_ignores_string_literals_in_test_modules(
        self,
        tmp_path: Path,
    ) -> None:
        """ENFORCE-054 must ignore string literals that only mention ``.Core.Tests``."""
        package_dir = tmp_path / "stringprobe"
        tests_dir = package_dir / "tests"
        tests_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "sample.py").write_text(
            textwrap.dedent(
                """
                class Probe:
                    value = "c.Core.Tests.ERR_OK_FAILED"
                """,
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = self._run_python(
            textwrap.dedent(
                f"""
                import sys

                sys.path.insert(0, {str(tmp_path)!r})
                from stringprobe.tests.sample import Probe
                from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be

                from flext_core._constants.enforcement import FlextConstantsEnforcement as c
                from flext_core._models.enforcement import FlextModelsEnforcement as me
                print(repr(be.apply(
                    c.EnforcementPredicateKind.DEPRECATED_SYNTAX,
                    me.DeprecatedSyntaxParams(ast_shape="no_core_tests_namespace"),
                    Probe,
                )))
                """,
            ),
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.exit_code == 0, result.stderr
        assert "None" in result.stdout
